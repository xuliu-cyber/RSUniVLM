import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import gradio as gr
import time
import numpy as np
import re
from PIL import Image  # 用于处理图片
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX
import copy
import torch
import cv2

# load model
model_path = "/data/liux/llava-ov-qwen-0.5b/llava-qwen-gmoe"
model_base = None
model_name = "llava_qwen_gmoe"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, torch_dtype="float16",device_map="auto", **llava_model_args)
model.eval()


def left_pad_sequences(sequences, desired_length, padding_value):
    """
    Pad each sequence in a tuple to the desired length with the specified padding value on the left.

    :param sequences: A tuple of sequences (e.g., lists, tuples).
    :param desired_length: The length to which each sequence will be padded.
    :param padding_value: The value used for padding.
    :return: A new tuple with padded sequences.
    """
    padded_sequences = tuple(
        [padding_value] * (desired_length - len(seq)) + list(seq)
        for seq in sequences
    )
    return padded_sequences


def replace_color(img, src_clr, dst_clr):
    img_arr = np.asarray(img, dtype=np.double)
    
    r_img = img_arr[:,:,0].copy()
    g_img = img_arr[:,:,1].copy()
    b_img = img_arr[:,:,2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2] #编码
    
    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]
    
    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1,2,0)
    
    return dst_img


def parse_output_string(output_string, imgsize, patch_size=20):
    output_string = output_string.lower()
    try:
        # 提取标签
        labels = set(re.findall(r'\b[a-z]+(?: [a-z]+)*\b', output_string))
        
        # 将"others"映射到黑色，另一个标签映射到白色
        label_mapping = {}
        for label in labels:
            if label == "others":
                label_mapping[label] = (0, 0, 0)  # 黑色
            else:
                label_mapping[label] = (255, 255, 255)  # 白色

        rows = output_string.strip().split("\n")
        parsed_mask = []

        for row in rows:
            row_data = []
            patches = row.split(", ")
            for patch in patches:
                label, count = patch.split("*")
                count = int(count)
                row_data.extend([label] * count)
            parsed_mask.append(row_data)
        
        parsed_mask = np.array(parsed_mask)
    
        height, width = parsed_mask.shape

        # 创建彩色图像
        color_image = np.zeros((height * patch_size, width * patch_size, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                color = label_mapping[parsed_mask[i, j].strip()]
                color_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = color
        color_image = cv2.resize(color_image, (imgsize[0], imgsize[1]), interpolation=cv2.INTER_NEAREST)
        return color_image
    except Exception as e:
        print(e)
        return None

def draw_box(text_output, image_pil):
    imagesize=image_pil.size
    pred_match = re.findall(r"\[([0-9., ]+)\]", text_output)
    new_pred_result=[]
    colorsmap=[(0,0,255), (255,0,0), (0,255,0), (255,255,0), (0,255,255), (255,0,255)]
    try:
        pred_result = [list(map(float, match.split(","))) for match in pred_match]
        for pred in pred_result:
            if len(pred) == 4:
                new_pred_result.append(pred)
            elif len(pred) > 4:
                while len(pred) != 4:
                    pred.pop()
                new_pred_result.append(pred)
        # vis
        imgnp = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        for i, box in enumerate(new_pred_result):
            box[0],box[2] = box[0]/100*imagesize[0], box[2]/100*imagesize[0]
            box[1],box[3] = box[1]/100*imagesize[1], box[3]/100*imagesize[1]
            box = list(map(int, box))
            cv2.rectangle(imgnp, (box[0],box[1]),  (box[2],box[3]), color=colorsmap[i], thickness=2)
        return Image.fromarray(cv2.cvtColor(imgnp,cv2.COLOR_BGR2RGB))
    except Exception as e:
        return None
    
# 定义一个函数，用来逐字输出响应消息，并返回图片
def echo(message, image1, image2):
    if image2 is None:
        images=[image1]
    else:
        images=[image1, image2]
    # 格式化响应消息，包含用户消息
    response = f""
    if images:
        images_pil = []
        for imagenp in images:
            images_pil.append(Image.fromarray(imagenp))
        image_tensors = process_images(images_pil, image_processor, model.config)
        image_sizes = [image_pil.size for image_pil in images_pil]
        if len(images)==1:
            question = '<image>\n'+message
        elif len(images)==2:
            question = '<image> <image>\n'+message

        if '[SEG]' in question: granularity=2
        elif '[VG]' in question or '[REF]' in question: granularity=1
        else: granularity=0

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_id = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = (input_id,)
        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        input_ids = left_pad_sequences(
            input_ids, max_length, tokenizer.pad_token_id
        )
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids[:, : tokenizer.model_max_length]


        with torch.no_grad():
            image_tensors = [_image.to(dtype= torch.float16, device="cuda") for _image in image_tensors]
            # Generate response
            cont = model.generate(
                input_ids.to("cuda"),
                images = image_tensors,
                image_sizes = image_sizes,
                modalities = ["image"]*len(image_sizes),
                do_sample=False,
                temperature=0,
                max_new_tokens=32000,
                granularity=granularity
            )
            text_output = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        if '[CAP]' in question or '[VQA]' in question or '[CLS]' in question or '[CCD]' in question:
            response = text_output
            output_image = None
        elif '[VG]' in question:
            response = text_output
            output_image = draw_box(text_output, images_pil[0])
        elif 'REF' in question:
            response = text_output
            output_image = draw_box(question, images_pil[0])
        if '[SEG]' in question:
            response = "Mask is: "
            pred_color_mask = parse_output_string(text_output, image_sizes[0],20)
            # print(pred_color_mask)
            if len(images_pil)==2:
                output_image = Image.fromarray(cv2.cvtColor(pred_color_mask,cv2.COLOR_BGR2RGB))
            else:
                imgnp = cv2.cvtColor(np.array(images_pil[0]), cv2.COLOR_RGB2BGR)
                # 创建一个红色图层
                red_layer = np.zeros_like(imgnp,dtype=np.uint8)
                red_layer[:, :, 2] = 255  # 将红色通道设置为255

                # 使用mask叠加红色图层到原图上
                overlay = cv2.bitwise_and(red_layer, red_layer, mask=pred_color_mask[:,:,0])
                combine = cv2.addWeighted(imgnp, 1, overlay, 1, 0) 
                output_image = Image.fromarray(cv2.cvtColor(combine,cv2.COLOR_BGR2RGB))
        # print(images)
        # response += f"Uploaded {len(images)} images: {[image.name for image in images]}."
        # # 加载第一张图片作为返回的示例
        # output_image = Image.open(images[0].name) if len(images) > 0 else None

    else:
        response += "No images uploaded."
        output_image = None  # 如果没有图片上传，返回空

    # 根据response的长度逐渐输出文字
    for i in range(len(response)):
        time.sleep(0.01)  # 每输出一个字符后暂停0.05秒，模拟打字效果
        yield response[:i + 1], output_image  # 返回文字和图片


# 添加示例数据
examples1 = [
    ["[CAP] Describe this image briefly.", "/home/liux/LLaVA-NeXT/playground/demo_images/05863_0000.png", None],
    ["[VG] Where is the windwill?", "/home/liux/LLaVA-NeXT/playground/demo_images/05863_0000.png", None],
    ["[SEG] water.", "/home/liux/LLaVA-NeXT/playground/demo_images/05864_0000.png", None],
    ["[SEG] bridge.", "/home/liux/LLaVA-NeXT/playground/demo_images/05864_0000.png", None],
    ["[VQA] How many basketball courts?", "/home/liux/LLaVA-NeXT/playground/demo_images/05865_0000.png", None],
    ["[REF] what is [1, 20, 9, 30]?", "/home/liux/LLaVA-NeXT/playground/demo_images/05865_0000.png", None],
    ["[CCD] Please briefly describe the changes in these two images.", "/home/liux/LLaVA-NeXT/playground/demo_images/CD/A/test_000289.png", "/home/liux/LLaVA-NeXT/playground/demo_images/CD/B/test_000289.png"],
    ["[SEG] Please segment the building area that have changed in the second image.", "/home/liux/LLaVA-NeXT/playground/demo_images/CD/A/image.png", "/home/liux/LLaVA-NeXT/playground/demo_images/CD/B/image.png"],
    ["[SEG] Please segment the road area that have changed in the second image.", "/home/liux/LLaVA-NeXT/playground/demo_images/CD/A/test_000289.png", "/home/liux/LLaVA-NeXT/playground/demo_images/CD/B/test_000289.png"],

]
examples2 = [ 
]

# 使用gr.Blocks创建一个自定义布局
with gr.Blocks(css=""".gradio-container {background-color: #f9f9f9; font-family: Arial, sans-serif;} """) as demo:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1 style="color: #483D8B;">🌟 RSUniVLM Chatbot 🌟</h1>
        <p>Analyze and generate responses with visual and text inputs seamlessly for remote sensing. Upload images, type your questions, and explore the magic!</p>
    </div>
    """)
    
    with gr.Row():
        # with gr.Column():
            # image_input = gr.Files(file_types=["image"], label="Upload Images")
        with gr.Row():
            image_input1 = gr.Image(label="Upload Image 1", height=400, width=400)
            image_input2 = gr.Image(label="Upload Image 2", height=400, width=400)

            # image_gallery = gr.Gallery(label="Uploaded Images", columns=2, height="300px",)
            # draw_pad = gr.Sketchpad(label="Draw Boxes", height=256, width=256)
            # clear_button = gr.Button("Clear Drawing")
        image_output = gr.Image(label="Image Output",
                                height=400,  # 设置高度
                                width=400,   # 设置宽度
                            )
        
    with gr.Row():
        message = gr.Textbox(placeholder="Type your message here...", label="User Message")
        chat_output = gr.Textbox(label="Chat Output", lines=5)
    submit_button = gr.Button("Send Message")

    # 绑定绘图控件的逻辑
    # clear_button.click(
    #     fn=lambda: None,  # 清空绘制
    #     inputs=None,
    #     outputs=draw_pad,
    # )

    # # 添加上传图片后同步到绘图控件
    # image_gallery.select(
    #     fn=lambda img: img[0] if img else None,
    #     inputs=image_gallery,
    #     outputs=draw_pad,
    # )

    # 添加示例展示
    gr.Examples(
        examples=examples1,
        inputs=[message, image_input1,image_input2],
        outputs=[chat_output, image_output],
        fn=echo,
        # examples_per_page=2, 
        cache_examples=False,
    )
    # with gr.Row():
    #     gr.Examples(
    #         examples=examples1,
    #         inputs=[message, image_input1,image_input2],
    #         outputs=[chat_output, image_output],
    #         fn=echo,
    #         # examples_per_page=2, 
    #         cache_examples=False,
    #     )
    #     gr.Examples(
    #         examples=examples2,
    #         inputs=[message, image_input1,image_input2],
    #         outputs=[chat_output, image_output],
    #         fn=echo,
    #         # examples_per_page=2, 
    #         cache_examples=False,
    #     )

    # 将提交按钮的点击事件绑定到 `echo` 函数
    submit_button.click(
        fn=echo,
        inputs=[message, image_input1, image_input2],
        outputs=[chat_output, image_output],
    )

# 启动 Gradio 应用
demo.launch(server_name="0.0.0.0", server_port=1234)
