from pathlib import Path
import sys
f = f"{Path(__file__).parent}"

sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from PIL import Image
import requests
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor
from deit_for_image_classification_with_teacher import deit_for_image_classification_with_teacher

def test_gs_demo():
    torch.manual_seed(3)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    inputs = image_processor(images=image, return_tensors="pt")

    torch_model_with_teacher = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    torch_model_with_teacher.eval()
    torch_out_with_teacher = torch_model_with_teacher(**inputs).logits

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
    tt_model_with_teacher = deit_for_image_classification_with_teacher(device)

    with torch.no_grad():
        tt_output_with_teacher = tt_model_with_teacher(tt_inputs)[0]
        tt_output_with_teacher = tt_to_torch_tensor(tt_output_with_teacher, host).squeeze(0)[:, 0, :]

    # model prediction
    image.save("deit_input_image.jpg")
    predicted_label = tt_output_with_teacher.argmax(-1).item()
    predicted_label_torch = torch_out_with_teacher.argmax(-1).item()

    print("\nTT's prediction class:",torch_model_with_teacher.config.id2label[predicted_label] , '\n')
    logger.info(f"Input image saved as input_image.jpg.")
    logger.info(f"TT's prediction: {torch_model_with_teacher.config.id2label[predicted_label]}.")
