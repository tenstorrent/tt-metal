from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor
from tt.modeling_vit import vit_for_image_classification


def test_gs_demo():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()


    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    HF_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224") # loaded for the labels
    inputs = image_processor(image, return_tensors="pt")

    tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
    tt_model = vit_for_image_classification(device)

    with torch.no_grad():
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output, host).squeeze(0)[:, 0, :]

    # model predicts one of the 1000 ImageNet classes
    image.save("vit_input_image.jpg")
    predicted_label = tt_output.argmax(-1).item()
    logger.info(f"Input image savd as input_image.jpg.")
    logger.info(f"CPU's predicted Output: {HF_model.config.id2label[predicted_label]}.")
