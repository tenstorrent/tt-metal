from pathlib import Path
import sys
import torch
import pytest
from loguru import logger
from torchvision.utils import save_image

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.utility_functions_new import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from transformers import AutoFeatureExtractor
import tt_lib
from python_api_testing.models.swin.tt.swin import *


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/swin-tiny-patch4-window7-224"),),
)
def test_gs_demo(imagenet_sample_input, model_name):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    image = imagenet_sample_input

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HF_SwinForImageClassification.from_pretrained(model_name)

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        tt_model = swin_for_image_classification(device, host)

        # Run tt model
        tt_pixel_values = inputs.pixel_values
        tt_pixel_values = torch_to_tt_tensor_rm(tt_pixel_values, device)
        tt_output = tt_model(tt_pixel_values)

        tt_output_torch = (
            tt_to_torch_tensor(tt_output.logits).squeeze(0).squeeze(0)
        )

        predicted_label = tt_output_torch.argmax(-1).item()
        logger.info("GS's Predicted Output")
        logger.info(model.config.id2label[predicted_label])

        save_image(imagenet_sample_input, "swin_input.jpg")
        logger.info("Input image is saved for reference as swin_input.jpg")

    tt_lib.device.CloseDevice(device)
