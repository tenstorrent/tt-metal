# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from torchvision.utils import save_image


from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from transformers import SwinForImageClassification as HF_SwinForImageClassification
from transformers import AutoFeatureExtractor
from models.experimental.swin.tt.swin import *


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/swin-tiny-patch4-window7-224"),),
)
def test_gs_demo(device, imagenet_sample_input, model_name):
    image = imagenet_sample_input

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HF_SwinForImageClassification.from_pretrained(model_name)

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        tt_model = swin_for_image_classification(device)

        # Run tt model
        tt_pixel_values = inputs.pixel_values
        tt_pixel_values = torch_to_tt_tensor_rm(tt_pixel_values, device)
        tt_output = tt_model(tt_pixel_values)

        tt_output_torch = tt_to_torch_tensor(tt_output.logits).squeeze(0).squeeze(0)

        predicted_label = tt_output_torch.argmax(-1).item()
        logger.info("GS's Predicted Output")
        logger.info(model.config.id2label[predicted_label])

        save_image(imagenet_sample_input, "swin_input.jpg")
        logger.info("Input image is saved for reference as swin_input.jpg")
