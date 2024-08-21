# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
from torchvision.utils import save_image


@pytest.mark.parametrize(
    "model_name",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k"),),
)
def test_timm_vovnet_image_classification_inference(
    model_name, imagenet_sample_input, imagenet_label_dict, reset_seeds
):
    class_labels = imagenet_label_dict

    Timm_model = timm.create_model(model_name, pretrained=True)
    Timm_model.eval()

    with torch.no_grad():
        Timm_output = Timm_model(imagenet_sample_input)

    logger.info("Timm Model answered")
    logger.info(class_labels[Timm_output[0].argmax(-1).item()])

    save_image(imagenet_sample_input[0], "image.jpeg")
    logger.info("Input is saved for reference as image.jpeg")
