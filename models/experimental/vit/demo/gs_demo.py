# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from loguru import logger
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

import ttnn

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.vit.tt.modeling_vit import vit_for_image_classification


def test_gs_demo():
    image = Image.open("models/sample_data/huggingface_cat_image.jpg")

    # Initialize the device
    device = ttnn.open_device(0)

    ttnn.SetDefaultDevice(device)

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    HF_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")  # loaded for the labels
    inputs = image_processor(image, return_tensors="pt")

    tt_inputs = torch_to_tt_tensor_rm(inputs["pixel_values"], device, put_on_device=False)
    tt_model = vit_for_image_classification(device)

    with torch.no_grad():
        tt_output = tt_model(tt_inputs)[0]
        tt_output = tt_to_torch_tensor(tt_output).squeeze(0)[:, 0, :]

    # model predicts one of the 1000 ImageNet classes
    image.save("vit_input_image.jpg")
    predicted_label = tt_output.argmax(-1).item()
    logger.info(f"Input image savd as input_image.jpg.")
    logger.info(f"CPU's predicted Output: {HF_model.config.id2label[predicted_label]}.")
