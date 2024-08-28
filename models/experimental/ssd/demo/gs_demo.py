# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn
from models.utility_functions import torch_to_tt_tensor_rm

from models.experimental.ssd.tt.ssd_lite import *
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image


def test_gs_demo():
    device = ttnn.open_device(0)

    ttnn.SetDefaultDevice(device)

    image = Image.open("models/sample_data/huggingface_cat_image.jpg")
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        tt_model = ssd_for_object_detection(device)
        tt_model.eval()
        tt_input = torch_to_tt_tensor_rm(image, device)
        tt_output = tt_model(tt_input)

        logger.info(f"GS's predicted scores: {tt_output[0]['scores']}")
        logger.info(f"GS's predicted labels: {tt_output[0]['labels']}")
        logger.info(f"GS's predicted boxes: {tt_output[0]['boxes']}")

        save_image(image, "ssd_input.jpg")
        logger.info("Input image is saved for reference as ssd_input.jpg")

    ttnn.close_device(device)
