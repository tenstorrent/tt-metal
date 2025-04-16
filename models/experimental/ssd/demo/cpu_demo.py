# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large


def test_cpu_demo():
    image = Image.open("models/sample_data/huggingface_cat_image.jpg")
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        torch_ssd = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights)
        torch_ssd.eval()
        torch_output = torch_ssd(image)

        logger.info(f"CPU's predicted Output: {torch_output[0]['scores']}")
        logger.info(f"CPU's predicted Output: {torch_output[0]['labels']}")
        logger.info(f"CPU's predicted Output: {torch_output[0]['boxes']}")
