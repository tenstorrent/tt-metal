# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
from loguru import logger


_batch_size = 1


def test_cpu_demo(imagenet_sample_input, imagenet_label_dict):
    image = imagenet_sample_input
    class_labels = imagenet_label_dict

    batch_size = _batch_size
    with torch.no_grad():
        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        torch_vgg.eval()
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)

        logger.info(f"CPU's predicted Output: {class_labels[torch.argmax(torch_output).item()]}")
