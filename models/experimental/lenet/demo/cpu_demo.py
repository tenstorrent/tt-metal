# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

from models.experimental.lenet.lenet_utils import load_torch_lenet, prepare_image


def test_cpu_demo(mnist_sample_input, model_location_generator):
    sample_image = mnist_sample_input
    image = prepare_image(sample_image)
    num_classes = 10
    batch_size = 1
    with torch.no_grad():
        pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
        torch_LeNet, state_dict = load_torch_lenet(pt_model_path, num_classes)

        torch_output = torch_LeNet(image).unsqueeze(1).unsqueeze(1)
        _, torch_predicted = torch.max(torch_output.data, -1)

        sample_image.save("input_image.jpg")
        logger.info(f"Input image savd as input_image.jpg.")
        logger.info(f"CPU's predicted Output: {torch_predicted[0][0][0]}.")
