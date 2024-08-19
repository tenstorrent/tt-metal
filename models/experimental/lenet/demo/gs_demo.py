# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn

from models.experimental.lenet.tt.lenet import lenet5
from models.experimental.lenet.lenet_utils import load_torch_lenet, prepare_image


def test_gs_demo(device, mnist_sample_input, model_location_generator):
    sample_image = mnist_sample_input
    image = prepare_image(sample_image)
    num_classes = 10
    batch_size = 1
    with torch.no_grad():
        tt_lenet = lenet5(num_classes, device, model_location_generator)

        tt_image = ttnn.Tensor(
            image.reshape(-1).tolist(),
            image.shape,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        tt_output = tt_lenet(tt_image)
        tt_output = tt_output.cpu()
        tt_output = tt_output.to_torch()

        _, tt_predicted = torch.max(tt_output.data, -1)

        sample_image.save("input_image.jpg")
        logger.info(f"Input image saved as input_image.jpg.")
        logger.info(f"GS's predicted Output: {tt_predicted[0][0][0]}.")
