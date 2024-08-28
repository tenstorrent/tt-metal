# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from loguru import logger
import ttnn

from models.utility_functions import (
    tt2torch_tensor,
    torch2tt_tensor,
)
from models.experimental.convnet_mnist.tt.convnet_mnist import convnet_mnist
from models.experimental.convnet_mnist.convnet_mnist_utils import get_test_data


def test_mnist_inference():
    device = ttnn.open_device(0)

    ttnn.SetDefaultDevice(device)

    tt_convnet, pt_convnet = convnet_mnist(device)

    test_input, images = get_test_data(1)
    input_path = ROOT / "input_image.jpg"
    images[0].save(input_path)
    logger.info(f"Input image saved to {input_path}")

    with torch.no_grad():
        tt_input = torch2tt_tensor(test_input, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_output = tt_convnet(tt_input)
        tt_output = tt2torch_tensor(tt_output)

        _, tt_predicted = torch.max(tt_output.data, -1)
        logger.info(f"tt_predicted {tt_predicted.squeeze()}")

    ttnn.close_device(device)
