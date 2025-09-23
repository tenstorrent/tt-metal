# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import ttnn

from models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    torch2tt_tensor,
)
from models.experimental.convnet_mnist.tt.convnet_mnist import convnet_mnist
from models.experimental.convnet_mnist.convnet_mnist_utils import get_test_data


def test_mnist_inference(device):
    tt_convnet, pt_convnet = convnet_mnist(device)
    test_input, images = get_test_data(64)

    with torch.no_grad():
        pt_output = pt_convnet(test_input).unsqueeze(1).unsqueeze(1)

        tt_input = torch2tt_tensor(test_input, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_output = tt_convnet(tt_input)
        tt_output = tt2torch_tensor(tt_output)

        pcc_passing, pcc_output = comp_pcc(pt_output, tt_output)
        logger.info(f"Output {pcc_output}")

        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
