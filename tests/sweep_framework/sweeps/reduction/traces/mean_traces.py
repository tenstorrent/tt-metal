# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

TIMEOUT = 15

parameters = {
    "pytorch": {
        "params": [
            ((1, 1, 1024), (-1), True),
            ((1, 1, 512), (-1), True),
            ((1, 1, 768), (-1), True),
            ((1, 10, 1024), (-1), True),
            ((1, 10, 512), (-1), True),
            ((1, 10, 768), (-1), True),
            ((1, 1008, 7, 7), (-1, -2), True),
            ((1, 1024, 7, 7), (-1, -2), True),
            ((1, 1024, 7, 7), (2, 3), True),
            ((1, 1024, 8, 8), (-1, -2), True),
            ((1, 104, 28, 28), (-1, -2), True),
            ((1, 1056, 48, 48), (-1, -2), True),
            ((1, 120, 14, 14), (-1, -2), True),
            ((1, 120, 28, 28), (-1, -2), True),
            ((1, 120, 28, 28), (2, 3), True),
            ((1, 120, 40, 40), (-1, -2), True),
            ((1, 1232, 14, 14), (-1, -2), True),
            ((1, 1280, 10, 10), (-1, -2), True),
            ((1, 1280, 12, 12), (-1, -2), True),
            ((1, 1280, 7, 7), (-1, -2), True),
            ((1, 1280, 8, 8), (-1, -2), True),
            ((1, 1280, 9, 9), (-1, -2), True),
            ((1, 1392, 14, 14), (-1, -2), True),
            ((1, 144, 14, 14), (-1, -2), True),
            ((1, 144, 28, 28), (-1, -2), True),
            ((1, 15, 512), (-1), True),
            ((1, 1512, 7, 7), (-1, -2), True),
            ((1, 1536, 8, 8), (-1, -2), True),
            ((1, 16, 56, 56), (-1, -2), True),
            ((1, 1664, 7, 7), (-1, -2), True),
            ((1, 1920, 7, 7), (-1, -2), True),
            ((1, 196, 1024), (1), False),
            ((1, 196, 768), (1), False),
            ((1, 2016, 7, 7), (-1, -2), True),
            ((1, 2048, 10, 10), (-1, -2), True),
            ((1, 2048, 7, 7), (-1, -2), True),
            ((1, 208, 14, 14), (-1, -2), True),
            ((1, 216, 28, 28), (-1, -2), True),
            ((1, 2208, 7, 7), (-1, -2), True),
            ((1, 224, 56, 56), (-1, -2), True),
            ((1, 232, 56, 56), (-1, -2), True),
            ((1, 240, 14, 14), (-1, -2), True),
            ((1, 2520, 7, 7), (-1, -2), True),
            ((1, 256, 56, 56), (2, 3), True),
            ((1, 288, 7, 7), (-1, -2), True),
            ((1, 2904, 24, 24), (-1, -2), True),
            ((1, 3024, 7, 7), (-1, -2), True),
            ((1, 320, 14, 14), (-1, -2), True),
            ((1, 336, 14, 14), (-1, -2), True),
            ((1, 3712, 7, 7), (-1, -2), True),
            ((1, 400, 7, 7), (-1, -2), True),
            ((1, 440, 7, 7), (-1, -2), True),
            ((1, 448, 28, 28), (-1, -2), True),
            ((1, 48, 56, 56), (-1, -2), True),
            ((1, 480, 10, 10), (-1, -2), True),
            ((1, 480, 14, 14), (-1, -2), True),
            ((1, 480, 14, 14), (2, 3), True),
            ((1, 480, 20, 20), (-1, -2), True),
            ((1, 512, 256), (2), False),
            ((1, 512, 28, 28), (2, 3), True),
            ((1, 512, 7, 7), (-1, -2), True),
            ((1, 528, 96, 96), (-1, -2), True),
            ((1, 576, 14, 14), (-1, -2), True),
            ((1, 576, 7, 7), (-1, -2), True),
            ((1, 64, 56, 56), (-1, -2), True),
            ((1, 672, 10, 10), (-1, -2), True),
            ((1, 672, 14, 14), (-1, -2), True),
            ((1, 672, 14, 14), (2, 3), True),
            ((1, 672, 20, 20), (-1, -2), True),
            ((1, 672, 7, 7), (-1, -2), True),
            ((1, 672, 7, 7), (2, 3), True),
            ((1, 696, 28, 28), (-1, -2), True),
            ((1, 72, 28, 28), (-1, -2), True),
            ((1, 72, 28, 28), (2, 3), True),
            ((1, 72, 40, 40), (-1, -2), True),
            ((1, 72, 56, 56), (-1, -2), True),
            ((1, 7392, 12, 12), (-1, -2), True),
            ((1, 768, 14, 14), (2, 3), True),
            ((1, 768, 7, 7), (-1, -2), True),
            ((1, 768, 8, 8), (-1, -2), True),
            ((1, 784, 7, 7), (-1, -2), True),
            ((1, 888, 7, 7), (-1, -2), True),
            ((1, 896, 14, 14), (-1, -2), True),
            ((1, 912, 7, 7), (-1, -2), True),
            ((1, 96, 14, 14), (-1, -2), True),
            ((1, 960, 7, 7), (-1, -2), True),
            ((1, 960, 7, 7), (2, 3), True),
        ],
    }
}


def run_mean(device, params):
    [input_shape, dim, keepdim] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.mean(torch_input_tensor, dim, keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["pytorch"]["params"])
def test_pytorch(device, params):
    run_mean(device, params)


def run(
    params,
    *,
    device,
) -> list:
    return run_mean(device, params)
