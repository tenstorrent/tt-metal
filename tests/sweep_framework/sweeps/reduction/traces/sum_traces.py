# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import ttnn

from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15

parameters = {
    "pytorch": {
        "params": [
            ((1, 1, 768), (0, 1), True),
            ((1, 1000), (0), True),
            ((1, 1024, 256), (0, 1), True),
            ((1, 1024, 7, 7), (2, 3), True),
            ((1, 10), (0), True),
            ((1, 12, 16), (1), False),
            ((1, 12, 16), (2), False),
            ((1, 120, 28, 28), (2, 3), True),
            ((1, 128), (0), True),
            ((1, 12), (0), True),
            ((1, 16384, 256), (0, 1), True),
            ((1, 197, 1024), (0, 1), True),
            ((1, 197, 768), (0, 1), True),
            ((1, 21843), (0), True),
            ((1, 256, 256), (0, 1), True),
            ((1, 256, 56, 56), (2, 3), True),
            ((1, 3), (0), True),
            ((1, 4096, 256), (0, 1), True),
            ((1, 480, 14, 14), (2, 3), True),
            ((1, 512, 28, 28), (2, 3), True),
            ((1, 512), (1), True),
            ((1, 64), (0), True),
            ((1, 672, 14, 14), (2, 3), True),
            ((1, 672, 7, 7), (2, 3), True),
            ((1, 72, 28, 28), (2, 3), True),
            ((1, 768, 14, 14), (2, 3), True),
            ((1, 768, 384), (0, 1), True),
            ((1, 784), (0), True),
            ((1, 960, 7, 7), (2, 3), True),
            ((1024, 160), (0), True),
            ((1024, 640), (0), True),
            ((14, 2048), (0), True),
            ((14, 512), (0), True),
            ((16384, 128), (0), True),
            ((16384, 32), (0), True),
            ((196, 3072), (0), True),
            ((196, 768), (0), True),
            ((197, 1024), (0), True),
            ((197, 3072), (0), True),
            ((197, 4096), (0), True),
            ((197, 768), (0), True),
            ((2, 512), (1), True),
            ((2, 7, 512), (0), True),
            ((256, 1024), (0), True),
            ((256, 160), (0), True),
            ((256, 256), (0), True),
            ((256, 32), (0), True),
            ((256, 512), (0), True),
            ((256, 64), (0), True),
            ((4096, 256), (0), True),
            ((4096, 64), (0), True),
            ((50, 3072), (0), True),
            ((50, 768), (0), True),
            ((768, 196), (0), True),
            ((2, 1), None, False),
            ((1), None, False),
        ],
    },
    "forge": {
        "params": [
            (
                (
                    1,
                    1,
                    1024,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    1,
                    16384,
                    256,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    1,
                    19200,
                    300,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    1,
                    512,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    1,
                    768,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    10,
                ),
                (1),
                False,
            ),
            (
                (
                    1,
                    10,
                    1024,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    10,
                    512,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    10,
                    768,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    12,
                    1,
                    1,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    12,
                    1,
                    10,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    12,
                    10,
                    10,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    12,
                    16,
                ),
                (1),
                False,
            ),
            (
                (
                    1,
                    12,
                    16,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    12,
                    197,
                    197,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    12,
                    201,
                    201,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    12,
                    8,
                    8,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    120,
                    40,
                    40,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    1280,
                    7,
                    7,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    15,
                    512,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    16,
                    1,
                    1,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    16,
                    1,
                    10,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    16,
                    10,
                    10,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    16,
                    197,
                    197,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    16,
                    32,
                    32,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    16,
                    5,
                    5,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    196,
                    1024,
                ),
                (1),
                False,
            ),
            (
                (
                    1,
                    196,
                    768,
                ),
                (1),
                False,
            ),
            (
                (
                    1,
                    2,
                    4096,
                    256,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    2,
                    4800,
                    300,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    2048,
                    7,
                    7,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    32,
                    4096,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    480,
                    10,
                    10,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    480,
                    20,
                    20,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    5,
                    1024,
                    256,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    5,
                    1200,
                    300,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    512,
                ),
                (1),
                False,
            ),
            (
                (
                    1,
                    512,
                    256,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    512,
                    7,
                    7,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    6,
                    1,
                    1,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    6,
                    1,
                    15,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    6,
                    15,
                    15,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    672,
                    10,
                    10,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    672,
                    20,
                    20,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    72,
                    40,
                    40,
                ),
                (2),
                False,
            ),
            (
                (
                    1,
                    8,
                    1,
                    1,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    8,
                    1,
                    10,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    8,
                    10,
                    10,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    8,
                    2048,
                    256,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    8,
                    256,
                    2048,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    8,
                    256,
                    256,
                ),
                (3),
                False,
            ),
            (
                (
                    1,
                    8,
                    300,
                    300,
                ),
                (3),
                False,
            ),
            (
                (
                    19,
                    256008,
                ),
                (1),
                False,
            ),
            (
                (
                    2,
                    512,
                ),
                (1),
                False,
            ),
            (
                (
                    8,
                    100,
                    100,
                ),
                (2),
                False,
            ),
            (
                (
                    8,
                    100,
                    920,
                ),
                (2),
                False,
            ),
            (
                (
                    8,
                    920,
                    920,
                ),
                (2),
                False,
            ),
        ],
    },
}


def run_sum(device, params):
    [input_shape, dim, keepdim] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.sum(torch_input_tensor, dim, keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["pytorch"]["params"])
def test_pytorch(device, params):
    run_sum(device, params)


@pytest.mark.parametrize("params", parameters["forge"]["params"])
def test_forge(device, params):
    run_sum(device, params)


def run(
    params,
    *,
    device,
) -> list:
    return run_sum(device, params)
