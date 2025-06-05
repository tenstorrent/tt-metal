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
            ((25, 4), None, False),
        ],
        "dtype": [ttnn.float32, ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
    "forge": {
        "params": [
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
                    10,
                ),
                (1),
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
        "dtype": [ttnn.float32, ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    },
}


def run_max(device, params, dtype, layout):
    [input_shape, dim, keepdim] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    print(f"inputs: a{torch_input_tensor} b{dim} c{keepdim}")
    if dim is None:
        assert not keepdim
        torch_output_tensor = torch.max(torch_input_tensor).values
    else:
        torch_output_tensor = torch.max(torch_input_tensor, dim=dim, keepdim=keepdim).values

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.max(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["pytorch"]["params"])
@pytest.mark.parametrize("dtype", parameters["pytorch"]["dtype"])
@pytest.mark.parametrize("layout", parameters["pytorch"]["layout"])
def test_pytorch(device, params, dtype, layout):
    run_max(device, params, dtype, layout)


@pytest.mark.parametrize("params", parameters["forge"]["params"])
@pytest.mark.parametrize("dtype", parameters["forge"]["dtype"])
@pytest.mark.parametrize("layout", parameters["forge"]["layout"])
def test_forge(device, params, dtype, layout):
    run_max(device, params, dtype, layout)


def run(
    params,
    dtype,
    layout,
    *,
    device,
) -> list:
    return run_max(device, params, dtype, layout)
