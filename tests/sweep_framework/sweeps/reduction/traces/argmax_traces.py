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
        "height": [1, 2],
        "width": [7, 51865],
        "dim": [-1],
        "dtype": [ttnn.float32, ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    }
}


def run_argmax(device, height, width, dim, dtype, layout):
    torch_input_tensor = torch.rand([height, width], dtype=torch.float32)
    torch_output_tensor = torch.argmax(torch_input_tensor, dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.argmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("height", parameters["pytorch"]["height"])
@pytest.mark.parametrize("width", parameters["pytorch"]["width"])
@pytest.mark.parametrize("dim", parameters["pytorch"]["dim"])
@pytest.mark.parametrize("dtype", parameters["pytorch"]["dtype"])
@pytest.mark.parametrize("layout", parameters["pytorch"]["layout"])
def test_pytorch(device, height, width, dim, dtype, layout):
    run_argmax(device, height, width, dim, dtype, layout)


def run(
    height,
    width,
    dim,
    dtype,
    layout,
    *,
    device,
) -> list:
    return run_argmax(device, height, width, dim, dtype, layout)
