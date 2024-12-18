# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15

parameters = {
    "default": {
        "height": [1, 2],
        "width": [7, 51865],
        "dim": [-1],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    }
}


def run(
    height,
    width,
    dim,
    dtype,
    layout,
    *,
    device,
) -> list:
    torch_input_tensor = torch.rand([height, width], dtype=torch.float32)
    torch_output_tensor = torch.argmax(torch_input_tensor, dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout, device=device)

    start_time = start_measuring_time()
    output_tensor = ttnn.argmax(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.9999
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
