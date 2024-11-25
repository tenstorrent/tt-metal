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
        "params": [
            ((1, 5), 3),
            ((1, 32), 3),
            ((1, 50), 50),
            ((1, 50), 50257),
        ],
    }
}


def run(
    params,
    *,
    device,
) -> list:
    [input_shape, k] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.topk(torch_input_tensor, k)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    output_tensor = ttnn.topk(input_tensor, k)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.9999
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
