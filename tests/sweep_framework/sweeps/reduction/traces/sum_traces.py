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
    }
}


def run(
    params,
    *,
    device,
) -> list:
    [input_shape, dim, keepdim] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.sum(torch_input_tensor, dim, keepdim)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.9999
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
