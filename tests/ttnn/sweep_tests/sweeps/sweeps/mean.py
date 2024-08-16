# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT],
    "dim": [-1, -2],
}


def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    dim,
    layout,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    output_tensor = ttnn.mean(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
