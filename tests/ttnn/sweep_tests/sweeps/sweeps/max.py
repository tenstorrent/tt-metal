# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

parameters = {
    "batch_sizes": [(1,)],
    "height": [32, 64],
    "width": [64, 64],
    "input_dtype": [ttnn.bfloat16],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT],
    "dim": [
        -1,
        -2,
        None,
    ],
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

    low = -100
    high = 100

    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.bfloat16)

    if dim == None:
        torch_output_tensor = torch.max(torch_input_tensor)
    else:
        torch_output_tensor, _ = torch.max(torch_input_tensor, dim=dim, keepdim=True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )

    if dim == None:
        output_tensor = ttnn.max(input_tensor)
    else:
        output_tensor = ttnn.max(input_tensor, dim=dim, memory_config=output_memory_config)

    output_tensor = ttnn.to_torch(output_tensor)
    if dim == None:
        output_tensor = output_tensor[0, 0, 0, 0]

    return check_with_pcc(torch_output_tensor, output_tensor)
