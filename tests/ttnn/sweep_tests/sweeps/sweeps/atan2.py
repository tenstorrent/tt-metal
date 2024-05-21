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
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat16],
    "input_a_layout": [ttnn.TILE_LAYOUT],
    "input_b_layout": [ttnn.TILE_LAYOUT],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "input_YX": [[20, 40, 40, 80], [-1, 1, -1, 1]],
}


def run(
    batch_sizes,
    height,
    width,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    output_memory_config,
    input_YX,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)
    a1, a2, b1, b2 = input_YX

    torch_input_tensor_a = torch.linspace(a1, a2, steps=height * width, dtype=torch.bfloat16).reshape(input_shape)
    torch_input_tensor_b = torch.linspace(b1, b2, steps=height * width, dtype=torch.bfloat16).reshape(input_shape)

    torch_output_tensor = torch.atan2(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        device=device,
        layout=input_a_layout,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        device=device,
        layout=input_b_layout,
        memory_config=input_b_memory_config,
    )
    output_tensor = ttnn.atan2(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
