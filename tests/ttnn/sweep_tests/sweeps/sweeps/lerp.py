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
    "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_a_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "input_b_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "input_w_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "weight_is_tensor": [True, False],
    "weight": [0.3, 0.5],
    "end": [100],
    "low": [-10, 10],
    "high": [40, 80],
}


def skip(
    batch_sizes,
    height,
    width,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    input_w_memory_config,
    output_memory_config,
    weight_is_tensor,
    weight,
    end,
    low,
    high,
) -> Tuple[bool, Optional[str]]:
    if input_a_layout == ttnn.ROW_MAJOR_LAYOUT or input_b_layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Skipped as ROW_MAJOR_LAYOUT not supported"
    return False, None


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
    input_w_memory_config,
    output_memory_config,
    weight_is_tensor,
    weight,
    end,
    low,
    high,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_input_tensor_a = torch.linspace(low, high, steps=height * width, dtype=torch.bfloat16).reshape(input_shape)
    torch_input_tensor_b = torch.full(input_shape, end, dtype=torch.bfloat16)

    torch_weight = weight
    if weight_is_tensor:
        torch_weight = torch.full(input_shape, weight, dtype=torch.bfloat16)

    torch_output_tensor = torch.lerp(torch_input_tensor_a, torch_input_tensor_b, torch_weight)

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
    input_weight = weight
    if weight_is_tensor:
        input_weight = ttnn.from_torch(
            torch_weight,
            dtype=input_a_dtype,
            device=device,
            layout=input_a_layout,
            memory_config=input_w_memory_config,
        )
    output_tensor = ttnn.lerp(input_tensor_a, input_tensor_b, input_weight, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.99)
