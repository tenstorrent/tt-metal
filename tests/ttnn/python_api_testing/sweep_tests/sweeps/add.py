# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

parameters = {
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "broadcast": [None, "h", "w", "hw"],
    "input_dtype_a": [ttnn.bfloat16],
    "input_dtype_b": [ttnn.bfloat16],
    "input_memory_config_a": [ttnn.DRAM_MEMORY_CONFIG],
    "input_memory_config_b": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}


def skip(**_):
    return False


def run(
    batch_sizes,
    height,
    width,
    broadcast,
    input_dtype_a,
    input_dtype_b,
    input_memory_config_a,
    input_memory_config_b,
    output_memory_config,
    *,
    device,
):
    input_shape_a = (*batch_sizes, height, width)
    input_shape_b = (*batch_sizes, height, width)
    if broadcast == "hw":
        input_shape_b = (*batch_sizes, 1, 1)
    elif broadcast == "h":
        input_shape_b = (*batch_sizes, 1, width)
    elif broadcast == "w":
        input_shape_b = (*batch_sizes, height, 1)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, device=device, dtype=input_dtype_a, memory_config=input_memory_config_a
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, device=device, dtype=input_dtype_b, memory_config=input_memory_config_b
    )

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
