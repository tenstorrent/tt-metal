# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "m_size": [384, 1024],  # [1, 16, 128, 1024]
    "k_size": [1024, 4096],  # [16, 128, 1024, 4096]
    "n_size": [1024, 4096],  # [16, 128, 1024, 4096]
    "use_bias": [False, True],
    "input_dtype_a": [ttnn.bfloat16],
    "input_dtype_b": [ttnn.bfloat16],
    "output_dtype": [ttnn.bfloat16],
    "input_memory_config_a": [ttnn.DRAM_MEMORY_CONFIG],
    "input_memory_config_b": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "core_grid": [None],
}


def skip(**_):
    return False


def run(
    batch_sizes,
    m_size,
    k_size,
    n_size,
    use_bias,
    input_dtype_a,
    input_dtype_b,
    output_dtype,
    input_memory_config_a,
    input_memory_config_b,
    output_memory_config,
    core_grid,
    *,
    device,
):
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.bfloat16)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.bfloat16)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(torch_input_tensor_a, torch_input_tensor_b, bias=torch_bias)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        device=device,
        dtype=input_dtype_a,
        memory_config=input_memory_config_a,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        device=device,
        dtype=input_dtype_b,
        memory_config=input_memory_config_b,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=input_dtype_b,
            memory_config=input_memory_config_b,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        dtype=output_dtype,
        memory_config=output_memory_config,
        core_grid=core_grid,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
