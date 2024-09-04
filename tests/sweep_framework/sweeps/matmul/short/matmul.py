# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 5

parameters = {
    "default": {
        "batch_sizes": [(1,)],
        "m_size": [384, 1024],  # [1, 16, 128, 1024]
        "k_size": [1024, 4096],  # [16, 128, 1024, 4096]
        "n_size": [1024, 4096],  # [16, 128, 1024, 4096]
        "batch_matrix_multiply": [True, False],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "output_dtype": [ttnn.bfloat16],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "core_grid": [None],
    }
}


def run(
    batch_sizes,
    m_size,
    k_size,
    n_size,
    batch_matrix_multiply,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    output_memory_config,
    core_grid,
    *,
    device,
) -> list:
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=input_a_layout,
        dtype=input_a_dtype,
        device=device,
        memory_config=input_b_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=input_b_layout,
        dtype=input_b_dtype,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(
        input_tensor_a, input_tensor_b, dtype=output_dtype, memory_config=output_memory_config, core_grid=core_grid
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.99

    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
