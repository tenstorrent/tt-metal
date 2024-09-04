# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 5

# TODO: Consolidate tests for duplicate use cases into sweep with shapes
# TODO: Missing coverage for Stable Diffusion matmul in: tests/ttnn/unit_tests/operations/test_matmul.py
parameters = {
    "default": {
        "matmul_specs": [
            # Create program config from core_grid
            (
                (1,),
                (1024, 1024, 512),
                False,
                (ttnn.CoreGrid(y=4, x=4), None),
            ),
            (
                (7,),
                (384, 1024, 1024),
                False,
                (ttnn.CoreGrid(y=7, x=8), None),
            ),
            # Create program config from use_1d_systolic_array flag: mcast in1 (ie. tall)
            (
                (1,),
                (1024, 1023, 32),
                False,
                (None, True),
            ),
            (
                (8,),
                (2048, 2048, 61),
                False,
                (None, True),
            ),
            # Create program config from use_1d_systolic_array flag: mcast in0 (ie. wide)
            (
                (1,),
                (31, 1024, 1023),
                False,
                (None, True),
            ),
            (
                (8,),
                (63, 2048, 2047),
                False,
                (None, True),
            ),
            # Create program config from core_grid and use_1d_systolic_array flag
            (
                (1,),
                (128, 4544, 4672),
                False,
                (ttnn.CoreGrid(y=7, x=8), True),
            ),
        ],
        "compute_kernel_config": [None],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
    }
}


def run(
    matmul_specs,
    compute_kernel_config,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    *,
    device,
) -> list:
    batch_sizes, input_shapes, batch_matrix_multiply, create_program_config_specs = matmul_specs

    (core_grid, use_1d_systolic_array) = create_program_config_specs

    (m_size, k_size, n_size) = input_shapes
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    input_a_layout = input_layout
    input_b_layout = input_layout

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=input_a_layout,
        dtype=input_a_dtype,
        device=device,
        memory_config=input_a_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=input_b_layout,
        dtype=input_b_dtype,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_memory_config,
        dtype=output_dtype,
        core_grid=core_grid or device.core_grid,
        compute_kernel_config=compute_kernel_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # TODO: For larger inner dims (ie. 2048), output in bfloat8_b will have low PCC
    expected_pcc = 0.99
    if k_size >= 2048 and output_dtype == ttnn.bfloat8_b:
        expected_pcc = 0.97

    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
