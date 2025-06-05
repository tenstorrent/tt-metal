# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch

import ttnn

from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

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


def run_matmul(
    device,
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
    op_output_tensor = ttnn.matmul(
        input_tensor_a, input_tensor_b, dtype=output_dtype, memory_config=output_memory_config, core_grid=core_grid
    )
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.99
    tensors = [input_tensor_a, input_tensor_b, op_output_tensor]
    flop_counts = [m_size, n_size, 2, k_size] + list(batch_sizes)
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf, flop_counts)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_matmul(
    device,
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
):
    run_matmul(
        device,
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
    )


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
    return run_matmul(
        device,
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
    )
