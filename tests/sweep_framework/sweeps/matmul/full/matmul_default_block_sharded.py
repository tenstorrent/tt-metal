# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
from loguru import logger
import itertools
import functools
import operator

import torch

import ttnn

from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    get_per_core_size_and_num_cores,
    start_measuring_time,
    stop_measuring_time,
)
from models.utility_functions import torch_random

TIMEOUT = 5


def get_block_sharded_specs(
    batch_sizes_choices: List[Tuple[int, ...]],
    m_size_choices: List[int],
    k_size_choices: List[int],
    num_cores_choices: List[int],
) -> Tuple[Tuple[int, ...], int, int, int, int, int]:
    for batch_sizes, m_size, k_size in itertools.product(batch_sizes_choices, m_size_choices, k_size_choices):
        total_height = functools.reduce(operator.mul, batch_sizes) * m_size
        for per_core_height, num_cores_height in get_per_core_size_and_num_cores(
            total_height, num_cores_choices, max_per_core_size=1024
        ):
            for per_core_width, num_cores_width in get_per_core_size_and_num_cores(
                k_size, num_cores_choices, max_per_core_size=1024
            ):
                yield (batch_sizes, m_size, k_size, per_core_height, per_core_width, num_cores_height, num_cores_width)


parameters = {
    f"n_size_{n if n > 0 else 32}": {
        "block_sharded_specs": list(
            get_block_sharded_specs(
                [(1,), (2,)],
                [x if x > 0 else 32 for x in range(0, 2048, 384)],
                [x if x > 0 else 32 for x in range(0, 2048, 256)],
                [1, 4, 7],
            )
        ),
        "n_size": [n if n > 0 else 32],
        "transpose_mcast": [False, True],
        "batch_matrix_multiply": [False],
        "input_a_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        "input_a_dtype": [ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16],
        "output_dtype": [ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "compute_kernel_config": [None],
    }
    for n in range(0, 4096, 384)
}
print(f"parameter keys: {parameters.keys()}")


def run(
    block_sharded_specs,
    n_size,
    transpose_mcast,
    batch_matrix_multiply,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    compute_kernel_config,
    *,
    device,
) -> list:
    (
        batch_sizes,
        m_size,
        k_size,
        per_core_height,
        per_core_width,
        num_cores_height,
        num_cores_width,
    ) = block_sharded_specs

    core_grid = device.compute_with_storage_grid_size()

    assert input_a_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
    if transpose_mcast:
        input_a_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_height - 1, num_cores_width - 1))}
            ),
            (per_core_height, per_core_width),
            ttnn.ShardOrientation.COL_MAJOR,
            False,
        )
    else:
        input_a_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_width - 1, num_cores_height - 1))}
            ),
            (per_core_height, per_core_width),
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
    input_a_memory_config.shard_spec = input_a_shard_spec

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
        compute_kernel_config=compute_kernel_config,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.989
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
