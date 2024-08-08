# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from loguru import logger

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


parameters = {
    "default": {
        "matmul_specs": [
            # mcast 2d
            """(
            (2, 3),
            (1600, 224, 896),
            False,
            dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK),
            None,
        )""",
            # mcast 2d transposed
            """(
            (2, 3),
            (1600, 224, 896),
            False,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=5),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            None,
        )""",
            # mcast 2d with shard width > 1 TILE
            """(
            (2, 1),
            (128, 256, 512),
            False,
            dict(core_grid=ttnn.CoreGrid(y=2, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
        )""",
            # mcast in0
            """(
            (2, 3),
            (64, 32 * 7, 1024),
            False,
            dict(
                core_grid=ttnn.CoreGrid(y=1, x=7),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
            None,
        )""",
            # mcast in1
            """(
            (2, 3),
            (160 * 7, 64, 64),
            False,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            None,
        )""",
            # bmm
            """(
            (7, 7),
            (384, 64, 384),
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=7),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            ),
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=7),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            ),
        )""",
        ],
        "compute_kernel_config": [None],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.TILE_LAYOUT],
    }
}


def run(
    matmul_specs,
    compute_kernel_config,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    input_layout,
    *,
    device,
) -> list:
    (
        batch_sizes,
        input_shapes,
        batch_matrix_multiply,
        input_a_sharded_memory_config_specs,
        input_b_sharded_memory_config_specs,
    ) = matmul_specs

    (m_size, k_size, n_size) = input_shapes
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)
    if batch_matrix_multiply:
        input_shape_b = (*batch_sizes, k_size, n_size)

    input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG
    input_b_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    if input_a_sharded_memory_config_specs:
        input_a_memory_config = ttnn.create_sharded_memory_config(input_shape_a, **input_a_sharded_memory_config_specs)
    if input_b_sharded_memory_config_specs:
        input_b_memory_config = ttnn.create_sharded_memory_config(input_shape_b, **input_b_sharded_memory_config_specs)

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
    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=output_memory_config, dtype=output_dtype)
    e2e_perf = stop_measuring_time(start_time)
    output_tensor = ttnn.to_torch(output_tensor)

    expected_pcc = 0.99
    return [check_with_pcc(torch_output_tensor, output_tensor, expected_pcc), e2e_perf]
