# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest

compute_grid = ttnn.CoreGrid(x=10, y=10)  # actual 13, 10
num_tiles_per_dim = 10
tensor_size = (32 * num_tiles_per_dim * compute_grid.y, 32 * num_tiles_per_dim * compute_grid.x)
# x,y=multiple of (compute_grid.y*compute_grid.x*32) <= sqrt(max elements per core * num cores) so that its shardable in any orientation and maxes out L1
# but numbers above should be good enough

# tensor width or height must be divisible by grid widht or height or total cores
"""
@pytest.mark.parametrize(
    "initial_memory_config, new_memory_config",
    [
        (ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
        )

    ],
)
"""


@pytest.mark.parametrize(
    "initial_memory_config, initial_memory_config_str",
    [
        (ttnn.L1_MEMORY_CONFIG, "INTR"),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "BS-RM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            "BS-CM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "WS-RM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            "WS-CM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "HS-RM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            "HS-CM",
        ),
    ],
)
@pytest.mark.parametrize(
    "new_memory_config, new_memory_config_str",
    [
        (ttnn.L1_MEMORY_CONFIG, "INTR"),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "BS-RM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            "BS-CM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "WS-RM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            "WS-CM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            "HS-RM",
        ),
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            "HS-CM",
        ),
    ],
)
def test_shard(device, initial_memory_config, new_memory_config, initial_memory_config_str, new_memory_config_str):
    # Create input tensor
    torch_input = torch.rand(tensor_size, dtype=torch.bfloat16)

    # print(initial_memory_config)
    # print(new_memory_config)
    if initial_memory_config_str != new_memory_config_str:
        print(f"test config: {initial_memory_config_str}->{new_memory_config_str}")

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=initial_memory_config,
    )

    tt_output = ttnn.to_memory_config(tt_input, new_memory_config)

    # shard
    print(tt_output.layout)
    print(tt_output.memory_config())
