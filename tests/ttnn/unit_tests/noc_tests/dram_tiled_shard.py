# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest

compute_grid = ttnn.CoreGrid(x=10, y=10)  # actual 13, 10 # 1,1 for just direct dram transfer
num_tiles_per_dim = 10
tensor_size = (32 * num_tiles_per_dim * compute_grid.y, 32 * num_tiles_per_dim * compute_grid.x)
# x,y=multiple of (compute_grid.y*compute_grid.x*32) <= sqrt(max elements per core * num cores) so that its shardable in any orientation and maxes out L1
# but numbers above should be good enough


@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.L1_MEMORY_CONFIG,
        (
            ttnn.create_sharded_memory_config(
                shape=tensor_size,
                core_grid=compute_grid,
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
        ),
        ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
        ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
        ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        ttnn.create_sharded_memory_config(
            shape=tensor_size,
            core_grid=compute_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        ),
    ],
)
def test_shard(device, memory_config):
    # Create input tensor
    torch_input = torch.rand(tensor_size, dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = ttnn.to_memory_config(tt_input, memory_config)

    # shard
    print(tt_output.layout)
    print(tt_output.memory_config())
