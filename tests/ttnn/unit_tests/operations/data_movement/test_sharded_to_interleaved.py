# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


def _grid(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


@pytest.mark.parametrize(
    "tensor_shape, layout, dtype, memory_layout, grid, shard_shape, output_mem_config",
    [
        (
            (1, 1, 8192, 256),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 8),
            (128, 256),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 1568, 1024),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 2),
            (98, 1024),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 4096, 256),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (8, 8),
            (512, 32),
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            (1, 1, 200, 320),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 4),
            (64, 96),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 1000, 48),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 8),
            (16, 48),
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            (1, 1, 224, 320),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (4, 4),
            (64, 96),
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            (1, 1, 2048, 512),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 8),
            (32, 512),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 4096, 256),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 4),
            (128, 256),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 128, 4096),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (8, 8),
            (128, 64),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 512, 64),
            ttnn.TILE_LAYOUT,
            ttnn.float32,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (4, 4),
            (32, 64),
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            (1, 1, 64, 8192),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 4),
            (2, 8192),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 1, 64, 8192),
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (8, 8),
            (64, 128),
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            (1, 1, 4096, 128),
            ttnn.TILE_LAYOUT,
            ttnn.float32,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 8),
            (64, 128),
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            (1, 32, 32, 256),
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (8, 4),
            (32, 256),
            ttnn.L1_MEMORY_CONFIG,
        ),
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_sharded_to_interleaved_exact(
    device, tensor_shape, layout, dtype, memory_layout, grid, shard_shape, output_mem_config, shard_orientation
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    shard_spec = ttnn.ShardSpec(_grid(*grid), shard_shape, shard_orientation)
    sharded_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)
    sharded = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=sharded_mem_config)

    expected = ttnn.to_torch(sharded)
    for _ in range(2):
        output = ttnn.sharded_to_interleaved(sharded, output_mem_config)
        assert_equal(expected, ttnn.to_torch(output))
