# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# A shard shape with a 0 in a sharded dim used to reach an integer division while the op built its
# output TensorSpec, which killed the process with SIGFPE instead of raising. Any op that takes a
# sharded memory_config hits the same check; add, embedding and to_memory_config are three of them.


def _one_core_shard(layout, shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


@pytest.mark.parametrize(
    "layout, shard_shape",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, 64)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (64, 0)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (0, 64)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (64, 0)),
    ],
    ids=["height", "width", "block_zero_height", "block_zero_width"],
)
def test_add_zero_shard_dim(device, expect_error, layout, shard_shape):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Shard shape must be greater than 0 in each sharded dim"):
        ttnn.add(x, x, memory_config=_one_core_shard(layout, shard_shape))


def test_embedding_zero_shard_shape(device, expect_error):
    indices = ttnn.from_torch(torch.randint(0, 32, (1, 32)), dtype=ttnn.uint32, device=device)
    weights = ttnn.from_torch(torch.randn(32, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "Shard shape must be greater than 0 in each sharded dim"):
        ttnn.embedding(
            indices,
            weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_one_core_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, 64)),
        )


# A row-major tensor takes its width alignment from the shard width, so a 0 width fails while the
# TensorLayout is built, ahead of the TensorSpec check.
@pytest.mark.parametrize(
    "layout", [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED], ids=["width", "block"]
)
def test_row_major_zero_shard_width(device, expect_error, layout):
    with expect_error(RuntimeError, "Row Major width alignment must be greater than 0"):
        ttnn.from_torch(
            torch.randn(1, 1, 64, 64).bfloat16(),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_one_core_shard(layout, (64, 0)),
        )
