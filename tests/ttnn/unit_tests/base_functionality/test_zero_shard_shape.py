# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


# A shard shape with a 0 dim used to reach an integer division while the op built its output
# TensorSpec, which killed the process with SIGFPE instead of raising. Any op that takes a sharded
# memory_config hits the same check; add and embedding are two of them.


def _zero_height_shard():
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (0, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )


def test_add_zero_shard_shape(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Shard shape must be greater than 0 in each sharded dim"):
        ttnn.add(x, x, memory_config=_zero_height_shard())


def test_embedding_zero_shard_shape(device, expect_error):
    indices = ttnn.from_torch(torch.randint(0, 32, (1, 32)), dtype=ttnn.uint32, device=device)
    weights = ttnn.from_torch(torch.randn(32, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "Shard shape must be greater than 0 in each sharded dim"):
        ttnn.embedding(indices, weights, layout=ttnn.TILE_LAYOUT, memory_config=_zero_height_shard())
