# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize(
    "memory_layout",
    [ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
    ids=["block", "width"],
)
@pytest.mark.parametrize("shard_shape", [(1, 7), (32, 7), (1, 63), (32, 63)], ids=["1x7", "32x7", "1x63", "32x63"])
def test_copy_row_major_oversized_shard(device, memory_layout, shard_shape):
    """The first input/output page can exceed the logical row; both directions must finish exactly."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    rows = shard_shape[0] if memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED else 1
    for offset in (0, 32):
        host = (torch.arange(rows * 7, dtype=torch.float32) + offset).reshape(rows, 7).to(torch.bfloat16)
        source = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        sharded = ttnn.to_memory_config(source, config)
        assert torch.equal(ttnn.to_torch(sharded), host)
        restored = ttnn.to_memory_config(sharded, ttnn.L1_MEMORY_CONFIG)
        assert torch.equal(ttnn.to_torch(restored), host)
