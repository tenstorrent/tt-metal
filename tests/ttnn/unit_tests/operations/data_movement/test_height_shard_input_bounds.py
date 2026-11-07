# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("shard_width", [4, 16])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_height_shard_wider_than_source(device, shard_width, memory_config):
    """Run with TT_METAL_WATCHER=1 to catch reads beyond the last L1 row."""
    values = (torch.arange(200 * 4, dtype=torch.int32) * 37 - 10000).reshape(1, 1, 200, 4)
    source = ttnn.from_torch(
        values, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    sharded_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [13, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    sharded = ttnn.to_memory_config(source, sharded_memory)
    result = ttnn.to_memory_config(sharded, ttnn.DRAM_MEMORY_CONFIG)
    torch.testing.assert_close(ttnn.to_torch(result), values, rtol=0, atol=0)
