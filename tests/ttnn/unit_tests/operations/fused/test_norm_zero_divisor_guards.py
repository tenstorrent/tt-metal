# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# Check that groupnorm rejects 0 groups.
def test_group_norm_num_groups_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.group_norm(x, num_groups=0)


# Check that masked sharded softmax rejects a zero shard volume (block_w * block_h * tile_hw), both for
# block_h = 0 and for block_h = 2**53 with block_w = 2 and a 1024-element tile, whose product wraps to 0.
@pytest.mark.parametrize("block_h", [0, 1 << 53], ids=["zero", "product_wraps_to_zero"])
def test_softmax_sharded_masked_zero_shard_volume(device, expect_error, block_h):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    mask = ttnn.from_torch(torch.zeros(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(1, 1), subblock_w=1, block_h=block_h, block_w=2
    )
    with expect_error(RuntimeError, "block_h must be greater than 0"):
        ttnn.scale_mask_softmax_in_place(x, 1.0, mask, program_config=program_config)
