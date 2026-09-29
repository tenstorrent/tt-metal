# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


# These zero arguments used to reach an integer division on the host, which killed the process
# with SIGFPE instead of raising.


def test_group_norm_num_groups_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "num_groups must be greater than 0"):
        ttnn.group_norm(x, num_groups=0)


# With a mask, the sharded arm divides by block_w * block_h before the block_h shard check runs.
def test_softmax_sharded_masked_block_h_zero(device, expect_error):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    mask = ttnn.from_torch(torch.zeros(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(1, 1), subblock_w=1, block_h=0, block_w=2
    )
    with expect_error(RuntimeError, "block_h must be greater than 0"):
        ttnn.scale_mask_softmax_in_place(x, 1.0, mask, program_config=program_config)
