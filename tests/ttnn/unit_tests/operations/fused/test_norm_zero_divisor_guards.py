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


# A non-sharded TILE mask shorter than one input tile makes the softmax factories divide by mask_Ht = 0.
@pytest.mark.parametrize("mask_case", ["tile16", "zero_height"])
@pytest.mark.parametrize("sharded", [False, True], ids=["interleaved", "sharded"])
def test_scale_mask_softmax_rejects_short_tile_mask(device, expect_error, mask_case, sharded):
    if mask_case == "tile16":
        mask = ttnn.from_torch(
            torch.zeros(1, 1, 16, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, tile=ttnn.Tile((16, 32))
        )
    else:
        mask = ttnn.from_torch(torch.zeros(1, 1, 0, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {}
    mem = None
    if sharded:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
        )
        kwargs = dict(
            program_config=ttnn.SoftmaxShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(1, 1), subblock_w=1, block_h=1, block_w=2
            ),
            is_causal_mask=True,
        )
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    with expect_error(RuntimeError, "Non-sharded TILE mask must span at least one 32x32 input tile"):
        ttnn.scale_mask_softmax_in_place(x, 1.0, mask, **kwargs)


# Distributed layernorm pre/post all-gather divide the volume by H * W.
@pytest.mark.parametrize("shape", [(1, 1, 32, 0), (1, 1, 0, 32)], ids=["zero_width", "zero_height"])
@pytest.mark.parametrize("op", ["layer_norm_pre_all_gather", "rms_norm_pre_all_gather"])
def test_norm_pre_all_gather_rejects_zero_hw(device, expect_error, shape, op):
    x = ttnn.from_torch(torch.zeros(shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input must have non-zero height and width"):
        getattr(ttnn, op)(x)


@pytest.mark.parametrize("op", ["layer_norm_post_all_gather", "rms_norm_post_all_gather"])
def test_norm_post_all_gather_rejects_zero_hw(device, expect_error, op):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    stats = ttnn.from_torch(torch.zeros(1, 1, 32, 64).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input must have non-zero height and width"):
        getattr(ttnn, op)(x, stats)


# A sharded input skips both checks above and goes straight to the sharded layernorm, whose shard
# validation divided the volume by the width.
def test_norm_pre_all_gather_sharded_rejects_zero_width(device, expect_error):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(1, 1), subblock_w=1, block_h=1, block_w=1, inplace=False
    )
    with expect_error(RuntimeError, "Sharded input must have a non-zero width"):
        ttnn.layer_norm_pre_all_gather(x, program_config=program_config)
