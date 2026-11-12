# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def _height_sharded_on(xt, device, start, end):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(*end))})
    num_cores = grid.num_cores()
    shard_spec = ttnn.ShardSpec(
        grid, [xt.volume() // xt.padded_shape[-1] // num_cores, xt.padded_shape[-1]], ttnn.ShardOrientation.ROW_MAJOR
    )
    return xt.to(device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec))


@pytest.mark.parametrize("grid_start, grid_end", [((1, 0), (4, 0)), ((0, 1), (3, 1))])
def test_fill_cache_offset_shard_grid(grid_start, grid_end, device):
    # Shard grid not rooted at (0, 0): per-core runtime args must follow the shard cores (cache miss and hit).
    torch.manual_seed(0)
    num_users, num_heads, max_seq_len, head_dim, seq_len = 4, 1, 512, 64, 128
    cache = torch.randn([num_users, num_heads, max_seq_len, head_dim]).bfloat16().float()
    cachett = ttnn.Tensor(cache, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    for batch_idx in (1, 3):
        x = torch.randn([1, num_heads, seq_len, head_dim]).bfloat16().float()
        xt = _height_sharded_on(ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT), device, grid_start, grid_end)
        cachett = ttnn.fill_cache(cachett, xt, batch_idx)
        cache[batch_idx : batch_idx + 1, :, :seq_len, :] = x

    eq, output = comp_equal(cache, cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch())
    logger.info(output)
    assert eq


@pytest.mark.parametrize("grid_start, grid_end", [((2, 0), (3, 0)), ((0, 1), (1, 1))])
def test_update_cache_offset_shard_grid(grid_start, grid_end, device):
    # Shard grid not rooted at (0, 0): per-core runtime args must follow the shard cores (cache miss and hit).
    torch.manual_seed(0)
    num_users, num_heads, max_seq_len, head_dim = 32, 2, 1024, 64
    cache = torch.randn([num_users, num_heads, max_seq_len, head_dim]).bfloat16().float()
    cachett = ttnn.Tensor(cache, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    for update_idx in (64, 130):
        x = torch.randn([num_users, num_heads, 1, head_dim]).bfloat16().float()
        xt = ttnn.Tensor(x.permute(2, 1, 0, 3), ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
        xt = _height_sharded_on(xt, device, grid_start, grid_end)
        cachett = ttnn.update_cache(cachett, xt, update_idx)
        cache[:, :, update_idx : update_idx + 1, :] = x

    eq, output = comp_equal(cache, cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch())
    logger.info(output)
    assert eq
