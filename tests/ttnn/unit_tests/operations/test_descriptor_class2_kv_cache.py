# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Varying-value cache-HIT correctness tests for the Metal 2.0 descriptor migration of the
kv_cache op family (UpdateKVCacheOperation FILL/UPDATE + ZeroCacheRangeOperation).

These ops have a CUSTOM compute_program_hash that DELIBERATELY EXCLUDES the per-call
values batch_idx / update_idx / batch_offset (and start_page/end_page for zero_cache_range)
from the program-cache key, so two calls differing only in those values are a cache HIT and
reuse one cached program. Those excluded values feed kernel runtime args (cache_start_id,
tile_update_offset, batch_read_offset, page_start/page_end) plus raw-baked tensor addresses.

On a cache hit those args would be STALE unless the op's get_dynamic_runtime_args re-applies
them. We set TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 BEFORE importing ttnn so the
framework FORBIDS the (correct-but-slow) descriptor rebuild and must take the fast path. Then,
for each op, we warm up once and call AGAIN with a DIFFERENT frozen value (forcing a cache hit
where the frozen value changed) and assert the output is correct for the NEW value. This proves
get_dynamic_runtime_args actually re-applies the changed value.
"""

import os

os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.common.utility_functions import run_for_blackhole, skip_for_blackhole


# ---------------------------------------------------------------------------
# FILL: fill_cache with a varying update_idx (hash-excluded -> cache hit)
# ---------------------------------------------------------------------------
@skip_for_blackhole("Mismatching on BH, see #12349")
def test_fill_cache_varying_update_idx_cache_hit(device):
    num_users, num_heads, max_seq_len, head_dim = 8, 2, 1024, 64
    slab = 32  # input seq_len; update_idx must be a multiple of TILE_HEIGHT
    dtype = ttnn.bfloat16

    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    torch_cache = torch.randn(cache_shape).bfloat16().float()
    cachett = ttnn.Tensor(torch_cache, dtype).to(ttnn.TILE_LAYOUT).to(device)

    def fill(batch_idx, update_idx):
        x = torch.randn([1, num_heads, slab, head_dim]).bfloat16().float()
        xt = ttnn.Tensor(x, dtype).to(ttnn.TILE_LAYOUT).to(device)
        nonlocal cachett
        cachett = ttnn.fill_cache(cachett, xt, batch_idx, update_idx=update_idx)
        torch_cache[batch_idx : batch_idx + 1, :, update_idx : update_idx + slab, :] = x

    # Warm up the program cache (miss): one update_idx.
    fill(batch_idx=0, update_idx=0)
    # Cache HIT, but with a DIFFERENT update_idx AND batch_idx (both hash-excluded).
    # If get_dynamic_runtime_args failed to re-apply cache_start_id, this would write to the
    # wrong place and the assert against the NEW-index torch reference would fail.
    fill(batch_idx=3, update_idx=256)
    fill(batch_idx=3, update_idx=512)

    got = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    eq, out = comp_equal(torch_cache, got)
    assert eq, out


# ---------------------------------------------------------------------------
# UPDATE (decode): update_cache with a varying update_idx + batch_offset
# ---------------------------------------------------------------------------
@skip_for_blackhole("Mismatching on BH, see #12349")
def test_update_cache_varying_idx_cache_hit(device):
    num_users, num_heads, max_seq_len, head_dim = 16, 2, 2048, 64
    dtype = ttnn.bfloat16

    cache_shape = [num_users, num_heads, max_seq_len, head_dim]
    torch_cache = torch.randn(cache_shape).bfloat16().float()
    cachett = ttnn.Tensor(torch_cache, dtype).to(ttnn.TILE_LAYOUT).to(device)

    def update(cache_idx, batch_offset):
        # decode input: [num_users, num_heads, 1, head_dim] padded to 32 along batch with batch_offset
        x = torch.randn([num_users, num_heads, 1, head_dim]).bfloat16().float()
        x_new = x.clone()
        x_new = torch.cat((torch.zeros(batch_offset, num_heads, 1, head_dim), x_new), dim=0)
        x_new = torch.cat((x_new, torch.zeros(32 - num_users - batch_offset, num_heads, 1, head_dim)), dim=0)
        xt = ttnn.Tensor(x_new.permute(2, 1, 0, 3), dtype).to(ttnn.TILE_LAYOUT).to(device)
        nonlocal cachett
        cachett = ttnn.update_cache(cachett, xt, cache_idx, batch_offset=batch_offset)
        torch_cache[0:num_users, 0:num_heads, cache_idx : cache_idx + 1, 0:head_dim] = x
        return x

    # Warm up (miss).
    update(cache_idx=0, batch_offset=0)
    # Cache HIT with a DIFFERENT update_idx (tile_update_offset + cache_start_id change) AND a
    # DIFFERENT batch_offset (batch_read_offset changes). Both are hash-excluded.
    x1 = update(cache_idx=37, batch_offset=8)
    x2 = update(cache_idx=129, batch_offset=0)

    got = cachett.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    # Whole-cache equality covers every prior write at its NEW index.
    eq_cache, out_cache = comp_equal(torch_cache, got)
    # Explicitly check the most-recent updated slices land at the NEW indices.
    eq_u1, out_u1 = comp_equal(x1, got[0:num_users, 0:num_heads, 37:38, 0:head_dim])
    eq_u2, out_u2 = comp_equal(x2, got[0:num_users, 0:num_heads, 129:130, 0:head_dim])
    assert eq_cache, out_cache
    assert eq_u1, out_u1
    assert eq_u2, out_u2


# ---------------------------------------------------------------------------
# zero_cache_range: varying [start_page, end_page) (hash-excluded -> cache hit)
# ---------------------------------------------------------------------------
BH_NUM_DRAM_BANKS = 8
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def _make_nd_sharded_ones_cache(device, seq_len, head_dim):
    torch_cache = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    kv_mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)
    return ttnn.from_torch(
        torch_cache, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=kv_mem_config
    )


@run_for_blackhole()
def test_zero_cache_range_varying_range_cache_hit(device):
    seq_len, head_dim = 3200, 576
    TILE_HEIGHT = 32

    def tile_bounds(start_token, end_token):
        ts = (start_token // TILE_HEIGHT) * TILE_HEIGHT
        te = ((end_token + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
        return ts, min(te, seq_len)

    def run_and_check(start_token, end_token):
        tt_cache = _make_nd_sharded_ones_cache(device, seq_len, head_dim)
        tt_cache = ttnn.kv_cache.zero_cache_range(tt_cache, start_token, end_token)
        result = ttnn.to_torch(tt_cache).to(torch.float32)
        ts, te = tile_bounds(start_token, end_token)
        if ts > 0:
            assert torch.all(result[:, :, :ts, :] != 0), f"before {ts} should be non-zero"
        assert torch.all(result[:, :, ts:te, :] == 0), f"[{ts}:{te}] should be zero"
        if te < seq_len:
            assert torch.all(result[:, :, te:, :] != 0), f"after {te} should be non-zero"

    # Warm up (miss): one page range.
    run_and_check(0, 32)
    # Cache HITs with DIFFERENT page ranges (start_page/end_page are hash-excluded). If
    # get_dynamic_runtime_args failed to re-apply page_start/page_end, the wrong region would be
    # zeroed and these asserts (built against the NEW range) would fail.
    run_and_check(64, 128)
    run_and_check(50, 256)
