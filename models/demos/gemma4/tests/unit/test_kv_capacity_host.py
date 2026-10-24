# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""KV capacity arithmetic (``tt/kv_capacity.py``), host-only: the plugin's pool, the lane engine's per-lane vs
box users, the per-chip bytes (the replicated global head), the DRAM-limited pools of the dense Galaxy engines.

    pytest models/demos/gemma4/tests/unit/test_kv_capacity_host.py -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.tt.kv_capacity import (
    BFP8_BYTES,
    GEMMA4_GEOMETRY,
    engine_capacity_plan,
    kv_bytes_per_token_per_chip,
    lane_engine_capacity,
    local_kv_heads,
    plugin_pool_tokens,
    pool_tokens_for_dram,
    sharded_minimum_bytes_per_token_per_chip,
    sliding_cache_bytes_per_chip,
    users_at_context,
)

K256 = 262144
GB = 1e9


def test_the_plugin_pool_adds_the_output_reservation_and_the_hybrid_headroom():
    tokens, blocks = plugin_pool_tokens(3145728, 32, 128, sliding_window=1024, hybrid_groups=True)
    assert tokens == 3145728 + 128 * 32 + 1024 * 32 * 8 == 3411968 and blocks == 26656
    tokens, blocks = plugin_pool_tokens(262144, 32, 64, sliding_window=1024, hybrid_groups=False)
    assert tokens == 262144 + 64 * 32 and blocks == tokens // 64


def test_the_lane_engines_capacity_line_is_one_lanes_view_and_the_box_holds_lanes_times_it():
    pool, _ = plugin_pool_tokens(3145728, 32, 128, sliding_window=1024, hybrid_groups=True)
    per_lane, box = lane_engine_capacity(pool, 4, K256, 128, sliding_window=1024, sliding_groups=5)
    assert per_lane == 12 and box == 48, "the catalog's '12 users at 256K' is per lane; vLLM's line counts one lane"
    # 36 box users at 256K need 9 per lane: a 2.36M-token pool per column, 25 % below today's
    needed = 9 * K256 + 5 * (1024 + 128) * 9
    assert users_at_context(needed + 128 * 32, K256, 128, sliding_window=1024, sliding_groups=5) == 9
    # 128 users at <= 64K: 32 per lane, well inside today's pool (47 per lane at 64K)
    assert lane_engine_capacity(pool, 4, 65536, 128, sliding_window=1024, sliding_groups=5) == (47, 188)
    assert lane_engine_capacity(32 * (65536 + 5 * 1152), 4, 65536, 128, sliding_window=1024, sliding_groups=5)[0] == 32


def test_global_heads_fewer_than_the_tp_group_are_replicated_one_per_chip():
    assert (
        local_kv_heads(2, 8) == 1
        and local_kv_heads(8, 8) == 1
        and local_kv_heads(16, 8) == 2
        and local_kv_heads(1, 4) == 1
    )
    g = GEMMA4_GEOMETRY["26B"]
    per_chip = kv_bytes_per_token_per_chip(
        global_layers=g["global_layers"],
        global_kv_heads=g["global_kv_heads"],
        global_head_dim=g["global_head_dim"],
        tp=8,
    )
    assert per_chip == pytest.approx(5 * 2 * 512 * BFP8_BYTES) == pytest.approx(5440)
    floor = sharded_minimum_bytes_per_token_per_chip(global_layers=5, global_kv_heads=2, global_head_dim=512, tp=8)
    assert floor == pytest.approx(
        5440 / 4
    ), "two global heads over eight chips: each head on four chips, 4x the sharded floor"
    # the 10-23 reading: 3.78M tokens per column = 29 GB per chip -> 7.7 KB per token, of which the global pages are 5.4
    assert 29 * GB / 3.78e6 == pytest.approx(7672, rel=0.01)


def test_dense_galaxy_engines_per_chip_bytes_match_the_ledger():
    g31 = GEMMA4_GEOMETRY["31B"]
    assert kv_bytes_per_token_per_chip(
        global_layers=g31["global_layers"],
        global_kv_heads=g31["global_kv_heads"],
        global_head_dim=g31["global_head_dim"],
        tp=8,
    ) == pytest.approx(10880)
    assert kv_bytes_per_token_per_chip(
        global_layers=g31["global_layers"],
        global_kv_heads=g31["global_kv_heads"],
        global_head_dim=g31["global_head_dim"],
        tp=4,
    ) == pytest.approx(10880), "4 heads over 4 chips: one each, the same as TP8's replicated one"
    g12 = GEMMA4_GEOMETRY["12B"]
    assert kv_bytes_per_token_per_chip(
        global_layers=g12["global_layers"],
        global_kv_heads=g12["global_kv_heads"],
        global_head_dim=g12["global_head_dim"],
        tp=8,
    ) == pytest.approx(8704), "one global head: replicated on every chip at any TP"


def test_dense_galaxy_engine_pools_from_the_chips_dram():
    # 31B: 7.75 GB of weights per chip (ledger), ~2 GB for traces (0.96) and prefill activations -> 7 users; a 3 GB
    # reserve gives 6: the window's capacity line settles it
    plan31 = engine_capacity_plan(
        model=GEMMA4_GEOMETRY["31B"],
        dram_bytes=32 * GB,
        weights_bytes=7.75 * GB,
        reserve_bytes=2.0 * GB,
        context=K256,
        block_size=64,
        users_for_sliding=32,
        tp=8,
    )
    assert plan31["users"] == 7 and plan31["env_pool"] == 7 * K256 == 1835008
    assert (
        engine_capacity_plan(
            model=GEMMA4_GEOMETRY["31B"],
            dram_bytes=32 * GB,
            weights_bytes=7.75 * GB,
            reserve_bytes=3.0 * GB,
            context=K256,
            block_size=64,
            users_for_sliding=32,
            tp=8,
        )["users"]
        == 6
    )
    assert plan31["sliding_bytes"] == pytest.approx(50 * 2 * 2 * 256 * BFP8_BYTES * 32 * (1024 + 64))
    plan12 = engine_capacity_plan(
        model=GEMMA4_GEOMETRY["12B"],
        dram_bytes=32 * GB,
        weights_bytes=3.0 * GB,
        reserve_bytes=2.0 * GB,
        context=K256,
        block_size=64,
        users_for_sliding=32,
        tp=8,
    )
    assert (
        plan12["users"] == 11 and plan12["env_pool"] == 11 * K256 == 2883584
    ), "the catalog's 12B pool already sits at the DRAM floor (one replicated global head)"
    assert (
        pool_tokens_for_dram(
            1 * GB, weights_bytes=2 * GB, reserve_bytes=0, sliding_bytes=0, bytes_per_token=1000, block_size=64
        )
        == 0
    )


def test_sliding_caches_are_window_sized_per_user():
    b = sliding_cache_bytes_per_chip(
        sliding_layers=25, sliding_kv_heads=8, sliding_head_dim=256, tp=8, users=32, sliding_window=1024, block_size=128
    )
    assert b == pytest.approx(25 * 2 * 1 * 256 * BFP8_BYTES * 32 * 1152)
