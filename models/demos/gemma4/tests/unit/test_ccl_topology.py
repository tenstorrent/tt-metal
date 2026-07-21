# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 CCL topology / async / L1 env knobs."""

import pytest

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_HARD_MAX,
    PREFILL_SDPA_MAX_SEQ,
    prefill_short_lived_memcfg,
)
from models.demos.gemma4.tt.ccl import ccl_async_enabled, default_ccl_topology
from models.demos.gemma4.tt.dram_sharded import can_dram_shard


@pytest.mark.parametrize(
    "env,expected",
    [
        ("ring", ttnn.Topology.Ring),
        ("linear", ttnn.Topology.Linear),
        ("LINE", ttnn.Topology.Linear),
    ],
)
def test_ccl_topology_env_override(monkeypatch, env, expected):
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", env)
    assert default_ccl_topology() == expected


def test_ccl_async_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_CCL_ASYNC", raising=False)
    assert ccl_async_enabled() is False
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "1")
    assert ccl_async_enabled() is True


def test_prefill_l1_act_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_PREFILL_L1_ACT", raising=False)
    assert prefill_short_lived_memcfg() == ttnn.DRAM_MEMORY_CONFIG
    monkeypatch.setenv("GEMMA4_PREFILL_L1_ACT", "1")
    assert prefill_short_lived_memcfg() == ttnn.L1_MEMORY_CONFIG


def test_prefill_sdpa_max_seq_clamped_to_hard_max():
    """Env override must not raise the non-chunked SDPA path past 2^15."""
    assert PREFILL_SDPA_MAX_SEQ <= PREFILL_SDPA_HARD_MAX


def test_shared_mlp_down_shard_unguarded_at_tp8(monkeypatch):
    """intermediate=2112 @ TP=8 → down_k=264 is not DRAM-shardable; gate_up is."""
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert can_dram_shard(2816, 528)  # gate_up n at tp=8
    assert not can_dram_shard(264, 2816)  # down_k at tp=8


def test_dram_shard_disabled_off_blackhole(monkeypatch):
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    assert not can_dram_shard(2816, 528)


def test_dram_shard_31b_gate_up_fits_with_in0_cap(monkeypatch):
    """31B fused gate_up @ TP=4 previously overflowed L1 at in0_block_w=6; cap=2 fits."""
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    import ttnn

    # hidden=5376, gu_n=2*21504/4=10752
    assert can_dram_shard(5376, 10752, dtype=ttnn.bfloat16)
    assert can_dram_shard(5376, 10752, dtype=ttnn.bfloat8_b)
