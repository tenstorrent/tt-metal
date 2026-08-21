# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 RMSNorm width-sharding and the prefill LN island.

Covers the shard spec ``tt/rms_norm.py`` derives for a given mesh and activation
height, and the env kill switches that opt the AR→LN→MLP island back out. Pure
host arithmetic over shapes and env vars; no device is opened.
"""

from .host_config_fakes import _FakeMeshDevice


def test_width_shard_spec_decode_matches_prefill_tile():
    """Decode (height=32) and prefill (height=128) share one layout builder."""
    import ttnn
    from models.demos.gemma4.tt.rms_norm import (
        _SHARDED_NORM_MAX_HEIGHT,
        decode_width_shard_memcfg,
        decode_width_shard_spec,
        width_shard_input_memcfg,
        width_shard_spec,
    )

    mesh = _FakeMeshDevice()
    dim = 5376
    decode_mem = decode_width_shard_memcfg(mesh, dim)
    prefill_mem = width_shard_input_memcfg(mesh, dim, 128)
    assert decode_mem is not None
    assert prefill_mem is not None
    assert decode_mem != prefill_mem  # different shard heights

    decode_spec = decode_width_shard_spec(mesh, dim)
    height128_spec = width_shard_spec(mesh, dim, 128)
    assert decode_spec[0] == width_shard_spec(mesh, dim, ttnn.TILE_SIZE)[0]
    assert height128_spec[1].block_h == 4  # 128 / 32
    assert width_shard_input_memcfg(mesh, dim, _SHARDED_NORM_MAX_HEIGHT + 32) is None


def test_sharded_norm_env_kill_switches(monkeypatch):
    """GEMMA4_SHARDED_NORM / GEMMA4_NORM_KEEP_SHARDED gate the AR→LN island."""
    from models.demos.gemma4.tt.rms_norm import norm_keep_sharded_enabled, sharded_norm_enabled

    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)
    monkeypatch.delenv("GEMMA4_NORM_KEEP_SHARDED", raising=False)
    assert sharded_norm_enabled()
    assert norm_keep_sharded_enabled()

    monkeypatch.setenv("GEMMA4_SHARDED_NORM", "0")
    assert not sharded_norm_enabled()
    monkeypatch.setenv("GEMMA4_NORM_KEEP_SHARDED", "false")
    assert not norm_keep_sharded_enabled()


def test_prefill_mlp_island_gates(monkeypatch):
    """Short dense prefill keeps the LN island; MoE / batch / long seq / env opt out."""
    from models.demos.gemma4.tt.rms_norm import (
        _PREFILL_ISLAND_MAX_HEIGHT,
        _SHARDED_NORM_MAX_HEIGHT,
        prefill_mlp_island_enabled,
    )

    monkeypatch.delenv("GEMMA4_PREFILL_ISLAND", raising=False)
    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)
    monkeypatch.delenv("GEMMA4_NORM_KEEP_SHARDED", raising=False)
    assert prefill_mlp_island_enabled(96)
    assert prefill_mlp_island_enabled(_PREFILL_ISLAND_MAX_HEIGHT)
    assert not prefill_mlp_island_enabled(_PREFILL_ISLAND_MAX_HEIGHT + 32)
    assert not prefill_mlp_island_enabled(_SHARDED_NORM_MAX_HEIGHT)
    assert not prefill_mlp_island_enabled(96, batch_size=2)
    assert not prefill_mlp_island_enabled(96, enable_moe=True)
    monkeypatch.setenv("GEMMA4_PREFILL_ISLAND", "0")
    assert not prefill_mlp_island_enabled(96)
