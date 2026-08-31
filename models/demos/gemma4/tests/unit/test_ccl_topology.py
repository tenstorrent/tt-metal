# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 CCL topology / async / L1 env knobs."""

import math
from pathlib import Path

import pytest

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_HARD_MAX,
    PREFILL_SDPA_MAX_SEQ,
    prefill_short_lived_memcfg,
    prefill_tensor_memcfg,
    prefill_tilize_memcfg,
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


class _FakeMesh:
    def __init__(self, n):
        self._n = n

    def get_num_devices(self):
        return self._n


def test_ccl_topology_linear_on_4_device_mesh(monkeypatch):
    """QB2 / P300x2 opened as 1x4: Ring drops 12B full-model PCC below 0.94."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(4)) == ttnn.Topology.Linear


def test_ccl_topology_ring_on_bh_8_device_mesh(monkeypatch):
    """LoudBox P150x8: Ring remains the TTFT-swept default on Blackhole."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Ring


def test_ccl_topology_linear_on_wh_8_device_mesh(monkeypatch):
    """T3K keeps main's validated Linear default; Ring remains an explicit opt-in."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8), is_moe=False) == ttnn.Topology.Linear


def test_ccl_topology_env_override_beats_device_count(monkeypatch):
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "ring")
    assert default_ccl_topology(_FakeMesh(4)) == ttnn.Topology.Ring
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "linear")
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Linear


def test_ccl_async_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_CCL_ASYNC", raising=False)
    monkeypatch.delenv("GEMMA4_CCL_ASYNC_PREFILL", raising=False)
    assert ccl_async_enabled() is False
    assert ccl_async_enabled(32) is False
    assert ccl_async_enabled(2048) is True
    monkeypatch.setenv("GEMMA4_CCL_ASYNC_PREFILL", "0")
    assert ccl_async_enabled(2048) is False
    monkeypatch.delenv("GEMMA4_CCL_ASYNC_PREFILL", raising=False)
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "1")
    assert ccl_async_enabled() is True
    assert ccl_async_enabled(32) is True
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "0")
    assert ccl_async_enabled(2048) is False


def test_prefill_l1_act_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_PREFILL_L1_ACT", raising=False)
    assert prefill_short_lived_memcfg() == ttnn.DRAM_MEMORY_CONFIG
    monkeypatch.setenv("GEMMA4_PREFILL_L1_ACT", "1")
    assert prefill_short_lived_memcfg() == ttnn.L1_MEMORY_CONFIG


def test_prefill_tensor_memcfg_size_budget(monkeypatch):
    """Short activations → L1; over budget / disabled → DRAM."""
    monkeypatch.delenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", raising=False)
    # 128 x 5376 BF16 ≈ 1.3 MiB < 4 MiB default
    assert prefill_tilize_memcfg(128, 5376) == ttnn.L1_MEMORY_CONFIG
    # RoPE slice 128 x 256 BF16 ≈ 64 KiB
    assert prefill_tensor_memcfg(128 * 256) == ttnn.L1_MEMORY_CONFIG
    # 512 x 5376 BF16 ≈ 5.3 MiB > 4 MiB
    assert prefill_tilize_memcfg(512, 5376) == ttnn.DRAM_MEMORY_CONFIG
    monkeypatch.setenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", "0")
    assert prefill_tilize_memcfg(128, 5376) == ttnn.DRAM_MEMORY_CONFIG


def test_prefill_sdpa_max_seq_clamped_to_hard_max():
    """Env override must not raise the non-chunked SDPA path past 2^15."""
    assert PREFILL_SDPA_MAX_SEQ <= PREFILL_SDPA_HARD_MAX


def test_shared_mlp_down_shard_unguarded_at_tp8(monkeypatch):
    """Unpadded intermediate=2112 @ TP=8 → down_k=264 is not DRAM-shardable.

    SharedMLP now pads to 288/device before sharding; this guards the raw shape.
    """
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert can_dram_shard(2816, 528)  # gate_up n at tp=8 (unpadded half*2)
    assert not can_dram_shard(264, 2816)  # raw down_k
    assert can_dram_shard(288, 2816)  # padded down_k used by SharedMLP


def test_dram_shard_disabled_off_blackhole(monkeypatch):
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    assert not can_dram_shard(2816, 528)


def test_dram_shard_31b_gate_up_fits_with_l1_aware_in0(monkeypatch):
    """31B fused gate_up @ TP=4: L1-aware in0 shrink keeps the shape shardable."""
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    import ttnn
    from models.demos.gemma4.tt.dram_sharded import decode_progcfg

    # hidden=5376, gu_n=2*21504/4=10752
    assert can_dram_shard(5376, 10752, dtype=ttnn.bfloat16)
    assert can_dram_shard(5376, 10752, dtype=ttnn.bfloat8_b)
    pc = decode_progcfg(32, 5376, 10752, dtype=ttnn.bfloat16)
    assert pc.in0_block_w >= 1


def test_decode_progcfg_covers_full_n_tiles(monkeypatch):
    """per_core_N * num_cores must cover padded N — K-only grids used to truncate.

    31B wqkv at TP=8: k=5376, n=2048 → old K-only 28-core grid left
    n_tiles % cores != 0 and silently wrong PCC (tt_transformers warning).
    """
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    from models.demos.gemma4.tt.dram_sharded import TILE_SIZE, _decode_core_grid, _padded_n_tiles, decode_progcfg

    k, n = 5376, 2048
    assert can_dram_shard(k, n)
    _r, _c, num_cores = _decode_core_grid(k, n)
    pc = decode_progcfg(TILE_SIZE, k, n)
    assert pc.per_core_N * num_cores >= math.ceil(n / TILE_SIZE)
    assert _padded_n_tiles(n) % num_cores == 0


def test_prefill_progcfg_in0_block_w_divides_kt():
    """26B padded down_proj K=288 → Kt=9; in0_block_w must divide Kt."""
    from models.demos.gemma4.tt.dram_sharded import prefill_progcfg

    pc = prefill_progcfg(m=512, k=288, n=2816)
    k_tiles = (288 + 31) // 32
    assert k_tiles % pc.in0_block_w == 0


def test_weight_cache_path_qualified_by_mesh(tmp_path, monkeypatch):
    """TP=4 on 1x4 vs 2x4 must not share tensorbin directories when mesh dirs are used."""
    import ttnn
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.delenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", raising=False)
    args = Gemma4ModelArgs()
    args.model_cache_path = tmp_path
    # Empty caches → write into mesh-qualified paths (cold start).
    p_1x4 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4))
    p_2x4 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(2, 4))
    p_1x1 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 1))
    assert "mesh1x4" in str(p_1x4)
    assert "mesh2x4" in str(p_2x4)
    assert p_1x4 != p_2x4
    assert "mesh" not in p_1x1.name


def test_weight_cache_path_reuses_legacy_when_mesh_empty(tmp_path, monkeypatch):
    """CI MLPerf: empty mesh dir + warm legacy → reuse legacy (avoid cold 31B rebuild)."""
    import ttnn
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.delenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", raising=False)
    legacy = tmp_path / "tensor_cache_bf16"
    legacy.mkdir()
    (legacy / "embed.tensorbin").write_text("x")
    args = Gemma4ModelArgs()
    args.model_cache_path = tmp_path
    assert args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4)) == legacy


def test_weight_cache_path_mesh_only_ignores_legacy(tmp_path, monkeypatch):
    import ttnn
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.setenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", "1")
    legacy = tmp_path / "tensor_cache_bf16"
    legacy.mkdir()
    (legacy / "embed.tensorbin").write_text("x")
    args = Gemma4ModelArgs()
    args.model_cache_path = tmp_path
    p = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4))
    assert "mesh1x4" in str(p)
    assert p != legacy


def test_weight_cache_path_ro_mount_falls_back_writable(tmp_path, monkeypatch):
    """CI MLPerf :ro — mkdir(tensor_cache_*) must not raise Errno 30; mirror under TT_METAL_HOME."""
    import errno

    import ttnn
    from models.demos.gemma4.tt import model_config as mc
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.delenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", raising=False)
    monkeypatch.setenv("TT_METAL_HOME", str(tmp_path / "metal_home"))
    ro_root = tmp_path / "mlperf_ro" / "google--gemma-4-12B-it"
    ro_root.mkdir(parents=True)

    real_mkdir = Path.mkdir

    def _ro_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if "mlperf_ro" in self.parts:
            raise OSError(errno.EROFS, "Read-only file system", str(self))
        return real_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", _ro_mkdir)
    args = Gemma4ModelArgs()
    args.model_cache_path = ro_root
    p = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 1))
    assert p.is_dir()
    assert "gemma4_tt_cache" in p.parts
    assert p.name == "tensor_cache_bf16"
    # Multi-device cold path on RO also mirrors (no warm legacy).
    p4 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4))
    assert p4.is_dir()
    assert "mesh1x4" in p4.name
    assert "gemma4_tt_cache" in p4.parts
    # Sanity: helper used by resolve path.
    assert mc._ensure_cache_dir(ro_root / "nested").is_dir()
