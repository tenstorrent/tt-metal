# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 CCL topology / async / packet / L1 env knobs."""

import json
import math
from pathlib import Path

import pytest

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_HARD_MAX,
    PREFILL_SDPA_MAX_SEQ,
    prefill_short_lived_memcfg,
)
from models.demos.gemma4.tt.ccl import ccl_async_enabled, default_ccl_packet_bytes, default_ccl_topology
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


class _FakeGrid:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeMesh:
    def __init__(self, n, grid=(8, 8)):
        self._n = n
        self._grid = grid

    def get_num_devices(self):
        return self._n

    def compute_with_storage_grid_size(self):
        return _FakeGrid(*self._grid)


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
    """WH T3K 1x8 MoE (26B-A4B): keep Linear — Ring regresses full-model PCC < 0.76.

    ``is_moe`` defaults True so a caller that forgets the flag cannot put 26B
    on Ring. Dense 12B/31B pass ``is_moe=False`` (see create_tt_model).
    """
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Linear
    assert default_ccl_topology(_FakeMesh(8), is_moe=True) == ttnn.Topology.Linear


def test_ccl_topology_ring_on_wh_8_dense(monkeypatch):
    """WH T3K 1x8 dense (12B/31B): Ring. BH 8-device is Ring regardless of MoE."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8), is_moe=False) == ttnn.Topology.Ring
    # N150 / N300 dense stay Linear (device count < 8).
    assert default_ccl_topology(_FakeMesh(1), is_moe=False) == ttnn.Topology.Linear
    assert default_ccl_topology(_FakeMesh(2), is_moe=False) == ttnn.Topology.Linear
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8), is_moe=True) == ttnn.Topology.Ring
    assert default_ccl_topology(_FakeMesh(4), is_moe=False) == ttnn.Topology.Linear


def test_bundled_configs_wh_t3k_topology_follows_moe(monkeypatch):
    """create_tt_model passes enable_moe_block into CCLManager; pin the pairing."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    root = Path(__file__).resolve().parents[2] / "configs"
    expected = {
        "gemma-4-12B-it": (False, ttnn.Topology.Ring),
        "gemma-4-31B-it": (False, ttnn.Topology.Ring),
        "gemma-4-26B-A4B-it": (True, ttnn.Topology.Linear),
        "gemma-4-E2B-it": (False, ttnn.Topology.Ring),
        "gemma-4-E4B-it": (False, ttnn.Topology.Ring),
    }
    t3k = _FakeMesh(8)
    for name, (want_moe, want_topo) in expected.items():
        cfg = json.loads((root / name / "config.json").read_text())
        text = cfg.get("text_config", cfg)
        is_moe = bool(text.get("enable_moe_block", False))
        assert is_moe is want_moe, name
        assert default_ccl_topology(t3k, is_moe=is_moe) == want_topo, name
        # Smaller WH meshes must not pick up the T3K Ring default.
        assert default_ccl_topology(_FakeMesh(1), is_moe=is_moe) == ttnn.Topology.Linear, name
        assert default_ccl_topology(_FakeMesh(2), is_moe=is_moe) == ttnn.Topology.Linear, name


def test_ccl_topology_env_override_beats_device_count(monkeypatch):
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "ring")
    assert default_ccl_topology(_FakeMesh(4)) == ttnn.Topology.Ring
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "linear")
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Linear


def test_ccl_async_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_CCL_ASYNC", raising=False)
    assert ccl_async_enabled() is False
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "1")
    assert ccl_async_enabled() is True


def test_default_ccl_packet_bytes_wormhole_packs_2048_tiles(monkeypatch):
    """WH fabric default 4352 B cannot hold an integer number of 2048 B pages."""
    monkeypatch.delenv("GEMMA4_CCL_PACKET_BYTES", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_packet_bytes() == 6144
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_packet_bytes() is None


def test_prefill_l1_act_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_PREFILL_L1_ACT", raising=False)
    assert prefill_short_lived_memcfg() == ttnn.DRAM_MEMORY_CONFIG
    monkeypatch.setenv("GEMMA4_PREFILL_L1_ACT", "1")
    assert prefill_short_lived_memcfg() == ttnn.L1_MEMORY_CONFIG


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


def test_wh_t3k_decode_gate_only_full_unharvested_t3k(monkeypatch):
    """12B/31B swept decode configs must not fire on BH, N150, or harvested WH."""
    from models.demos.gemma4.tt.dram_sharded import wh_t3k_decode_enabled, wh_t3k_decode_progcfg

    monkeypatch.delenv("GEMMA4_WH_T3K_DECODE_MM", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    assert wh_t3k_decode_enabled(_FakeMesh(8, (8, 8))) is True
    assert wh_t3k_decode_progcfg(_FakeMesh(8), 3840, 1024) is not None
    assert wh_t3k_decode_progcfg(_FakeMesh(8), 5376, 2048) is not None

    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert wh_t3k_decode_enabled(_FakeMesh(8, (8, 8))) is False
    assert wh_t3k_decode_progcfg(_FakeMesh(8), 3840, 1024) is None

    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    assert wh_t3k_decode_enabled(_FakeMesh(1, (8, 8))) is False  # N150
    assert wh_t3k_decode_enabled(_FakeMesh(2, (8, 8))) is False  # N300
    assert wh_t3k_decode_enabled(_FakeMesh(4, (8, 8))) is False
    assert wh_t3k_decode_enabled(_FakeMesh(8, (8, 7))) is False  # x2-harvested
    monkeypatch.setenv("GEMMA4_WH_T3K_DECODE_MM", "0")
    assert wh_t3k_decode_enabled(_FakeMesh(8, (8, 8))) is False


def _dense_decode_kn(
    hidden, heads, kv_heads, head_dim, intermediate, tp, *, global_hd=None, global_kv=None, kv_replicated=False
):
    """Per-device (k, n) for the six decode matmuls (qkv / o_proj × slide|global, gate_up, down)."""
    hd = head_dim
    q_per = (heads // tp) * hd
    kv_per = hd if kv_replicated else (kv_heads // tp) * hd
    qkv = (hidden, q_per + 2 * kv_per)
    o_proj = (q_per, hidden)
    gate_up = (hidden, 2 * (intermediate // tp))
    down = (intermediate // tp, hidden)
    shapes = {"qkv_slide": qkv, "o_proj_slide": o_proj, "gate_up": gate_up, "down": down}
    if global_hd is not None:
        gq = (heads // tp) * global_hd
        # Fewer KV heads than TP (12B global nkv=1 at TP=8) are replicated.
        if kv_replicated or global_kv is None or global_kv < tp:
            gkv = global_hd
        else:
            gkv = (global_kv // tp) * global_hd
        shapes["qkv_global"] = (hidden, gq + 2 * gkv)
        shapes["o_proj_global"] = (gq, hidden)
    return shapes


def test_wh_t3k_decode_table_hits_only_12b_31b_tp8():
    """E2B / E4B / 26B-A4B (k,n) at TP 1/2/4/8 must miss the T3K table; 12B/31B TP=8 hit."""
    from models.demos.gemma4.tt.dram_sharded import _WH_T3K_DECODE_1D

    keys = set(_WH_T3K_DECODE_1D)

    def all_kn(hidden, heads, kv_heads, head_dim, intermediate, **g):
        out = []
        for tp in (1, 2, 4, 8):
            if heads % tp:
                continue
            out.extend(_dense_decode_kn(hidden, heads, kv_heads, head_dim, intermediate, tp, **g).values())
        return out

    other = []
    other += all_kn(1536, 8, 1, 256, 6144, global_hd=512, global_kv=1, kv_replicated=True)  # E2B
    other += all_kn(2560, 8, 2, 256, 10240, global_hd=512, global_kv=1)  # E4B
    other += all_kn(2816, 16, 8, 256, 2112, global_hd=512, global_kv=2)  # 26B-A4B
    assert not (set(other) & keys), f"non-12B/31B shapes hit T3K table: {set(other) & keys}"

    # 12B TP=8: the six swept shapes. 31B: only sliding qkv was a table win.
    s12 = _dense_decode_kn(3840, 16, 8, 256, 15360, 8, global_hd=512, global_kv=1)
    assert s12["qkv_slide"] == (3840, 1024)
    assert s12["qkv_global"] == (3840, 2048)
    assert s12["gate_up"] == (3840, 3840)
    assert s12["down"] == (1920, 3840)
    assert s12["o_proj_slide"] == (512, 3840)
    assert s12["o_proj_global"] == (1024, 3840)
    assert set(s12.values()) <= keys

    s31 = _dense_decode_kn(5376, 32, 16, 256, 21504, 8, global_hd=512, global_kv=4)
    assert s31["qkv_slide"] == (5376, 2048)
    assert s31["qkv_slide"] in keys
    # 31B MLP / o_proj / qkv-global were swept; auto won — must stay absent.
    assert s31["gate_up"] not in keys
    assert s31["down"] not in keys
    assert s31["o_proj_slide"] not in keys
    assert s31["o_proj_global"] not in keys
    assert s31["qkv_global"] not in keys
