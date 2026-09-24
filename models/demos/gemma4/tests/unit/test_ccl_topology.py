# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 CCL topology / async / packet / L1 env knobs."""

import math
from pathlib import Path

import pytest

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_HARD_MAX,
    PREFILL_SDPA_MAX_SEQ,
    prefill_short_lived_memcfg,
)
from models.demos.gemma4.tt.ccl import (
    LINEAR_PIN_MIN_SEQ_LEN,
    ccl_async_enabled,
    default_ccl_packet_bytes,
    default_ccl_topology,
    effective_pinned_ccl_topology,
)
from models.demos.gemma4.tt.dram_sharded import can_dram_shard, swept_decode_enabled


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


def test_ccl_topology_ring_on_wh_t3k_dense(monkeypatch):
    """WH T3K 1x8 dense (12B/31B): Ring. Every other WH mesh stays Linear."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8), is_moe=False) == ttnn.Topology.Ring
    # N150 / N300 dense stay Linear (device count < 8).
    assert default_ccl_topology(_FakeMesh(1), is_moe=False) == ttnn.Topology.Linear
    assert default_ccl_topology(_FakeMesh(2), is_moe=False) == ttnn.Topology.Linear
    # An x2-harvested T3K (8x7 grid) is outside the sweep.
    assert default_ccl_topology(_FakeMesh(8, (8, 7)), is_moe=False) == ttnn.Topology.Linear
    # WH Galaxy is n>=8 but not a T3K.
    assert default_ccl_topology(_FakeMesh(32), is_moe=False) == ttnn.Topology.Linear
    # Blackhole keeps the n>=8 rule it already had, MoE or not.
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8), is_moe=True) == ttnn.Topology.Ring
    assert default_ccl_topology(_FakeMesh(4), is_moe=False) == ttnn.Topology.Linear


def test_31b_linear_pin_only_at_128k(monkeypatch):
    """Dense Linear pin applies at 128k; shorter seq and Ring/None pins pass through."""
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    linear = ttnn.Topology.Linear
    assert effective_pinned_ccl_topology(linear, is_moe=False, max_seq_len=1024) is None
    assert effective_pinned_ccl_topology(linear, is_moe=False, max_seq_len=64 * 1024) is None
    assert effective_pinned_ccl_topology(linear, is_moe=False, max_seq_len=LINEAR_PIN_MIN_SEQ_LEN) is linear
    assert effective_pinned_ccl_topology(linear, is_moe=False, max_seq_len=None) is linear
    # MoE keeps Linear at every length, and a Ring / absent pin passes through.
    assert effective_pinned_ccl_topology(linear, is_moe=True, max_seq_len=1024) is linear
    assert effective_pinned_ccl_topology(None, is_moe=False, max_seq_len=1024) is None
    assert effective_pinned_ccl_topology(ttnn.Topology.Ring, is_moe=False, max_seq_len=1024) is ttnn.Topology.Ring
    # Blackhole never takes a dense Linear pin: the loop it fixes is Wormhole's.
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert effective_pinned_ccl_topology(linear, is_moe=False, max_seq_len=262144) is None
    assert effective_pinned_ccl_topology(linear, is_moe=True, max_seq_len=262144) is linear


def test_tuned_decode_gate_only_full_unharvested_t3k(monkeypatch):
    """12B/31B T3K decode knobs must not fire on BH, N150/N300, or a harvested WH."""
    from models.demos.gemma4.tt.dram_sharded import (
        decode_1d_matmul_config,
        is_t3k_dense_target,
        is_t3k_mesh,
        lm_head_decode_config,
        wh_t3k_decode_progcfg,
    )

    class _Cfg:
        def __init__(self, moe=False, pli=0):
            self.enable_moe_block = moe
            self.hidden_size_per_layer_input = pli

    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    t3k = _FakeMesh(8, (8, 8))
    assert is_t3k_mesh(t3k) is True
    assert is_t3k_dense_target(t3k, _Cfg()) is True
    assert is_t3k_dense_target(t3k, _Cfg(moe=True)) is False  # 26B-A4B
    assert is_t3k_dense_target(t3k, _Cfg(pli=256)) is False  # E2B / E4B
    # Every other Wormhole mesh, and a harvested T3K, fails the mesh half.
    for mesh in (_FakeMesh(1, (8, 8)), _FakeMesh(2, (8, 8)), _FakeMesh(32, (8, 8)), _FakeMesh(8, (8, 7))):
        assert is_t3k_mesh(mesh) is False
        assert is_t3k_dense_target(mesh, _Cfg()) is False

    assert wh_t3k_decode_progcfg(t3k, 3840, 1024, tuned_decode=True) is not None
    assert wh_t3k_decode_progcfg(t3k, 3840, 3840, tuned_decode=True) is not None
    # 31B sliding qkv is deliberately not in the table; see the table comment.
    assert wh_t3k_decode_progcfg(t3k, 5376, 2048, tuned_decode=True) is None

    # Off the gate every builder declines, so the call sites keep ttnn auto.
    assert wh_t3k_decode_progcfg(t3k, 3840, 1024, tuned_decode=False) is None
    assert decode_1d_matmul_config(t3k, 3840, 1024, tuned_decode=False) is None
    assert lm_head_decode_config(t3k, 32, 3840, 32768, tuned_decode=False) == (None, None, None)

    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert is_t3k_mesh(t3k) is False
    assert is_t3k_dense_target(t3k, _Cfg()) is False


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
    # Auto-enable is limited to tall activations on the tuned-prefill target.
    assert ccl_async_enabled(2048) is False
    assert ccl_async_enabled(2016, tuned_prefill=True) is False
    assert ccl_async_enabled(2048, tuned_prefill=True) is True
    monkeypatch.setenv("GEMMA4_CCL_ASYNC_PREFILL", "0")
    assert ccl_async_enabled(2048, tuned_prefill=True) is False
    monkeypatch.delenv("GEMMA4_CCL_ASYNC_PREFILL", raising=False)
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "1")
    assert ccl_async_enabled() is True
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "0")
    assert ccl_async_enabled(2048, tuned_prefill=True) is False


def test_default_ccl_packet_bytes_only_on_a_wormhole_t3k(monkeypatch):
    """WH fabric default 4352 B cannot hold an integer number of 2048 B pages.

    T3K is the cluster that was measured; every other one keeps the default,
    because a non-default payload is Fabric-wide.
    """
    monkeypatch.delenv("GEMMA4_CCL_PACKET_BYTES", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_t3k_cluster", lambda: True)
    assert default_ccl_packet_bytes() == 6144
    # N150 / N300 / WH Galaxy are Wormhole but not T3K.
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_t3k_cluster", lambda: False)
    assert default_ccl_packet_bytes() is None
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_packet_bytes() is None


def test_t3k_cluster_predicate(monkeypatch):
    """Cluster-type read, so device_params can gate before a mesh is open."""
    from models.demos.gemma4.tt.ccl import is_t3k_cluster

    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    monkeypatch.setattr("ttnn.cluster.get_cluster_type", lambda: ttnn.cluster.ClusterType.T3K)
    assert is_t3k_cluster() is True
    for other in ("N150", "N300", "GALAXY", "TG"):
        monkeypatch.setattr("ttnn.cluster.get_cluster_type", lambda o=other: getattr(ttnn.cluster.ClusterType, o))
        assert is_t3k_cluster() is False, other
    monkeypatch.setattr("ttnn.cluster.get_cluster_type", lambda: ttnn.cluster.ClusterType.T3K)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert is_t3k_cluster() is False


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


# --- the 31B 128k swept-decode gate ------------------------------------------
# Found by reading generated text on a real WH T3K: the demo reports PASSED in
# the degenerate case, so its verdict cannot catch this.


class _DenseCfg:
    """Dense 12B/31B: not MoE, no per-layer inputs."""

    enable_moe_block = False
    hidden_size_per_layer_input = 0


def test_swept_decode_disabled_flag_is_honoured(monkeypatch):
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    cfg = _DenseCfg()
    assert swept_decode_enabled(_FakeMesh(8), cfg) is True
    cfg.gemma4_swept_decode_disabled = True
    assert swept_decode_enabled(_FakeMesh(8), cfg) is False


def test_swept_decode_still_off_wherever_the_dense_gate_is(monkeypatch):
    """The new flag narrows the gate; it must never widen it."""
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert swept_decode_enabled(_FakeMesh(8), _DenseCfg()) is False
