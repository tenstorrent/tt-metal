# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the decode fusion defaults (tt/mesh_orientation.py): the fused decode step is the default
of a Blackhole MoE build on the 4-chip (1,4) / (4,1) meshes only, every env var still wins, and without a recorded
profile the modules keep the old paths. No device, no ttnn import.

    pytest models/demos/gemma4/tests/unit/test_decode_fusion_defaults_host.py --noconftest -p no:cacheprovider
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import models.common.utility_functions as uf
import ttnn
from models.demos.gemma4.tt import ccl as ccl_mod
from models.demos.gemma4.tt import mesh_orientation as mo
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_HARD_MAX,
    PREFILL_SDPA_MAX_SEQ,
    prefill_short_lived_memcfg,
)
from models.demos.gemma4.tt.ccl import ccl_async_enabled, default_ccl_topology
from models.demos.gemma4.tt.common import gemma4_cache_layout
from models.demos.gemma4.tt.dram_sharded import can_dram_shard
from models.demos.gemma4.tt.experts.weights import _refuse_placeholder_on_cache_miss
from models.demos.gemma4.tt.generator import VISION_KEY_PREFIXES, load_vision_state_dict
from models.demos.gemma4.tt.generator_trace import (
    get_gemma4_long_context_policy,
    normalize_gemma4_model_key,
    resolve_gemma4_demo_long_context,
)
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.precision import Gemma4Precision


@pytest.fixture()
def _clean(monkeypatch):
    for name in mo.DECODE_FUSION_ON:
        monkeypatch.delenv(name, raising=False)
    mo.clear_decode_fusion_profile()
    yield
    mo.clear_decode_fusion_profile()


@pytest.mark.usefixtures("_clean")
def test_four_chip_blackhole_moe_is_fused():
    for shape in ((1, 4), (4, 1)):
        assert mo.fused_decode_build(shape, blackhole=True, moe=True)
        assert mo.decode_fusion_defaults(shape, blackhole=True, moe=True) == mo.DECODE_FUSION_ON
    assert mo.DECODE_FUSION_ON["GEMMA4_NORM_FUSED"] == "2"
    assert set(mo.DECODE_FUSION_ON) == {
        "GEMMA4_MLP_DECODE_1D", "GEMMA4_ATTN_DECODE_1D", "GEMMA4_ROUTER_DECODE_1D", "GEMMA4_HEAD_NORM_4D",
        "GEMMA4_MOE_FUSED", "GEMMA4_MLP_GEGLU_FUSED", "GEMMA4_SOFTCAP_FUSED", "GEMMA4_CONCAT_HEADS_FUSED",
        "GEMMA4_QKV_L1_OUT", "GEMMA4_NORM_FUSED", "GEMMA4_LAYER_NORM_FUSED", "GEMMA4_QKV_NORM_ROPE_FUSED",
        "GEMMA4_SMALL_ROW_EXPERTS",
    }  # fmt: skip


@pytest.mark.usefixtures("_clean")
def test_other_builds_keep_the_old_defaults():
    assert not mo.fused_decode_build((1, 4), blackhole=False, moe=True)  # Wormhole
    assert not mo.fused_decode_build((1, 4), blackhole=True, moe=False)  # dense 31B / 12B
    for shape in ((1, 1), (1, 8), (2, 4), (4, 8), (8, 4), (1, 32)):  # one chip, the Galaxy lane engine
        assert not mo.fused_decode_build(shape, blackhole=True, moe=True)
    assert mo.decode_fusion_defaults((4, 8), blackhole=True, moe=True) == mo.DECODE_FUSION_OFF
    assert mo.decode_fusion_defaults((1, 4), blackhole=False, moe=True) == mo.DECODE_FUSION_OFF


@pytest.mark.usefixtures("_clean")
def test_galaxy_lane_build_fuses_the_qkv_norm_rope_by_default(monkeypatch):
    """The (8,4) Blackhole MoE build (the Galaxy lane engine) defaults the fused QKV norm + RoPE program on and
    nothing else (window 13: c1 32.4 -> 37.1 tok/s/user); the dense (8,4), Wormhole and the other lane-less meshes
    keep the old defaults; an env override still wins."""
    assert mo.lane_fused_decode_build((8, 4), blackhole=True, moe=True)
    lane = mo.decode_fusion_defaults((8, 4), blackhole=True, moe=True)
    assert lane == mo.LANE_DECODE_FUSION_ON and lane["GEMMA4_QKV_NORM_ROPE_FUSED"] == "1"
    assert {k for k, v in lane.items() if v != "0"} == {"GEMMA4_QKV_NORM_ROPE_FUSED"}, "only the QKV fusion"
    assert set(lane) == set(mo.DECODE_FUSION_OFF)
    assert not mo.fused_decode_build((8, 4), blackhole=True, moe=True), "not the 4-chip MoE profile"
    assert not mo.lane_fused_decode_build((8, 4), blackhole=True, moe=False)  # dense 31B / 12B on the Galaxy
    assert not mo.lane_fused_decode_build((8, 4), blackhole=False, moe=True)  # Wormhole
    assert mo.decode_fusion_defaults((8, 4), blackhole=True, moe=False) == mo.DECODE_FUSION_OFF
    assert mo.decode_fusion_defaults((8, 4), blackhole=False, moe=True) == mo.DECODE_FUSION_OFF
    for shape in ((4, 8), (2, 4), (1, 8), (1, 32), (1, 1)):
        assert not mo.lane_fused_decode_build(shape, blackhole=True, moe=True)
        assert mo.decode_fusion_defaults(shape, blackhole=True, moe=True) == mo.DECODE_FUSION_OFF
    assert mo.decode_fusion_defaults((1, 4), blackhole=True, moe=True) == mo.DECODE_FUSION_ON, "4-chip unchanged"
    assert mo.decode_fusion_defaults((1, 4), blackhole=True, moe=False) == mo.DENSE_DECODE_FUSION_ON, "dense unchanged"
    monkeypatch.delenv("GEMMA4_QKV_NORM_ROPE_FUSED", raising=False)
    assert mo.set_decode_fusion_profile((8, 4), blackhole=True, moe=True) == mo.LANE_DECODE_FUSION_ON
    assert mo.decode_flag("GEMMA4_QKV_NORM_ROPE_FUSED") == "1" and mo.decode_flag("GEMMA4_NORM_FUSED") == "0"
    monkeypatch.setenv("GEMMA4_QKV_NORM_ROPE_FUSED", "0")
    assert mo.decode_flag("GEMMA4_QKV_NORM_ROPE_FUSED") == "0", "the env override wins"
    mo.clear_decode_fusion_profile()


def test_profile_and_env_override(monkeypatch):
    # no profile: the old paths, unless the env var asks
    assert mo.decode_flag("GEMMA4_MOE_FUSED") == "0"
    monkeypatch.setenv("GEMMA4_MOE_FUSED", "1")
    assert mo.decode_flag("GEMMA4_MOE_FUSED") == "1"
    monkeypatch.delenv("GEMMA4_MOE_FUSED")
    # the 4-chip Blackhole MoE profile turns everything on ...
    assert mo.set_decode_fusion_profile((1, 4), blackhole=True, moe=True) == mo.DECODE_FUSION_ON
    for name, value in mo.DECODE_FUSION_ON.items():
        assert mo.decode_flag(name) == value
    # ... and a set env var still wins, in either direction
    monkeypatch.setenv("GEMMA4_QKV_NORM_ROPE_FUSED", "0")
    assert mo.decode_flag("GEMMA4_QKV_NORM_ROPE_FUSED") == "0"
    monkeypatch.setenv("GEMMA4_NORM_FUSED", "1")
    assert mo.decode_flag("GEMMA4_NORM_FUSED") == "1"
    monkeypatch.setenv("GEMMA4_MOE_FUSED", "")  # empty = unset
    assert mo.decode_flag("GEMMA4_MOE_FUSED") == "1"
    # a Wormhole profile turns everything off again, env still wins
    assert mo.set_decode_fusion_profile((1, 4), blackhole=False, moe=True) == mo.DECODE_FUSION_OFF
    assert mo.decode_flag("GEMMA4_MLP_DECODE_1D") == "0"
    monkeypatch.setenv("GEMMA4_MLP_DECODE_1D", "1")
    assert mo.decode_flag("GEMMA4_MLP_DECODE_1D") == "1"


@pytest.mark.usefixtures("_clean")
def test_four_chip_blackhole_dense_runs_the_combined_dense_configuration():
    for shape in ((1, 4), (4, 1)):
        assert mo.dense_fused_decode_build(shape, blackhole=True, moe=False)
        assert not mo.fused_decode_build(shape, blackhole=True, moe=False)
        d = mo.decode_fusion_defaults(shape, blackhole=True, moe=False)
        assert d == mo.DENSE_DECODE_FUSION_ON
        for name in mo.DECODE_FUSION_ON:
            if name not in mo.DENSE_DECODE_FUSION_ON:
                assert (
                    d.get(name, "0") == "0"
                )  # the MoE-only fusions and the fused norm family stay off on dense widths
    # the MoE profile keeps every MoE flag on and none of the dense-only ones
    d = mo.decode_fusion_defaults((1, 4), blackhole=True, moe=True)
    assert d == mo.DECODE_FUSION_ON
    for name in mo.DENSE_DECODE_FUSION_ON:
        assert d.get(name, "0") == mo.DECODE_FUSION_ON.get(name, "0")
    # Wormhole, one chip and the (1,8) line keep the old dense defaults
    assert not mo.dense_fused_decode_build((1, 4), blackhole=False, moe=False)
    for shape in ((1, 1), (1, 8), (4, 8)):
        assert not mo.dense_fused_decode_build(shape, blackhole=True, moe=False)
        assert mo.decode_fusion_defaults(shape, blackhole=True, moe=False) == mo.DECODE_FUSION_OFF


@pytest.mark.usefixtures("_clean")
def test_dense_profile_env_override(monkeypatch):
    monkeypatch.delenv("GEMMA4_QKV_HEADS_FUSED", raising=False)
    mo.set_decode_fusion_profile((1, 4), blackhole=True, moe=False)
    try:
        assert mo.decode_flag("GEMMA4_QKV_HEADS_FUSED") == "1"
        assert mo.decode_flag("GEMMA4_NORM_FUSED") == "0"
        monkeypatch.setenv("GEMMA4_QKV_HEADS_FUSED", "0")
        assert mo.decode_flag("GEMMA4_QKV_HEADS_FUSED") == "0"  # a set env var wins
    finally:
        mo.clear_decode_fusion_profile()


@pytest.mark.parametrize(
    "flag, blackhole, expect", [("1", True, True), ("1", False, False), ("0", True, False), ("", False, False)]
)
@pytest.mark.usefixtures("_clean")
def test_fused_qkv_norm_rope_prefill_runs_on_blackhole_only(monkeypatch, flag, blackhole, expect):
    """GEMMA4_QKV_NORM_ROPE_PREFILL=1 enables the fused prefill op on Blackhole only; on Wormhole the flag is refused
    with one warning and the composite path runs (CI at 66238f4353b hung a wh_llmbox_perf box on that row)."""
    from loguru import logger

    monkeypatch.setenv("GEMMA4_QKV_NORM_ROPE_PREFILL", flag)
    monkeypatch.setattr(mo, "_QKV_PREFILL_WH_WARNED", False)
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="WARNING")
    try:
        assert mo.qkv_norm_rope_prefill_enabled(blackhole) is expect
        assert mo.qkv_norm_rope_prefill_enabled(blackhole) is expect  # stable on a second call
    finally:
        logger.remove(sink)
    warned = [l for l in lines if "GEMMA4_QKV_NORM_ROPE_PREFILL=1 ignored" in l]
    assert len(warned) == (1 if flag == "1" and not blackhole else 0), lines  # once, not per call


@pytest.mark.usefixtures("_clean")
def test_fused_qkv_norm_rope_prefill_is_the_four_chip_blackhole_dense_default(monkeypatch):
    """No env: the 4-chip Blackhole dense profile turns the fused prefill op on; the device arch still decides (a
    Wormhole device refuses it even if a profile carried it); other builds and no profile keep the composite path;
    an env "0" wins over the profile."""
    monkeypatch.delenv("GEMMA4_QKV_NORM_ROPE_PREFILL", raising=False)
    monkeypatch.setattr(mo, "_QKV_PREFILL_WH_WARNED", False)
    assert mo.DENSE_DECODE_FUSION_ON["GEMMA4_QKV_NORM_ROPE_PREFILL"] == "1"
    assert mo.DECODE_FUSION_OFF["GEMMA4_QKV_NORM_ROPE_PREFILL"] == "0"
    mo.clear_decode_fusion_profile()
    assert mo.qkv_norm_rope_prefill_enabled(True) is False
    for shape in ((1, 4), (4, 1)):
        mo.set_decode_fusion_profile(shape, blackhole=True, moe=False)
        assert mo.qkv_norm_rope_prefill_enabled(True) is True
        assert mo.qkv_norm_rope_prefill_enabled(False) is False
    mo.set_decode_fusion_profile((1, 4), blackhole=True, moe=True)  # the MoE profile: composite prefill
    assert mo.qkv_norm_rope_prefill_enabled(True) is False
    for shape in ((1, 1), (1, 8), (4, 8)):
        mo.set_decode_fusion_profile(shape, blackhole=True, moe=False)
        assert mo.qkv_norm_rope_prefill_enabled(True) is False
    mo.set_decode_fusion_profile((1, 4), blackhole=False, moe=False)  # Wormhole build: no dense profile
    assert mo.qkv_norm_rope_prefill_enabled(False) is False
    mo.set_decode_fusion_profile((1, 4), blackhole=True, moe=False)
    monkeypatch.setenv("GEMMA4_QKV_NORM_ROPE_PREFILL", "0")
    assert mo.qkv_norm_rope_prefill_enabled(True) is False
    mo.clear_decode_fusion_profile()


# ===== merged from tests/unit/test_mesh_orientation_host.py: Host-only checks of the 4-chip column-mesh defaults (tt/mesh_orientation.py): the (1,4) -> (4,1) reshape
_ENVS = (mo.COLUMN_MESH_ENV, mo.TP_AXIS_ENV, mo.DISPATCH_ENV, mo.DISPATCH_MIN_TOKENS_ENV)


@pytest.fixture()
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("_clean_env")
def test_column_mesh_detection():
    assert mo.is_column_mesh((4, 1)) and mo.is_column_mesh((8, 1))
    for s in ((1, 4), (1, 1), (2, 4), (8, 4), (1, 8)):
        assert not mo.is_column_mesh(s)


@pytest.mark.usefixtures("_clean_env")
def test_tp_axis_default_and_override(monkeypatch):
    assert mo.tp_axis_for((4, 1)) == 0
    assert mo.tp_axis_for((1, 4)) == 1
    assert mo.tp_axis_for((2, 4)) == 1
    assert mo.tp_axis_for(None) == 1
    monkeypatch.setenv(mo.TP_AXIS_ENV, "0")
    assert mo.tp_axis_for((1, 4)) == 0
    monkeypatch.setenv(mo.TP_AXIS_ENV, "1")
    assert mo.tp_axis_for((4, 1)) == 1


@pytest.mark.usefixtures("_clean_env")
def test_reshape_decision(monkeypatch):
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True) == (4, 1)
    # already a column mesh, dense models, Wormhole, DP > 1, lane fold, other shapes: keep
    assert mo.should_reshape_to_column((4, 1), blackhole=True, moe=True) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=False) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=False, moe=True) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True, data_parallel=2) is None
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True, lanes=True) is None
    for s in ((1, 8), (2, 4), (8, 4), (1, 1)):
        assert mo.should_reshape_to_column(s, blackhole=True, moe=True) is None
    monkeypatch.setenv(mo.COLUMN_MESH_ENV, "0")
    assert mo.should_reshape_to_column((1, 4), blackhole=True, moe=True) is None


@pytest.mark.usefixtures("_clean_env")
def test_dispatch_defaults(monkeypatch):
    assert mo.dispatch_prefill_default((4, 1), blackhole=True)
    assert mo.dispatch_min_tokens_default((4, 1), blackhole=True) == 128
    assert not mo.dispatch_prefill_default((1, 4), blackhole=True)
    assert mo.dispatch_min_tokens_default((1, 4), blackhole=True) == 256
    assert not mo.dispatch_prefill_default((4, 1), blackhole=False)
    assert not mo.dispatch_prefill_default((8, 4), blackhole=True)
    monkeypatch.setenv(mo.DISPATCH_ENV, "0")
    assert not mo.dispatch_prefill_default((4, 1), blackhole=True)
    assert mo.dispatch_min_tokens_default((4, 1), blackhole=True) == 256
    monkeypatch.setenv(mo.DISPATCH_ENV, "1")
    assert mo.dispatch_prefill_default((1, 4), blackhole=True)
    monkeypatch.setenv(mo.DISPATCH_MIN_TOKENS_ENV, "512")
    assert mo.dispatch_min_tokens_default((1, 4), blackhole=True) == 512


# ===== merged from tests/unit/test_ccl_topology.py: Host-only tests for Gemma4 CCL topology / async / L1 env knobs.
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
    """WH T3K 1x8: keep Linear — Ring regresses 26B-A4B full-model PCC < 0.76."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Linear


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


def test_ccl_l1_small_answer_is_per_mesh_id_not_python_address(monkeypatch):
    """The L1_SMALL answer is cached by the mesh's own id (``MeshDevice.id()``, never reused), not ``id(mesh_device)``:
    the next pytest case's mesh, opened without the region at whatever address, is queried itself, never served
    the closed mesh's stale True."""
    with_region, without = 10**9 + 11, 10**9 + 12  # far above any real mesh counter in this process
    regions = {with_region: 65536, without: 0}
    monkeypatch.setattr(
        ttnn, "get_memory_view", lambda mesh, buffer_type: SimpleNamespace(total_bytes_per_bank=regions[mesh.id()])
    )
    try:
        assert ccl_mod.ccl_semaphores_in_l1_small(SimpleNamespace(id=lambda: with_region)) is True
        assert ccl_mod.ccl_semaphores_in_l1_small(SimpleNamespace(id=lambda: with_region)) is True  # another handle
        assert ccl_mod.ccl_semaphores_in_l1_small(SimpleNamespace(id=lambda: without)) is False
        assert {with_region: True, without: False}.items() <= ccl_mod._L1_SMALL_SEMAPHORES.items()
    finally:
        for k in (with_region, without):
            ccl_mod._L1_SMALL_SEMAPHORES.pop(k, None)


# ===== merged from tests/unit/test_precision_overrides.py: Host-only regression tests for Gemma4Precision.load key resolution.
SNAPSHOT_PATH = (
    "/mnt/MLPerf/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
)


def test_repo_id_resolves_overrides():
    p = Gemma4Precision.load("google/gemma-4-31B-it", (1, 4))
    assert p.get("shared_mlp") == ttnn.bfloat8_b
    assert p.get("attention") == ttnn.bfloat8_b


def test_hf_snapshot_path_resolves_same_overrides():
    direct = Gemma4Precision.load("google/gemma-4-31B-it", (1, 4))
    snapshot = Gemma4Precision.load(SNAPSHOT_PATH, (1, 4))
    assert snapshot._overrides == direct._overrides
    assert snapshot.get("shared_mlp") == ttnn.bfloat8_b


def test_hf_snapshot_path_trailing_slash():
    p = Gemma4Precision.load(SNAPSHOT_PATH + "/", (1, 4))
    assert p.get("shared_mlp") == ttnn.bfloat8_b


def test_unknown_model_still_empty():
    p = Gemma4Precision.load("/some/local/dir/my-finetune", (1, 4))
    assert p._overrides == {}
    # default fallback stays the caller-supplied dtype
    assert p.get("shared_mlp", ttnn.bfloat16) == ttnn.bfloat16


def test_the_bfp8_ceiling_applies_on_the_mesh_it_was_measured_on():
    """31B/tp=8 at 262k degenerates with shared_mlp in bfp8, so it downgrades."""
    o = Gemma4Precision.load("/x/gemma-4-31B-it", (1, 8), max_seq_len=262144)
    assert o.get("shared_mlp") == ttnn.bfloat16


def test_the_bfp8_ceiling_does_not_apply_on_a_mesh_where_bf16_will_not_fit():
    """The downgrade DOUBLES those weights, so it may only be declared where
    bf16 fits. Applying it to every mesh hung Gemma4-31B on bh_quietbox_2
    (1x4) in model init -- 31B shared_mlp in bf16 at 262144 does not fit on
    four chips, and the leg passed on main, which had no ceiling at all. A mesh
    with no entry keeps bfp8, i.e. main's behaviour."""
    o = Gemma4Precision.load("/x/gemma-4-31B-it", (1, 4), max_seq_len=262144)
    assert o.get("shared_mlp") == ttnn.bfloat8_b


def test_below_the_ceiling_bfp8_is_kept_on_the_measured_mesh():
    o = Gemma4Precision.load("/x/gemma-4-31B-it", (1, 8), max_seq_len=131072)
    assert o.get("shared_mlp") == ttnn.bfloat8_b


def test_column_mesh_takes_the_transposed_entry():
    """The 4-chip line as the (4,1) column mesh (dispatch prefill orientation) holds the same per-chip weights as
    (1,4): the 26B experts stay bfp8 in both orientations (a bf16 miss doubles both expert copies)."""
    row = Gemma4Precision.load("google/gemma-4-26B-A4B-it", (1, 4))
    col = Gemma4Precision.load("google/gemma-4-26B-A4B-it", (4, 1))
    assert row.get("experts", ttnn.bfloat16) == ttnn.bfloat8_b
    assert col.get("experts", ttnn.bfloat16) == ttnn.bfloat8_b


# ===== merged from tests/unit/test_cache_layout.py: The warm-cache identity pins every knob that names a tensorbin file. Pure Python.
def _galaxy_config(ep):
    return SimpleNamespace(tp=8, ep=ep, tp_axis=0, weight_fracture=True)


def test_expert_partition_keys_the_cache_layout():
    galaxy = SimpleNamespace(shape=(8, 4))
    ep4 = gemma4_cache_layout(galaxy, _galaxy_config(ep=4))
    ep1 = gemma4_cache_layout(galaxy, _galaxy_config(ep=1))  # GEMMA4_MOE_EP=0: experts whole per chip
    assert ep4["expert_partition"] == "ep4tp8_tp_axis0"
    assert ep1["expert_partition"] == "ep1tp8_tp_axis0"
    assert ep4 != ep1
    assert ep4["weight_layout"] == "fractured-tp_axis0"


def test_single_device_layout_has_the_trivial_partition():
    layout = gemma4_cache_layout(SimpleNamespace(), None)
    assert layout["expert_partition"] == "ep1tp1_tp_axis1"
    assert layout["weight_layout"] == "tp"


# ===== merged from tests/unit/test_served_weights_dir.py: The loaders read the served checkpoint (MODEL_WEIGHTS_DIR / HF_MODEL) when the path vLLM
@pytest.fixture
def served_checkpoint(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "gemma-4-test"
    weights.mkdir(parents=True)
    vision_key = f"{VISION_KEY_PREFIXES[0]}probe.weight"
    save_file(
        {"model.language_model.probe.weight": torch.ones(2, 2), vision_key: torch.full((2,), 3.0)},
        str(weights / "model-00001-of-00001.safetensors"),
    )
    empty = tmp_path / "hub_snapshot"  # what vLLM leaves behind: no weight files
    empty.mkdir()
    for env in ("MODEL_WEIGHTS_DIR", "HF_MODEL", "GEMMA4_MODEL_PATH"):
        monkeypatch.delenv(env, raising=False)
    return weights, empty, vision_key


def test_served_weights_dir_wants_a_directory_with_safetensors(served_checkpoint, monkeypatch):
    weights, empty, _ = served_checkpoint
    assert Gemma4ModelArgs.served_weights_dir() is None
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-test")  # a repo id, not a directory
    assert Gemma4ModelArgs.served_weights_dir() is None
    monkeypatch.setenv("HF_MODEL", str(empty))
    assert Gemma4ModelArgs.served_weights_dir() is None
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(weights))
    assert Gemma4ModelArgs.served_weights_dir() == str(weights)


def test_loaders_fall_back_to_the_served_checkpoint(served_checkpoint, monkeypatch):
    weights, empty, vision_key = served_checkpoint
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(weights))
    state_dict = Gemma4ModelArgs.load_state_dict(str(empty))
    assert torch.equal(state_dict["model.language_model.probe.weight"], torch.ones(2, 2))
    vision = load_vision_state_dict(str(empty))
    assert list(vision) == [vision_key]


def test_vision_loader_still_fails_loudly_without_a_checkpoint(served_checkpoint, expect_error):
    _, empty, _ = served_checkpoint
    with expect_error(FileNotFoundError, "no vision tensors"):
        load_vision_state_dict(str(empty))


# ===== merged from tests/unit/test_expert_cache_guard.py: Host-only: the expert loader refuses a warm-cache placeholder state dict when its tensorbins are missing.
class _Placeholder(dict):
    is_placeholder = True


def _tensorbin(prefix):
    return f"{prefix}_dtype_{ttnn.bfloat8_b.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"


def test_real_state_dict_never_refused(tmp_path):
    _refuse_placeholder_on_cache_miss({}, [str(tmp_path / "gate_up_proj_x")], ttnn.bfloat8_b)


def test_placeholder_refused_when_tensorbin_missing(tmp_path, expect_error):
    with expect_error(RuntimeError, "cache miss"):
        _refuse_placeholder_on_cache_miss(_Placeholder(), [str(tmp_path / "gate_up_proj_x")], ttnn.bfloat8_b)


def test_placeholder_refused_without_cache_path(expect_error):
    with expect_error(RuntimeError, "cache miss"):
        _refuse_placeholder_on_cache_miss(_Placeholder(), [None], ttnn.bfloat8_b)


def test_placeholder_accepted_when_all_tensorbins_present(tmp_path):
    prefixes = [str(tmp_path / "gate_up_proj_x"), str(tmp_path / "down_proj_x")]
    for p in prefixes:
        open(_tensorbin(p), "wb").close()
    _refuse_placeholder_on_cache_miss(_Placeholder(), prefixes, ttnn.bfloat8_b)


def test_placeholder_refused_when_one_tensorbin_missing(tmp_path, expect_error):
    prefixes = [str(tmp_path / "gate_up_proj_x"), str(tmp_path / "down_proj_x")]
    open(_tensorbin(prefixes[0]), "wb").close()
    with expect_error(RuntimeError, "down_proj_x"):
        _refuse_placeholder_on_cache_miss(_Placeholder(), prefixes, ttnn.bfloat8_b)


# ===== merged from tests/unit/test_long_context_policy.py: Pin GEMMA4_LONG_CONTEXT_POLICY cutovers used by text_demo / text_demo_v2 / vLLM.
_EXPECTED = [
    # ── 31B / LoudBox ─────────────────────────────────────────────────────
    ("P150x8", "google/gemma-4-31B-it", 4096, False, 4096),
    ("P150x8", "google/gemma-4-31B-it", 32768, False, 4096),
    ("P150x8", "google/gemma-4-31B-it", 65536, False, 4096),
    ("P150x8", "google/gemma-4-31B-it", 131072, True, 2048),
    ("P150x8", "google/gemma-4-31B-it", 262144, True, 2048),
    # ── 31B / QB2 ─────────────────────────────────────────────────────────
    ("P150x4", "google/gemma-4-31B-it", 32768, False, 4096),
    ("P150x4", "google/gemma-4-31B-it", 65536, True, 4096),
    ("P150x4", "google/gemma-4-31B-it", 131072, True, 2048),
    ("P300x2", "google/gemma-4-31B-it", 131072, True, 2048),
    # ── 31B / T3K (WH 12 GB/ASIC: multi-chunk 2048; bound from 32k) ───────
    ("T3K", "google/gemma-4-31B-it", 8192, False, 2048),
    ("T3K", "google/gemma-4-31B-it", 16384, False, 2048),
    # Unbounded ceiling re-measured at 32768; bounded now starts at 65536 so the
    # multi-request sliding remap stays out of the served range.
    ("T3K", "google/gemma-4-31B-it", 131072, True, 2048),
    # ── 12B: stay unbounded through 128k on multi-chip BH for perf ────────
    ("P150x8", "google/gemma-4-12B-it", 131072, False, 4096),
    ("P150x8", "google/gemma-4-12B-it", 262144, False, 4096),
    ("P150x4", "google/gemma-4-12B-it", 131072, False, 4096),
    ("P150x4", "google/gemma-4-12B-it", 262144, True, 4096),
    ("P150", "google/gemma-4-12B-it", 32768, False, 4096),
    ("P150", "google/gemma-4-12B-it", 65536, True, 4096),
    ("P150", "google/gemma-4-12B-it", 131072, True, 4096),
    ("P150", "google/gemma-4-12B-it", 262144, True, 4096),  # full HF ISL
    # ── 26B-A4B / LoudBox: same 128k coherency cutover as 31B ─────────────
    ("P150x8", "google/gemma-4-26B-A4B-it", 65536, False, 4096),
    ("P150x8", "google/gemma-4-26B-A4B-it", 131072, True, 2048),
    ("P150x4", "google/gemma-4-26B-A4B-it", 131072, False, 4096),
    # ── E2B/E4B: unbounded through measured 256k on P150 / LB / QB2 ───────
    ("P150", "google/gemma-4-E2B-it", 262144, False, 4096),
    ("P150", "google/gemma-4-E4B-it", 262144, False, 4096),
    ("P150x8", "google/gemma-4-E4B-it", 262144, False, 4096),
    ("P150x4", "google/gemma-4-E2B-it", 262144, False, 4096),
    # ── Wormhole, measured on a real T3K / N300 (12 GB per ASIC) ──────────
    # Bounded sliding is deliberately kept OUT of the WH serving range: its
    # sliding page-table remap is keyed on the current tensor's row index rather
    # than the request's persistent KV slot, which corrupts any multi-request
    # batch (see generator_trace 12B/T3K comment). These cases pin that
    # bounded_sliding stays False across the whole served context.
    # 12B/T3K: unbounded measured through 131072; bounded only at the full 256k.
    ("T3K", "google/gemma-4-12B-it", 8192, False, 2048),
    ("T3K", "google/gemma-4-12B-it", 16384, False, 2048),
    ("T3K", "google/gemma-4-12B-it", 131072, False, 2048),
    ("T3K", "google/gemma-4-12B-it", 262144, True, 2048),
    # 12B/N300 (24 GB, the only variant that fits one WH card): unbounded to 8k,
    # bounded above — so concurrent serving stays at/below 8192 there.
    ("N300", "google/gemma-4-12B-it", 8192, False, 2048),
    ("N300", "google/gemma-4-12B-it", 16384, True, 2048),
    ("N300", "google/gemma-4-12B-it", 131072, True, 2048),
    # 26B-A4B/T3K and 31B/T3K: unbounded measured through 32768.
    ("T3K", "google/gemma-4-26B-A4B-it", 8192, False, 2048),
    ("T3K", "google/gemma-4-26B-A4B-it", 32768, False, 2048),
    ("T3K", "google/gemma-4-26B-A4B-it", 65536, True, 2048),
    ("T3K", "google/gemma-4-31B-it", 32768, False, 2048),
    ("T3K", "google/gemma-4-31B-it", 65536, True, 2048),
]


@pytest.mark.parametrize(
    "mesh,model,max_seq_len,exp_bounded,exp_chunk",
    _EXPECTED,
    ids=[f"{x[0]}-{x[1].rsplit('/', 1)[-1]}-{x[2]}" for x in _EXPECTED],
)
def test_long_context_policy_defaults(mesh, model, max_seq_len, exp_bounded, exp_chunk, monkeypatch):
    monkeypatch.delenv("GEMMA4_BOUNDED_SLIDING", raising=False)
    monkeypatch.delenv("GEMMA4_GEN_PREFILL_CHUNK", raising=False)
    monkeypatch.delenv("GEMMA4_DEMO_SINGLE_CHUNK", raising=False)
    monkeypatch.setenv("MESH_DEVICE", mesh)

    lc = resolve_gemma4_demo_long_context(max_seq_len, mesh_device=None, model_name_or_path=model)
    assert (
        lc["bounded_sliding"] is exp_bounded
    ), f"{mesh}/{model}@{max_seq_len}: bounded={lc['bounded_sliding']} want {exp_bounded}"
    assert (
        lc["prefill_chunk"] == exp_chunk
    ), f"{mesh}/{model}@{max_seq_len}: chunk={lc['prefill_chunk']} want {exp_chunk}"


def test_env_override_bounded_and_chunk(monkeypatch):
    monkeypatch.setenv("MESH_DEVICE", "P150x8")
    monkeypatch.setenv("GEMMA4_BOUNDED_SLIDING", "0")
    monkeypatch.setenv("GEMMA4_GEN_PREFILL_CHUNK", "4096")
    lc = resolve_gemma4_demo_long_context(131072, None, "google/gemma-4-31B-it")
    assert lc["bounded_sliding"] is False
    assert lc["prefill_chunk"] == 4096


def test_mesh_device_alias_p300x2(monkeypatch):
    monkeypatch.delenv("GEMMA4_BOUNDED_SLIDING", raising=False)
    monkeypatch.delenv("GEMMA4_GEN_PREFILL_CHUNK", raising=False)
    monkeypatch.setenv("MESH_DEVICE", "P300x2")
    lc = resolve_gemma4_demo_long_context(131072, None, "google/gemma-4-31B-it")
    assert lc["bounded_sliding"] is True
    assert lc["prefill_chunk"] == 2048


@pytest.mark.parametrize(
    "path,key",
    [
        ("google/gemma-4-31B-it", "31B"),
        (
            "/mnt/MLPerf/huggingface/hub/models--google--gemma-4-31B-it/snapshots/ba74f5b6c647c0911554e50278d6f6f4477f9010",
            "31B",
        ),
        (
            "/mnt/MLPerf/huggingface/hub/models--google--gemma-4-12B-it/snapshots/deadbeef",
            "12B",
        ),
    ],
)
def test_normalize_gemma4_model_key_snapshot_paths(path, key):
    assert normalize_gemma4_model_key(path) == key


@pytest.mark.parametrize("mesh", ["N300", "T3K", "N150", "N150x4"])
@pytest.mark.parametrize(
    "model",
    [
        "google/gemma-4-12B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
    ],
)
def test_wormhole_never_inherits_blackhole_policy(mesh, model, monkeypatch):
    """A WH board must never resolve to a Blackhole (QB2 / P150x*) entry.

    Regression for the silent cross-arch fallback: every (model, device) combo
    missing from the table used to fall back to the QB2 entry, handing a 24 GB
    N300 / 96 GB T3K Blackhole's 128 GB headroom ("unbounded KV through 128k")
    and OOMing on KV allocation. Wormhole must bound early and keep chunk 2048.

    Force the WH host view: ``_canonical_device_name`` only treats ``N150`` as a
    real WH board when the host is Wormhole (on Blackhole, ``N150`` is a
    historical P150 alias). Without this patch the N150 cases pass only on WH
    runners and falsely fail on BH.
    """
    monkeypatch.setattr(
        "models.demos.gemma4.tt.generator_trace._host_is_wormhole",
        lambda: True,
    )
    monkeypatch.delenv("GEMMA4_BOUNDED_SLIDING", raising=False)
    monkeypatch.delenv("GEMMA4_GEN_PREFILL_CHUNK", raising=False)
    monkeypatch.delenv("GEMMA4_DEMO_SINGLE_CHUNK", raising=False)
    monkeypatch.setenv("MESH_DEVICE", mesh)

    policy = get_gemma4_long_context_policy(None, model)
    assert "device_fallback" not in policy["source"], (
        f"{mesh}/{model} fell back to the Blackhole QB2 entry (source={policy['source']}); "
        "WH has 12 GB per ASIC and cannot inherit Blackhole DRAM headroom."
    )
    # Blackhole gemma4 entries all use chunk 4096; every WH entry must stay at
    # the 2048 chunk the WH boards were measured with. (unbounded_isl_max is no
    # longer a useful discriminator: WH T3K is now measured unbounded through
    # 131072 for 12B, which legitimately exceeds some Blackhole entries.)
    assert policy["prefill_chunk"] == 2048, f"{mesh}/{model}: {policy}"


def test_blackhole_qb2_fallback_still_applies(monkeypatch):
    """The QB2 fallback must stay intact for Blackhole boards (no regression)."""
    monkeypatch.setenv("MESH_DEVICE", "P150")
    policy = get_gemma4_long_context_policy(None, "google/gemma-4-31B-it")
    assert policy["source"].endswith("_device_fallback")
    assert policy["prefill_chunk"] == 4096


# ===== merged from tests/unit/test_wormhole_batched_prefill_guard.py: Wormhole keeps gemma4 on the sequential per-user prefill path.
def test_wormhole_disables_batched_prefill(monkeypatch):
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    assert Gemma4ModelArgs().disable_batched_prefill is True


def test_blackhole_keeps_batched_prefill(monkeypatch):
    monkeypatch.setattr(uf, "is_blackhole", lambda: True)
    assert Gemma4ModelArgs().disable_batched_prefill is False


def test_explicit_value_is_never_overridden(monkeypatch):
    """A caller opting in stays opted in -- test_batched_prefill_perf relies on this."""
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    args = Gemma4ModelArgs()
    args.disable_batched_prefill = False  # post-construction opt back in
    assert args.disable_batched_prefill is False

    monkeypatch.setattr(uf, "is_blackhole", lambda: True)
    assert Gemma4ModelArgs(disable_batched_prefill=True).disable_batched_prefill is True


def test_guard_is_silent_when_arch_cannot_be_determined(monkeypatch):
    """Host-only environments must not crash; they keep the shared default."""

    def boom():
        raise RuntimeError("no device")

    monkeypatch.setattr(uf, "is_blackhole", boom)
    assert Gemma4ModelArgs().disable_batched_prefill is False


def test_hatch_env_bypasses_guard_on_wormhole(monkeypatch):
    """G4_FORCE_BATCH_PREFILL=1 must reach the generator gates on Wormhole too:
    the guard steps aside so the hatch means the same thing on both arches."""
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    monkeypatch.setenv("G4_FORCE_BATCH_PREFILL", "1")
    assert Gemma4ModelArgs().disable_batched_prefill is False


def test_hatch_env_off_keeps_guard(monkeypatch):
    monkeypatch.setattr(uf, "is_blackhole", lambda: False)
    monkeypatch.setenv("G4_FORCE_BATCH_PREFILL", "0")
    assert Gemma4ModelArgs().disable_batched_prefill is True
