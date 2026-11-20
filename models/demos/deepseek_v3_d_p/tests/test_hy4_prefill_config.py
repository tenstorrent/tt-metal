# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the Hy4-preview prefill adapter plumbing: config constants and config.json parsing, the
indexer layer map, the mesh-layout switch, the pending-delta guard, the manifest, and the checkpoint reader
(fused experts, MXFP8, attention key remap). No device. Tests against the real checkpoint skip when it is absent.
"""

import json
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.hy4_config import (
    Hy4Config,
    hy4_hf_config,
    load_hy4_config_json,
    rope_theta_from_config,
    validate_hy4_config_dict,
)
from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout

HY4_CHECKPOINT = Path("/mnt/tt-data/Hy4-Preview")
MANIFEST = Path(__file__).resolve().parents[1] / "tt" / "runners" / "manifests" / "hy4_preview.json"
FULL_LAYERS = [0, 1] + list(range(5, 78, 4))


def _config_dict() -> dict:
    """A config.json equivalent to the Hy4-preview release, built from the constants."""
    c = Hy4Config
    return {
        "model_type": "hy_v4",
        "hidden_size": c.EMB_SIZE,
        "num_hidden_layers": c.NUM_LAYERS,
        "intermediate_size": c.INTERMEDIATE_SIZE,
        "moe_intermediate_size": c.MOE_INTERMEDIATE_SIZE,
        "vocab_size": c.VOCAB_SIZE,
        "max_position_embeddings": c.MAX_POSITION_EMBEDDINGS,
        "num_attention_heads": c.NUM_ATTENTION_HEADS,
        "num_key_value_heads": 8,
        "q_lora_rank": c.Q_LORA_RANK,
        "kv_lora_rank": c.KV_LORA_RANK,
        "qk_nope_head_dim": c.QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": c.QK_ROPE_HEAD_DIM,
        "v_head_dim": c.V_HEAD_DIM,
        "n_routed_experts": c.NUM_ROUTED_EXPERTS,
        "num_experts_per_tok": c.NUM_EXPERTS_PER_TOKEN,
        "n_shared_experts": c.NUM_SHARED_EXPERTS,
        "n_group": 1,
        "topk_group": 1,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "rms_norm_eps": 1e-5,
        "routed_scaling_factor": 2.827,
        "swiglu_limit": 10.0,
        "gated_mla": True,
        "gating_type": "elementwise",
        "learnable_sink": True,
        "enable_ihc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_magnitude": 2.0,
        "num_nextn_predict_layers": 1,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "indexer_types": ["full" if i in FULL_LAYERS else "shared" for i in range(78)],
        "mlp_layer_types": ["dense"] + ["sparse"] * 77,
        "use_mla": True,
        "use_dsa": True,
    }


# ---------------------------------------------------------------------------------------------------------------
# Config constants: GLM-5.3 + deltas
# ---------------------------------------------------------------------------------------------------------------


def test_hy4_shares_glm53_attention_and_moe_dims():
    for name in (
        "EMB_SIZE",
        "NUM_LAYERS",
        "NUM_ATTENTION_HEADS",
        "Q_LORA_RANK",
        "KV_LORA_RANK",
        "QK_NOPE_HEAD_DIM",
        "QK_ROPE_HEAD_DIM",
        "V_HEAD_DIM",
        "INDEX_TOPK",
        "INDEX_HEAD_DIM",
        "INDEX_N_HEADS",
        "NUM_ROUTED_EXPERTS",
        "NUM_EXPERTS_PER_TOKEN",
        "MOE_INTERMEDIATE_SIZE",
        "NUM_SHARED_EXPERTS",
        "NUM_EXPERT_GROUPS",
        "RMS_NORM_EPS",
        "FABRIC_PAYLOAD_SIZE",
    ):
        assert getattr(Hy4Config, name) == getattr(GLM53Config, name), name


@pytest.mark.parametrize(
    "name, glm, hy4",
    [
        ("NUM_DENSE_LAYERS", 3, 1),
        ("INTERMEDIATE_SIZE", 12288, 18432),
        ("VOCAB_SIZE", 154880, 120832),
        ("ROUTE_SCALE", 2.5, 2.827),
        ("ROPE_THETA", 8000000, 10000000),
    ],
)
def test_hy4_deltas_from_glm53(name, glm, hy4):
    assert getattr(GLM53Config, name) == glm
    assert getattr(Hy4Config, name) == hy4


def test_clamped_swiglu_plumbing():
    # The routed kernel bakes limit 10 (ClampedSiluGluConfigDsV4), so SWIGLU_LIMIT must stay 10.
    assert Hy4Config.SWIGLU_LIMIT == 10.0
    assert Hy4Config.ROUTED_EXPERT_ACTIVATION == "clamped_silu_glu"
    # HF hy_v4: the shared expert and dense layer 0 are HYV4MLP, which does not clamp.
    assert Hy4Config.SHARED_EXPERT_ACTIVATION == "silu"
    assert Hy4Config.DENSE_FFN_ACTIVATION == "silu"
    # moe_fused_swiglu cannot run ClampedSiluGlu, so the hybrid split must be off.
    assert Hy4Config.ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD is None
    assert not hasattr(GLM53Config, "SWIGLU_LIMIT")


def _glu_by_name(name: str, limit: float):
    """What the TT FFN paths compute for an activation name (TtSharedExpert / TtFfn / the routed kernel)."""
    if name == "silu":
        return lambda g, u: torch.nn.functional.silu(g) * u
    if name == "clamped_silu_glu":
        return lambda g, u: torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
    raise AssertionError(f"unexpected activation {name!r}")


def test_swiglu_clamp_matches_hf_per_ffn_kind():
    """Pins Hy4's clamp to the routed experts, against the HF hy_v4 formulas (reference/hy4/model.py): HYV4MoE
    clamps gate above and up both sides at swiglu_limit; HYV4MLP (shared expert, dense layer 0) does not."""
    lim = Hy4Config.SWIGLU_LIMIT
    g = torch.tensor([-30.0, -5.0, 0.5, 9.0, 12.0, 40.0])
    u = torch.tensor([-25.0, 11.0, -3.0, 2.0, 15.0, -40.0])
    hf_routed = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(min=-lim, max=lim)
    hf_mlp = torch.nn.functional.silu(g) * u
    assert not torch.allclose(hf_routed, hf_mlp)  # the inputs exercise the limit
    torch.testing.assert_close(_glu_by_name(Hy4Config.ROUTED_EXPERT_ACTIVATION, lim)(g, u), hf_routed)
    torch.testing.assert_close(_glu_by_name(Hy4Config.SHARED_EXPERT_ACTIVATION, lim)(g, u), hf_mlp)
    torch.testing.assert_close(_glu_by_name(Hy4Config.DENSE_FFN_ACTIVATION, lim)(g, u), hf_mlp)


# ---------------------------------------------------------------------------------------------------------------
# Indexer layer map
# ---------------------------------------------------------------------------------------------------------------


def test_indexer_map_is_21_full_57_shared():
    types = Hy4Config.indexer_types()
    assert len(types) == 78
    assert Hy4Config.full_indexer_layers() == FULL_LAYERS
    assert types.count("full") == 21 and types.count("shared") == 57
    assert types[0] == "full"  # a shared layer needs a full layer before it to reuse


def test_indexer_freq_offset_agree_with_explicit_map():
    freq, off = Hy4Config.INDEX_TOPK_FREQ, Hy4Config.INDEX_SKIP_TOPK_OFFSET
    derived = ["full" if (max(i - off + 1, 0) % freq) == 0 else "shared" for i in range(78)]
    assert derived == Hy4Config.indexer_types()


def test_indexer_map_differs_from_glm53():
    # Same count (21), different positions: GLM-5.3 is full at 0, 1, 2, 6, ..., 74; Hy4 at 0, 1, 5, ..., 77.
    glm = GLM53Config.indexer_types()
    assert glm != Hy4Config.indexer_types()
    assert glm.count("full") == 21
    assert glm[2] == "full" and Hy4Config.indexer_types()[2] == "shared"


@pytest.mark.parametrize(
    "first, count, want_first_rank, want_index_layers",
    [(0, 78, 0, 21), (0, 6, 0, 3), (0, 1, 0, 1), (39, 39, 11, 10), (2, 3, 2, 0)],
)
def test_index_cache_sizing_per_stage(first, count, want_first_rank, want_index_layers):
    """allocate_kv_cache sizes the index cache to full_indexer_rank(first + count) - full_indexer_rank(first)."""
    pytest.importorskip("ttnn")
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, indexer_layer_is_reused

    cfg = hy4_hf_config()
    assert full_indexer_rank(cfg, first) == want_first_rank
    assert full_indexer_rank(cfg, first + count) - full_indexer_rank(cfg, first) == want_index_layers
    assert [not indexer_layer_is_reused(cfg, i) for i in range(78)] == [i in FULL_LAYERS for i in range(78)]


def test_kv_only_last_layer_needs_its_indexer_cache_when_full(tmp_path):
    """Hy4's last layer (77) is a full indexer layer; a kv-only build still writes its index keys."""
    pytest.importorskip("ttnn")
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc

    hy4 = hy4_hf_config()
    assert hy4.indexer_types[77] == "full" and glm_5_3_hf_config().indexer_types[77] == "shared"
    assert not wc._kv_only_indexer_cache_complete(tmp_path, "layer_77", 77, hy4)
    assert wc._kv_only_indexer_cache_complete(tmp_path, "layer_76", 76, hy4)
    assert wc._kv_only_indexer_cache_complete(tmp_path, "layer_77", 77, glm_5_3_hf_config())
    assert wc._kv_only_indexer_cache_complete(tmp_path, "layer_77", 77, None)


def _kv_only_cache(root, layer, with_indexer):
    """The files a runner writes for a kv-only last layer: attn_norm, the KV projections (+ the indexer), and the
    layer's iHC parameters (its KV is computed from the collapsed iHC streams)."""
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import TtIndexer

    names = [
        f"layer_{layer}.ihc.pt",
        f"layer_{layer}.attn_norm_weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin",
        f"layer_{layer}.mla.kv_a_layernorm_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin",
        f"layer_{layer}.mla.kv_a_proj_with_mqa_dtype_BFLOAT8_B_layout_TILE.tensorbin",
    ]
    if with_indexer:
        for name in TtIndexer.WEIGHT_NAMES:
            short = TtIndexer._cache_short_name(name)
            names.append(f"layer_{layer}.mla.indexer_{short}_dtype_{TtIndexer.WEIGHT_DTYPES[name].name}_x.tensorbin")
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / n).touch()


@pytest.mark.parametrize("with_indexer", [False, True], ids=["no-indexer", "indexer"])
def test_transformer_cache_check_is_kv_only_aware_on_the_last_layer(tmp_path, with_indexer):
    """The non-rebuild ``TtPrefillTransformer.check_cache_complete`` (used by the chunked test driver and the
    runtime when it cannot rebuild): a kv-only last layer needs only attn_norm + the KV projections, plus its
    indexer on a ``full`` indexer layer (Hy4's 77); GLM-5.3's ``shared`` 77 does not. Without
    ``kv_only_last_layer`` the check is unchanged: a kv-only cache is not complete."""
    pytest.importorskip("ttnn")
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer

    _kv_only_cache(tmp_path, 77, with_indexer)

    def check(**kw):
        return TtPrefillTransformer.check_cache_complete(
            tmp_path, 1, first_k_dense=1, first_layer_idx=77, is_first_rank=False, model_cfg=Hy4Config, **kw
        )

    assert check(kv_only_last_layer=True, hf_config=hy4_hf_config()) == with_indexer
    assert check(kv_only_last_layer=True, hf_config=glm_5_3_hf_config())  # shared layer: no indexer needed
    assert not check(hf_config=hy4_hf_config())  # full-layer check (FFN/MoE, Q, ...) as before
    assert not check(kv_only_last_layer=True, is_last_rank=False, hf_config=hy4_hf_config())
    (tmp_path / "layer_77.ihc.pt").unlink()  # Hy4 (ENABLE_IHC) also needs the layer's iHC parameters
    assert not check(kv_only_last_layer=True, hf_config=hy4_hf_config())


# ---------------------------------------------------------------------------------------------------------------
# HF-attribute config + config.json validation
# ---------------------------------------------------------------------------------------------------------------


def test_hf_config_from_constants():
    cfg = hy4_hf_config(max_seq=4096)
    assert cfg.model_type == "hy_v4"
    assert cfg.first_k_dense_replace == 1
    assert cfg.intermediate_size == 18432
    assert cfg.vocab_size == 120832
    assert cfg.rope_theta == 1e7
    assert cfg.rope_scaling["factor"] == 1.0  # no YaRN
    assert cfg.max_seq_len == 4096 and cfg.rope_scaling["original_max_position_embeddings"] == 4096
    assert cfg.routed_scaling_factor == 2.827 and cfg.swiglu_limit == 10.0
    assert cfg.mla_use_output_gate is True and cfg.mla_learnable_sink is True
    assert cfg.enable_ihc is True and cfg.hc_mult == 4
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.indexer_types == Hy4Config.indexer_types()
    assert cfg.mlp_layer_types == ["dense"] + ["sparse"] * 77


def test_hf_config_has_every_field_glm53_has():
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config

    missing = set(vars(glm_5_3_hf_config())) - set(vars(hy4_hf_config()))
    assert not missing, missing


def test_config_dict_validates_and_builds():
    d = _config_dict()
    assert validate_hy4_config_dict(d) == []
    cfg = hy4_hf_config(max_seq=8192, config_dict=d)
    assert cfg.indexer_types == d["indexer_types"]
    assert cfg.rope_theta == 1e7


@pytest.mark.parametrize(
    "key, value, needle",
    [
        ("num_hidden_layers", 61, "num_hidden_layers"),
        ("vocab_size", 154880, "vocab_size"),
        ("routed_scaling_factor", 2.5, "routed_scaling_factor"),
        ("swiglu_limit", 7.0, "swiglu_limit"),
        ("model_type", "glm_moe_dsa", "model_type"),
        ("gating_type", "headwise", "gating_type"),
        ("mlp_layer_types", ["dense"] * 3 + ["sparse"] * 75, "mlp_layer_types"),
    ],
)
def test_config_dict_mismatch_is_reported(key, value, needle, expect_error):
    d = _config_dict()
    d[key] = value
    problems = validate_hy4_config_dict(d)
    assert any(needle in p for p in problems), problems
    with expect_error(ValueError, "does not match Hy4Config"):
        hy4_hf_config(config_dict=d)


def test_indexer_map_mismatch_is_reported():
    d = _config_dict()
    d["indexer_types"] = GLM53Config.indexer_types()
    assert any("indexer_types" in p for p in validate_hy4_config_dict(d))


def test_rope_theta_parsing(expect_error):
    assert rope_theta_from_config({"rope_parameters": {"rope_theta": 1e7, "rope_type": "default"}}) == 1e7
    assert rope_theta_from_config({"rope_theta": 5e5}) == 5e5
    with expect_error(ValueError, "rope_type"):
        rope_theta_from_config({"rope_parameters": {"rope_theta": 1e7, "rope_type": "yarn"}})
    with expect_error(ValueError, "no rope_theta"):
        rope_theta_from_config({})


@pytest.mark.skipif(not (HY4_CHECKPOINT / "config.json").is_file(), reason="Hy4-preview checkpoint not on this host")
def test_real_config_json_matches_constants():
    d = load_hy4_config_json(HY4_CHECKPOINT)
    assert validate_hy4_config_dict(d) == []
    assert [i for i, t in enumerate(d["indexer_types"]) if t == "full"] == FULL_LAYERS


# ---------------------------------------------------------------------------------------------------------------
# Mesh layout: TP32 / DP1 is the Hy4 layout (#65884); glm_prefill_sp8_tp4 is dev / numerics only
# ---------------------------------------------------------------------------------------------------------------


def test_layouts_and_heads_per_chip():
    tp32 = hy4_layout.HY4_PREFILL_LAYOUTS["tp32"]
    dev = hy4_layout.HY4_PREFILL_LAYOUTS["glm_prefill_sp8_tp4"]
    assert set(hy4_layout.HY4_PREFILL_LAYOUTS) == {"tp32", "glm_prefill_sp8_tp4"}
    assert hy4_layout.DEFAULT_LAYOUT == tp32.name and tp32.role == "production"
    assert dev.role == "dev" and "DEV / NUMERICS" in dev.note and dev.runnable
    assert tp32.sp * tp32.tp == dev.sp * dev.tp == 32
    # Same 8x4 mesh, same axes: sequence (SP 8) on axis 0, hidden (TP 4) on axis 1. They differ in where the heads go.
    assert (tp32.sp, tp32.tp, tp32.sp_axis) == (dev.sp, dev.tp, dev.sp_axis) == (8, 4, 0)
    assert tp32.hidden_axis == dev.hidden_axis == 1
    assert tp32.heads_per_chip(64) == 8  # heads over the 8-chip axis 0 (#65884's 8 columns), with the sequence
    assert dev.heads_per_chip(64) == 16  # heads over the 4-chip TP axis, with the hidden
    # TP32 is blocked until the shared stack has its placement; every blocker names its owning issue.
    assert not tp32.runnable
    assert {issue for _, issue in tp32.blockers} == {"#65885", "#65887", "#65892"}


def test_layout_cache_keys_differ():
    tp32 = hy4_layout.HY4_PREFILL_LAYOUTS["tp32"]
    dev = hy4_layout.HY4_PREFILL_LAYOUTS["glm_prefill_sp8_tp4"]
    assert dev.cache_key("8x4") == "8x4"  # existing dev caches stay valid
    assert tp32.cache_key("8x4") == "8x4_tp32"  # 2D-TP shards never share the dev layout's 8x4 cache


def test_resolve_layout(monkeypatch, expect_error):
    monkeypatch.delenv(hy4_layout.LAYOUT_ENV, raising=False)
    # Unset on a full Galaxy means tp32, which fails fast naming its blockers and the dev override.
    for mesh in ((4, 8, 0), (8, 4, 0)):
        with expect_error(NotImplementedError, "tp32 .*#65885, #65887, #65892.*glm_prefill_sp8_tp4"):
            hy4_layout.resolve_layout(*mesh, 64)
    assert hy4_layout.resolve_layout(2, 4, 0, 64) is None  # small test mesh, nothing requested
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, "tp32")
    with expect_error(NotImplementedError, "cannot run on the shared prefill stack yet"):
        hy4_layout.resolve_layout(4, 8, 0, 64)
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, "glm_prefill_sp8_tp4")
    assert hy4_layout.resolve_layout(8, 4, 0, 64).name == "glm_prefill_sp8_tp4"
    with expect_error(ValueError, "needs PREFILL_SP=8"):
        hy4_layout.resolve_layout(4, 8, 0, 64)
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, "decode_aligned_sp4_tp8")  # removed: TP32 supersedes it
    with expect_error(ValueError, "not a Hy4 prefill layout"):
        hy4_layout.resolve_layout(4, 8, 0, 64)
    monkeypatch.delenv(hy4_layout.LAYOUT_ENV)
    with expect_error(ValueError, "do not split"):
        hy4_layout.resolve_layout(4, 6, 0, 64)


def test_manifest_selects_tp32():
    env = json.loads(MANIFEST.read_text())["env"]
    assert env["PREFILL_MODEL"] == "hy4_preview"
    lay = hy4_layout.HY4_PREFILL_LAYOUTS[env["PREFILL_HY4_LAYOUT"]]
    assert lay.name == hy4_layout.DEFAULT_LAYOUT == "tp32"
    assert lay.matches(int(env["PREFILL_SP"]), int(env["PREFILL_TP"]), int(env["PREFILL_SP_AXIS"]))
    max_seq = int(env["PREFILL_MAX_SEQ_LEN"])
    # Bank-aligned KV: a multiple of stripes (all 32 chips) x 32 rows x 8 DRAM banks.
    assert max_seq % (lay.sp * lay.tp * 32 * 8) == 0
    assert max_seq >= Hy4Config.MAX_POSITION_EMBEDDINGS
    from models.demos.deepseek_v3_d_p.utils.chunk_config import DEFAULT_PREFILL_CHUNK_TOKENS

    assert max_seq % DEFAULT_PREFILL_CHUNK_TOKENS == 0  # the runtime asserts max_seq_len % chunk_size == 0
    assert int(env["PREFILL_NUM_USERS"]) >= 1


# ---------------------------------------------------------------------------------------------------------------
# Adapter (imports ttnn through the MLA adapter base)
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def adapter():
    pytest.importorskip("ttnn")
    from models.demos.common.prefill.adapter import ADAPTER_PATHS, get_adapter

    assert "hy4_preview" in ADAPTER_PATHS
    return get_adapter("hy4_preview")


def test_adapter_identity(adapter):
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter

    assert isinstance(adapter, GLM53Adapter)
    assert adapter.name == "hy4_preview"
    assert adapter.model_config is Hy4Config
    assert adapter.supports_mtp is True  # model.mtp_layers.0 via TtHy4Runtime (no iHC in the MTP layer)
    # iHC: a pipeline stage hands the next the 4 packed fp32 streams, not one bf16 hidden.
    assert (adapter.pipeline_activation_streams, adapter.pipeline_activation_dtype) == (4, "float32")
    assert adapter.builds_missing_weight_cache is True
    assert adapter.supports_tp_shard_kv and adapter.tp_shard_kv_default
    # Must not inherit GLM-5.3's golden trace or checkpoint.
    assert "GLM" not in adapter.prefill_trace_default and "GLM" not in adapter.test_prefill_trace_default
    assert "GLM" not in adapter.hf_model_default and "GLM" not in adapter.ttnn_cache_default


def test_adapter_layout_gate_and_cache_key(adapter, tmp_path, monkeypatch, expect_error):
    """The runner resolves weight_cache_path before it opens the mesh, so TP32 fails there, at startup."""
    import ttnn
    from models.common.utility_functions import is_blackhole
    from models.demos.common.prefill.adapter import PrefillRunParams

    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))
    monkeypatch.setattr(ttnn, "get_num_devices", lambda: 32)
    monkeypatch.delenv(hy4_layout.LAYOUT_ENV, raising=False)
    with expect_error(NotImplementedError, "#65885"):
        adapter.weight_cache_path((4, 8), sp_axis=0)
    with expect_error(NotImplementedError, "#65885"):
        adapter.weight_cache_path((8, 4), sp_axis=0)
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, "glm_prefill_sp8_tp4")
    arch = "bh" if is_blackhole() else "wh"
    assert adapter.weight_cache_path((8, 4), sp_axis=0) == tmp_path / f"hy4_preview_{arch}_32dev" / "8x4"
    params = PrefillRunParams(
        mesh_shape=(8, 4),
        num_layers=1,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=8192,
        chunk_size=4096,
        num_users=1,
        capacity_factor=1,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
    )
    adapter._check_layout(params)
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, "tp32")
    with expect_error(NotImplementedError, "glm_prefill_sp8_tp4"):
        adapter._check_layout(params)


def test_adapter_loads_and_validates_config_json(adapter, tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_HF_MODEL", str(tmp_path))
    monkeypatch.setenv("PREFILL_MAX_SEQ_LEN", "16384")
    (tmp_path / "config.json").write_text(json.dumps(_config_dict()))
    cfg = adapter.load_hf_config()
    assert cfg.max_seq_len == 16384 and cfg.first_k_dense_replace == 1
    bad = _config_dict()
    bad["hidden_size"] = 7168
    (tmp_path / "config.json").write_text(json.dumps(bad))
    with expect_error(ValueError, "hidden_size"):
        adapter.load_hf_config()
    (tmp_path / "config.json").unlink()
    assert adapter.load_hf_config().indexer_types == Hy4Config.indexer_types()  # falls back to the constants


def test_pending_deltas_guard(adapter, monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.runners.adapters import hy4_preview

    cfg = hy4_hf_config()
    # Sink + gate (#65921) are implemented in ttMLA and iHC (#65891) in the block: nothing is pending.
    assert cfg.enable_ihc is True
    assert hy4_preview.pending_deltas(cfg) == []
    monkeypatch.delenv(hy4_preview.ALLOW_PENDING_DELTAS_ENV, raising=False)
    adapter._check_pending_deltas(cfg)  # no override needed
    # The guard itself still works for a delta that is pending.
    monkeypatch.setitem(hy4_preview.PENDING_DELTAS, "enable_ihc", ("iHC", "#65891"))
    with expect_error(NotImplementedError, "#65891"):
        adapter._check_pending_deltas(cfg)
    monkeypatch.setenv(hy4_preview.ALLOW_PENDING_DELTAS_ENV, "1")
    adapter._check_pending_deltas(cfg)  # bring-up override: warns, does not raise


def test_adapter_builds_the_hy4_runtime_class(adapter):
    from models.demos.deepseek_v3_d_p.tt.hy4.runtime import TtHy4Runtime
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import load_hy4_trunk_state_dict

    assert adapter._runtime_cls() is TtHy4Runtime
    assert TtHy4Runtime._load_trunk_state_dict is load_hy4_trunk_state_dict


# ---------------------------------------------------------------------------------------------------------------
# Checkpoint reader
# ---------------------------------------------------------------------------------------------------------------

H, Q, KV, I, E, HEADS = 64, 32, 32, 32, 4, 2  # tiny stand-ins (every quantized last dim a multiple of 32)
NOPE, ROPE, INDEX_D, INDEX_ROPE = 4, 8, 8, 4  # tiny q/k head and index-key widths
DENSE_I = 64


def _to_mxfp8(w: torch.Tensor, block: int = 32):
    """Quantize along the last dim: per-block power-of-two scale (E8M0), fp8 e4m3 values."""
    shape = w.shape
    blocks = w.float().reshape(*shape[:-1], shape[-1] // block, block)
    amax = blocks.abs().amax(-1).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(exp)[..., None]).reshape(shape).to(torch.float8_e4m3fn)
    return q, (exp + 127).to(torch.uint8)


def _write_checkpoint(root: Path, mxfp8: bool) -> dict:
    """A 3-layer Hy4-layout checkpoint (layer 0 dense + full indexer, 1 MoE + full, 2 MoE + shared)."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g).to(torch.bfloat16)

    ref, store = {}, {}

    def put(key, t, quant=True):
        ref[key] = t
        if mxfp8 and quant and t.dim() >= 2 and t.shape[-1] % 32 == 0:
            q, s = _to_mxfp8(t)
            store[key] = q
            store[key + "_scale"] = s
        else:
            store[key] = t

    put("model.embed_tokens.weight", r(128, H), quant=False)
    for layer in range(3):
        p = f"model.layers.{layer}."
        put(p + "input_layernorm.weight", r(H), quant=False)
        put(p + "post_attention_layernorm.weight", r(H), quant=False)
        a = p + "self_attn."
        put(a + "q_a_proj.weight", r(Q, H))
        put(a + "q_a_layernorm.weight", r(Q), quant=False)
        put(a + "q_b_proj.weight", r(HEADS * (NOPE + ROPE), Q))
        put(a + "kv_a_proj_with_mqa.weight", r(KV + ROPE, H))
        put(a + "kv_a_layernorm.weight", r(KV), quant=False)
        put(a + "kv_b_proj.weight", r(HEADS * 8, KV))
        put(a + "o_proj.weight", r(H, HEADS * 8 * 4))
        put(a + "linear_gate.weight", r(HEADS * 8 * 4, H), quant=False)
        put(a + "learnable_sink_param", torch.randn(HEADS, generator=g), quant=False)
        if layer < 2:
            put(a + "indexer.wq_b.weight", r(32, Q))
            put(a + "indexer.wk.weight", r(8, H))
            put(a + "indexer.k_norm.weight", r(8), quant=False)
            put(a + "indexer.k_norm.bias", r(8), quant=False)
            put(a + "indexer.weights_proj.weight", r(2, H), quant=False)
        for sub in ("hc_attn_layer", "hc_mlp_layer"):
            put(f"{p}{sub}.hc_pre.hc_base", torch.randn(8, generator=g), quant=False)
            put(f"{p}{sub}.hc_pre.hc_fn", torch.randn(8, 4 * H, generator=g), quant=False)
            put(f"{p}{sub}.hc_pre.hc_scale", torch.randn(2, generator=g), quant=False)
        m = p + "mlp."
        if layer == 0:
            put(m + "gate_proj.weight", r(DENSE_I, H))
            put(m + "up_proj.weight", r(DENSE_I, H))
            put(m + "down_proj.weight", r(H, DENSE_I))
        else:
            put(m + "gate.weight", r(E, H), quant=False)
            put(m + "gate.e_score_correction_bias", torch.randn(E, generator=g), quant=False)
            put(m + "experts.gate_up_proj", r(E, 2 * I, H))
            put(m + "experts.down_proj", r(E, H, I))
            for proj in ("gate_proj", "up_proj"):
                put(f"{m}shared_experts.{proj}.weight", r(I, H))
            put(m + "shared_experts.down_proj.weight", r(H, I))
    root.mkdir(parents=True, exist_ok=True)
    save_file(store, str(root / "model-00001-of-00001.safetensors"))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in store}})
    )
    return ref


def _close(a, b, mxfp8):
    a, b = a.float(), b.float()
    if not mxfp8:
        return torch.equal(a, b)
    return torch.allclose(a, b, rtol=0.07, atol=1e-3 * b.abs().max().item())


def test_dequant_mxfp8_exact_on_representable_values(expect_error):
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import dequant_mxfp8

    w = torch.tensor([[1.0, -2.0, 0.5, 3.0] * 16], dtype=torch.float8_e4m3fn)  # [1, 64]
    s = torch.tensor([[127 + 3, 127 - 1]], dtype=torch.uint8)  # x8 then x0.5
    out = dequant_mxfp8(w, s)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out[0, :32].float(), w[0, :32].float() * 8)
    assert torch.equal(out[0, 32:].float(), w[0, 32:].float() * 0.5)
    with expect_error(ValueError, "does not cover"):
        dequant_mxfp8(w, s[:, :1])


@pytest.mark.parametrize("mxfp8", [False, True], ids=["bf16", "mxfp8"])
def test_checkpoint_reads_fused_experts_and_remaps_attention(tmp_path, mxfp8):
    pytest.importorskip("safetensors")
    pytest.importorskip("ttnn")  # hy4_layer_state_dict uses the indexer's layer-map helper
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc

    ref = _write_checkpoint(tmp_path, mxfp8)
    ckpt = hc.Hy4Checkpoint(tmp_path)

    # Routed experts: DeepSeek-layout keys answered from the fused tensors (gate rows, then up rows).
    for e in range(E):
        gu = ref["model.layers.1.mlp.experts.gate_up_proj"][e]
        assert _close(ckpt.bf16(f"model.layers.1.mlp.experts.{e}.gate_proj.weight"), gu[:I], mxfp8)
        assert _close(ckpt.bf16(f"model.layers.1.mlp.experts.{e}.up_proj.weight"), gu[I:], mxfp8)
        assert _close(
            ckpt.bf16(f"model.layers.1.mlp.experts.{e}.down_proj.weight"),
            ref["model.layers.1.mlp.experts.down_proj"][e],
            mxfp8,
        )

    class Cfg:  # the per-layer maps and head dims hy4_layer_state_dict reads
        indexer_types = ["full", "full", "shared"]
        mla_use_output_gate = True
        kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim = KV, NOPE, ROPE
        index_head_dim, index_rope_dim = INDEX_D, INDEX_ROPE

    class MCfg:
        MOE_INTERMEDIATE_SIZE = I
        NUM_ROUTED_EXPERTS = E

    everything = {wc.NORMS, wc.MLA, wc.DENSE_FFN}
    sd0 = hc.hy4_layer_state_dict(ckpt, Cfg, MCfg, 0, everything)
    mla0 = sd0["mla_weights"]
    assert _close(mla0["g_proj.weight"], ref["model.layers.0.self_attn.linear_gate.weight"], False)
    # In the device RoPE frames (test_hy4_rope_frames.py covers them): the indexer's rope block moves to the front; the
    # MLA projections are the checkpoint's (interleaved pairing, as ttMLA ropes).
    index_perm = hc.index_k_device_perm(INDEX_D, INDEX_ROPE)
    assert _close(
        mla0["indexer.k_norm_bias.weight"], ref["model.layers.0.self_attn.indexer.k_norm.bias"][index_perm], False
    )
    assert _close(mla0["kv_a_proj_with_mqa.weight"], ref["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"], mxfp8)
    assert mla0["q_a_layernorm.weight"].dtype == torch.float32
    assert mla0["indexer.weights_proj.weight"].dtype == torch.float32
    assert mla0["q_b_proj.weight"].dtype == torch.bfloat16
    assert _close(mla0["kv_b_proj.weight"], ref["model.layers.0.self_attn.kv_b_proj.weight"], mxfp8)
    assert not any("linear_gate" in k for k in mla0)
    assert "attn_sink.weight" not in mla0  # Cfg below turns the sink on only where asked
    assert set(sd0["ffn_weights"]) == set(wc._PROJ)
    assert _close(sd0["ffn_weights"]["gate_proj"], ref["model.layers.0.mlp.gate_proj.weight"], mxfp8)

    moe = {wc.MLA, wc.MOE_GATE, wc.SHARED_EXPERTS, wc.ROUTED_EXPERTS}
    sd2 = hc.hy4_layer_state_dict(ckpt, Cfg, MCfg, 2, moe)
    assert not any(k.startswith("indexer.") for k in sd2["mla_weights"])  # shared layer owns no indexer
    assert len(sd2["routed_expert_weights"]) == E
    assert _close(
        sd2["routed_expert_weights"][3]["up_proj"], ref["model.layers.2.mlp.experts.gate_up_proj"][3][I:], mxfp8
    )
    assert sd2["gate_weights"]["e_score_correction_bias"].dtype == torch.float32
    assert _close(
        sd2["shared_expert_weights"]["down_proj"], ref["model.layers.2.mlp.shared_experts.down_proj.weight"], mxfp8
    )

    Cfg.mla_use_output_gate = False
    assert "g_proj.weight" not in hc.hy4_layer_state_dict(ckpt, Cfg, MCfg, 1, {wc.MLA})["mla_weights"]

    Cfg.mla_learnable_sink = True
    sink = hc.hy4_layer_state_dict(ckpt, Cfg, MCfg, 1, {wc.MLA})["mla_weights"]["attn_sink.weight"]
    assert sink.dtype == torch.float32 and sink.shape == (HEADS,)
    assert torch.equal(sink, ref["model.layers.1.self_attn.learnable_sink_param"].float())

    ihc = hc.hy4_layer_state_dict(ckpt, Cfg, MCfg, 1, {wc.IHC})["ihc_weights"]
    assert set(ihc) == {"attn", "ffn"} and all(set(site) == {"fn", "base", "scale"} for site in ihc.values())
    assert torch.equal(ihc["ffn"]["fn"], ref["model.layers.1.hc_mlp_layer.hc_pre.hc_fn"].float())
    assert ihc["attn"]["fn"].dtype == torch.float32

    # The shared lazy loader drives the Hy4 per-layer reader.
    trunk = hc.load_hy4_trunk_state_dict(
        tmp_path,
        Cfg,
        MCfg,
        missing={wc.EMBEDDING: [-1], wc.ROUTED_EXPERTS: [1, 2]},
        first_layer_idx=0,
        num_layers=3,
    )
    assert trunk["embed_weight"].shape == (128, H)
    assert trunk["layers"][0] == {}
    assert len(trunk["layers"][2]["routed_expert_weights"]) == E


@pytest.mark.skipif(
    not (HY4_CHECKPOINT / "model.safetensors.index.json").is_file(), reason="Hy4-preview checkpoint not on this host"
)
def test_real_checkpoint_keys_match_the_reader():
    """Every key the Hy4 reader asks for exists, and indexer weights sit on exactly the full layers."""
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc

    wm = hc.Hy4Checkpoint(HY4_CHECKPOINT).weight_map
    for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", *hc.IHC_HEAD_KEYS):
        assert key in wm, key
    for layer in range(78):
        p = f"model.layers.{layer}."
        for src in (*hc._MLA_KEYS, *hc._GATE_KEYS):
            assert p + "self_attn." + src in wm, (layer, src)
        assert p + hc.SINK_KEY in wm
        assert all(p + k in wm for k in hc.IHC_LAYER_KEYS)
        has_indexer = all(p + "self_attn." + k in wm for k in hc._INDEXER_KEYS)
        has_any_indexer = any(k.startswith(p + "self_attn.indexer.") for k in wm)
        assert has_indexer == has_any_indexer == (layer in FULL_LAYERS), layer
        if layer < Hy4Config.NUM_DENSE_LAYERS:
            assert all(f"{p}mlp.{proj}.weight" in wm for proj in ("gate_proj", "up_proj", "down_proj"))
        else:
            for k in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias", "mlp.experts.gate_up_proj"):
                assert p + k in wm, (layer, k)
            assert all(f"{p}mlp.shared_experts.{proj}.weight" in wm for proj in ("gate_proj", "up_proj", "down_proj"))
