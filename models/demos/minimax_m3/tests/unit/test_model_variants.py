# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MiniMax-M3 / MiniMax-M3.1 configs and variant selection (utils/model_variant.py).

Both repo configs load and pass the supported-config checks, variant detection picks the right one from each
checkpoint's config.json (the real weka configs when present, read-only), the per-layer attention / MLP types and
RoPE base come through to the layer helpers, ModelArgs, the torch reference and the prefill adapter, and configs the
kernels cannot run are rejected.
"""

import copy
import json
import os

import pytest

from models.demos.minimax_m3.reference.model import MiniMaxM3Config
from models.demos.minimax_m3.utils.checkpoint import check_supported_config, check_supported_quantization
from models.demos.minimax_m3.utils.general_utils import is_dense_mlp_layer, is_sparse_attention_layer, msa_params
from models.demos.minimax_m3.utils.model_variant import (
    DEFAULT_VARIANT,
    VARIANT_ENV,
    VARIANTS,
    config_differences,
    detect_variant,
    get_variant,
    load_config,
    load_hf_config,
    read_checkpoint_text_config,
    select_variant,
)

M3, M31 = "MiniMax-M3", "MiniMax-M3.1"
CHECKPOINTS = {M3: VARIANTS[M3].weights_dir, M31: VARIANTS[M31].weights_dir}


@pytest.fixture
def clean_env(monkeypatch):
    """No checkpoint and no explicit variant unless a test sets one."""
    for var in ("HF_MODEL", "PREFILL_HF_MODEL", "TT_CACHE_PATH", VARIANT_ENV):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _checkpoint(name):
    path = CHECKPOINTS[name]
    if not os.path.isfile(os.path.join(path, "config.json")):
        pytest.skip(f"{name} checkpoint config not on this host ({path})")
    return path


def _write_checkpoint(tmp_path, text_config, wrapped=True):
    cfg = {"architectures": ["MiniMaxM3SparseForConditionalGeneration"], "text_config": text_config}
    (tmp_path / "config.json").write_text(json.dumps(cfg if wrapped else text_config))
    return str(tmp_path)


# ---- repo configs -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", [M3, M31])
def test_repo_config_loads_and_is_supported(name):
    cfg = load_config(name)
    check_supported_config(cfg)
    assert cfg["num_hidden_layers"] == 60 and len(cfg["moe_layer_freq"]) == 60
    assert get_variant(name).config_path.is_file()


def test_variants_differ_only_where_the_releases_do():
    assert config_differences(load_config(M3), load_config(M31)) == [
        "num_mtp_modules",
        "num_nextn_predict_layers",
        "rope_theta",
        "sparse_attention_config",
    ]
    m3_sparse, m31_sparse = load_config(M3)["sparse_attention_config"], load_config(M31)["sparse_attention_config"]
    assert config_differences(m3_sparse, m31_sparse) == [
        "sparse_attention_freq",
        "sparse_disable_index_value",
        "sparse_enable_estimate_lse",
        "sparse_enable_kv_share",
        "sparse_enable_vo_rope",
        "sparse_kvshare_value_drop_rope_dim",
    ]
    assert "quantization_config" in load_config(M31) and "quantization_config" not in load_config(M3)


def test_load_config_returns_copies():
    cfg = load_config(M3)
    cfg["sparse_attention_config"]["sparse_attention_freq"][0] = 1
    assert load_config(M3)["sparse_attention_config"]["sparse_attention_freq"][0] == 0


# ---- per-layer types and RoPE -----------------------------------------------------------------------------
@pytest.mark.parametrize("name, msa_layers", [(M3, range(3, 60)), (M31, range(60))])
def test_layer_types_follow_config(name, msa_layers):
    hf_config = load_hf_config(name)
    attention = [is_sparse_attention_layer(hf_config, i) for i in range(60)]
    assert attention == [i in msa_layers for i in range(60)]
    assert [is_dense_mlp_layer(hf_config, i) for i in range(60)] == [i < 3 for i in range(60)]
    assert msa_params(hf_config) == {"msa_block_size": 128, "msa_topk_blocks": 16, "msa_index_dim": 128}


@pytest.mark.parametrize("name, theta", [(M3, 5_000_000), (M31, 10_000_000)])
def test_reference_config_follows_variant(name, theta):
    cfg = MiniMaxM3Config.from_hf_config(load_config(name))
    assert cfg.rope_theta == theta and cfg.rotary_dim == 64
    assert [cfg.is_sparse_attn_layer(i) for i in range(3)] == [name == M31] * 3
    assert cfg.dense_equiv_tokens == 2048


@pytest.mark.parametrize("name, theta", [(M3, 5_000_000), (M31, 10_000_000)])
def test_model_args_dummy_weights_follow_variant(clean_env, name, theta):
    from models.demos.minimax_m3.tt.model_config import ModelArgs

    clean_env.setenv(VARIANT_ENV, name)
    args = ModelArgs(mesh_device=None, dummy_weights=True)
    assert args.model_variant.name == name and args.rope_theta == theta
    assert is_sparse_attention_layer(args.hf_config, 0) == (name == M31)


@pytest.mark.parametrize("name", [M3, M31])
def test_prefill_adapter_defaults_follow_variant(clean_env, name):
    from models.demos.minimax_m3.tt.runners.adapters.minimax_m3 import MiniMaxM3PrefillAdapter

    clean_env.delenv("PREFILL_HF_MODEL", raising=False)
    clean_env.setenv(VARIANT_ENV, name)
    adapter = MiniMaxM3PrefillAdapter()
    variant = get_variant(name)
    assert adapter.variant == variant and adapter.hf_model_default.rstrip("/") == variant.weights_dir
    assert adapter.prefill_trace_default.startswith(variant.golden_dir + "/")
    assert adapter.model_config.NUM_LAYERS == 60 and adapter.model_config.EMB_SIZE == 6144


def test_weight_cache_root_is_per_variant(clean_env, tmp_path):
    from models.demos.minimax_m3.tt.model_config import default_weight_cache_root

    for variant in VARIANTS.values():
        expected = variant.cache_root if os.path.isdir(variant.cache_root) else variant.weights_dir
        assert str(default_weight_cache_root(variant.weights_dir)) == expected
    assert default_weight_cache_root(str(tmp_path)) == tmp_path
    clean_env.setenv("TT_CACHE_PATH", str(tmp_path / "cache"))
    assert default_weight_cache_root(VARIANTS[M31].weights_dir) == tmp_path / "cache"


# ---- detection / selection --------------------------------------------------------------------------------
@pytest.mark.parametrize("name", [M3, M31])
def test_detect_from_repo_config(name):
    assert detect_variant(load_config(name)).name == name
    no_arch = load_config(name)
    del no_arch["architectures"]  # falls back to the model-defining fields
    assert detect_variant(no_arch).name == name


def test_detect_rejects_unknown_config(expect_error):
    cfg = load_config(M31)
    del cfg["architectures"]
    cfg["rope_theta"] = 1234
    with expect_error(ValueError, "cannot tell which"):
        detect_variant(cfg)


def test_detect_patched_checkpoint_keeps_architecture():
    cfg = load_config(M31, num_hidden_layers=4)  # truncated checkpoint: still M3.1, with a warning
    assert detect_variant(cfg).name == M31


@pytest.mark.parametrize("name", [M3, M31])
def test_real_checkpoint_configs(clean_env, name):
    path = _checkpoint(name)
    text_config = read_checkpoint_text_config(path)
    assert detect_variant(text_config).name == name
    assert select_variant(path).name == name
    clean_env.setenv("HF_MODEL", path)
    assert select_variant().name == name
    # The repo config is the checkpoint's text_config verbatim (metadata aside).
    assert config_differences(load_config(name), text_config) == []
    with open(os.path.join(path, "config.json")) as f:
        check_supported_quantization(json.load(f).get("quantization_config"))


def test_select_default_and_override(clean_env, expect_error):
    assert select_variant().name == DEFAULT_VARIANT == M3
    clean_env.setenv(VARIANT_ENV, M31)
    assert select_variant().name == M31
    assert select_variant("/nonexistent/checkpoint").name == M31  # no config.json: the env decides
    clean_env.setenv(VARIANT_ENV, "MiniMax-M4")
    with expect_error(ValueError, "unknown MiniMax-M3 model variant"):
        select_variant()


@pytest.mark.parametrize("wrapped", [True, False], ids=["vl_wrapped", "flat"])
def test_select_from_checkpoint_config(clean_env, tmp_path, wrapped, expect_error):
    path = _write_checkpoint(tmp_path, load_config(M31), wrapped)
    assert select_variant(path).name == M31
    clean_env.setenv(VARIANT_ENV, M31)
    assert select_variant(path).name == M31
    clean_env.setenv(VARIANT_ENV, M3)
    with expect_error(ValueError, "but the checkpoint"):
        select_variant(path)


def test_select_unrecognized_checkpoint_needs_override(clean_env, tmp_path, expect_error):
    cfg = load_config(M3)
    del cfg["architectures"]
    cfg["rope_theta"] = 1234
    path = _write_checkpoint(tmp_path, cfg)
    with expect_error(ValueError, VARIANT_ENV):
        select_variant(path)
    clean_env.setenv(VARIANT_ENV, M3)
    assert select_variant(path).name == M3


# ---- supported-config guards ------------------------------------------------------------------------------
def _sparse(**kw):
    def patch(cfg):
        cfg["sparse_attention_config"].update(kw)

    return patch


def _set(**kw):
    return lambda cfg: cfg.update(kw)


def _drop(key):
    return lambda cfg: cfg.pop(key)


@pytest.mark.parametrize(
    "patch, error, match",
    [
        (_set(swiglu_alpha=1.5), NotImplementedError, "routed-expert kernels"),
        (_set(swiglu_limit=10.0), NotImplementedError, "routed-expert kernels"),
        (_set(scoring_func="softmax"), NotImplementedError, "sigmoid-only"),
        (_set(n_shared_experts=2), NotImplementedError, "exactly one"),
        (_set(hidden_act="silu"), NotImplementedError, "swigluoai"),
        (_set(qk_norm_type="per_layer"), NotImplementedError, "per_head"),
        (_set(tie_word_embeddings=True), NotImplementedError, "lm_head"),
        (_set(attention_output_gate=True), NotImplementedError, "attention_output_gate"),
        (_set(rope_scaling={"type": "yarn"}), NotImplementedError, "plain RoPE"),
        (_set(rotary_dim=48), NotImplementedError, "partial_rotary_factor"),
        (_set(moe_layer_freq=[0] * 59), NotImplementedError, "moe_layer_freq"),
        (_drop("rope_theta"), ValueError, "rope_theta missing"),
        (_drop("use_gemma_norm"), ValueError, "use_gemma_norm missing"),
        (_set(use_qk_norm=False), NotImplementedError, "use_qk_norm"),
        (_set(use_gemma_norm=False), NotImplementedError, "use_gemma_norm"),
        (_set(use_routing_bias=False), NotImplementedError, "use_routing_bias"),
        (_sparse(sparse_local_block=2), NotImplementedError, "sparse_local_block"),
        (_sparse(sparse_init_block=1), NotImplementedError, "sparse_init_block"),
        (_sparse(sparse_score_type="mean"), NotImplementedError, "sparse_score_type"),
        (_sparse(sparse_num_index_heads=8), NotImplementedError, "sparse_num_index_heads"),
        (_sparse(sparse_index_dim=64), NotImplementedError, "sparse_index_dim"),
        (_sparse(sparse_topk_blocks=None), NotImplementedError, "sparse_topk_blocks missing"),
        (_sparse(sparse_enable_kv_share=True), NotImplementedError, "sparse_enable_kv_share"),
        (_sparse(sparse_disable_index_value=[0] * 60), NotImplementedError, "index value branch"),
    ],
)
@pytest.mark.parametrize("name", [M3, M31])
def test_unsupported_config_rejected(name, patch, error, match, expect_error):
    cfg = copy.deepcopy(load_config(name))
    patch(cfg)
    with expect_error(error, match):
        check_supported_config(cfg)


def test_partial_model_config_supported():
    # Partial-model runs lower num_hidden_layers and keep the 60-entry per-layer lists.
    check_supported_config(load_config(M31, num_hidden_layers=5))


@pytest.mark.parametrize(
    "patch, ok",
    [
        ({}, True),
        ({"group_size": 32}, False),
        ({"weight_block_size": [1, 64]}, False),
        ({"quant_method": "fp8"}, False),
        ({"moe_quant_format": "nvfp4-unpacked"}, False),
    ],
)
def test_quantization_guard(patch, ok, expect_error):
    qcfg = dict(load_config(M31)["quantization_config"], **patch)
    if ok:
        check_supported_quantization(qcfg)
        check_supported_quantization(None)  # bf16 checkpoint
    else:
        with expect_error(NotImplementedError, "unsupported checkpoint quantization"):
            check_supported_quantization(qcfg)
