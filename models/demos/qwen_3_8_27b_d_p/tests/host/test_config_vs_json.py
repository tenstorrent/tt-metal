# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1: every constant in the config class against the vendored ``config.json``, and the binding
spec against its own invariants. Host only — no ttnn, no checkpoint, no HuggingFace.

This is the test that catches a hand-typed dimension. It is deliberately exhaustive rather than
representative: a wrong ``head_dim`` or a mis-transcribed ``mrope_section`` is invisible until a
PCC number is merely disappointing instead of obviously broken.
"""

from __future__ import annotations

import json

import pytest

from models.demos.qwen_3_8_27b_d_p.reference.config import (
    CONFIG_PATH,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35TextConfig,
)
from models.demos.qwen_3_8_27b_d_p.spec import load_spec

# (config-class attribute, json key inside text_config). Derived fields are checked separately.
_DIRECT_FIELDS = [
    ("hidden_size", "hidden_size"),
    ("intermediate_size", "intermediate_size"),
    ("num_hidden_layers", "num_hidden_layers"),
    ("vocab_size", "vocab_size"),
    ("rms_norm_eps", "rms_norm_eps"),
    ("hidden_act", "hidden_act"),
    ("tie_word_embeddings", "tie_word_embeddings"),
    ("max_position_embeddings", "max_position_embeddings"),
    ("full_attention_interval", "full_attention_interval"),
    ("num_attention_heads", "num_attention_heads"),
    ("num_key_value_heads", "num_key_value_heads"),
    ("head_dim", "head_dim"),
    ("attention_bias", "attention_bias"),
    ("attn_output_gate", "attn_output_gate"),
    ("linear_num_key_heads", "linear_num_key_heads"),
    ("linear_num_value_heads", "linear_num_value_heads"),
    ("linear_key_head_dim", "linear_key_head_dim"),
    ("linear_value_head_dim", "linear_value_head_dim"),
    ("linear_conv_kernel_dim", "linear_conv_kernel_dim"),
    ("output_gate_type", "output_gate_type"),
]

_ROPE_FIELDS = [
    ("rope_theta", "rope_theta"),
    ("partial_rotary_factor", "partial_rotary_factor"),
    ("mrope_interleaved", "mrope_interleaved"),
    ("rope_type", "rope_type"),
]


@pytest.fixture(scope="module")
def raw_text_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)["text_config"]


@pytest.fixture(scope="module")
def cfg() -> Qwen35TextConfig:
    return Qwen35TextConfig.from_json()


@pytest.mark.parametrize("attr, key", _DIRECT_FIELDS, ids=[a for a, _ in _DIRECT_FIELDS])
def test_scalar_field_matches_json(cfg, raw_text_config, attr, key):
    assert getattr(cfg, attr) == raw_text_config[key]


@pytest.mark.parametrize("attr, key", _ROPE_FIELDS, ids=[a for a, _ in _ROPE_FIELDS])
def test_rope_field_matches_json(cfg, raw_text_config, attr, key):
    assert getattr(cfg, attr) == raw_text_config["rope_parameters"][key]


def test_mrope_section_matches_json(cfg, raw_text_config):
    assert list(cfg.mrope_section) == raw_text_config["rope_parameters"]["mrope_section"]


def test_layer_types_match_json(cfg, raw_text_config):
    assert list(cfg.layer_types) == raw_text_config["layer_types"]


def test_derived_fields():
    """The derived widths, spelled out so a refactor of __post_init__ cannot drift silently."""
    cfg = Qwen35TextConfig.from_json()
    assert cfg.rotary_dim == 64  # head_dim 256 * partial_rotary_factor 0.25
    assert cfg.num_key_value_groups == 6  # 24 q heads / 4 kv heads
    assert cfg.gdn_key_dim == 2048  # 16 key heads * 128
    assert cfg.gdn_value_dim == 6144  # 48 value heads * 128
    assert cfg.gdn_conv_dim == 10240  # 2 * key_dim + value_dim
    assert cfg.gdn_num_value_groups == 3  # 48 value heads / 16 key heads


def test_hybrid_schedule(cfg, expect_error):
    """48 Gated DeltaNet layers + 16 GQA layers, full attention every 4th (indices 3, 7, ... 63)."""
    assert len(cfg.full_attention_layers) == 16
    assert len(cfg.linear_attention_layers) == 48
    assert cfg.full_attention_layers == tuple(range(3, 64, 4))
    assert cfg.layer_types[0] == LINEAR_ATTENTION and cfg.layer_types[3] == FULL_ATTENTION
    # kv_slot packs ONLY the full-attention layers, 0..15 in order.
    assert [cfg.kv_slot(i) for i in cfg.full_attention_layers] == list(range(16))
    with expect_error(AssertionError, "layer 0 is linear_attention"):
        cfg.kv_slot(0)


def test_gated_deltanet_head_geometry(cfg):
    """The 3:1 value:key head ratio is what makes the delta rule GQA-shaped; q/k are
    repeat_interleaved to the value head count before the scan."""
    assert cfg.linear_num_value_heads == cfg.linear_num_key_heads * cfg.gdn_num_value_groups
    assert cfg.linear_key_head_dim == cfg.linear_value_head_dim, "the ttnn scan assumes K == V per head"


def test_spec_is_self_consistent():
    spec = load_spec()
    assert spec.model_name == "qwen_3_8_27b"
    assert spec.hf_repo == "Qwen/Qwen3.8-27B"
    assert (spec.sp, spec.tp) == (8, 4)
    assert spec.mesh_shape == (8, 4)
    spec.validate()


def test_spec_and_config_agree_on_divisibility():
    """The graded TP must divide every head count the model shards on it."""
    spec = load_spec()
    cfg = Qwen35TextConfig.from_json()
    assert cfg.num_attention_heads % spec.tp == 0
    assert cfg.num_key_value_heads % spec.tp == 0
    assert cfg.linear_num_key_heads % spec.tp == 0
    assert cfg.linear_num_value_heads % spec.tp == 0
    assert cfg.hidden_size % (spec.tp * 32) == 0
    assert cfg.intermediate_size % (spec.tp * 32) == 0
    assert spec.max_seq_len <= cfg.max_position_embeddings


def test_spec_dataformat_defaults_resolve():
    """Empty overrides inherit their group default rather than becoming ''."""
    spec = load_spec()
    assert spec.attention_weight_dtype == spec.weight_dtype
    assert spec.mlp_up_dtype == spec.mlp_gate_dtype == spec.mlp_down_dtype == spec.weight_dtype
    assert spec.activation_dtype == "bfloat16"
    assert spec.kv_cache_dtype == "bfloat8_b"
