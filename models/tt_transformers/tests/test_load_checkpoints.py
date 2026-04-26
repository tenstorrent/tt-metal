# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight CPU-only regression tests for load_checkpoints.py.

This file intentionally keeps both:
- Phi checkpoint-conversion / partial-RoPE tests from this branch
- Upstream multimodal HF-key-remapping regression tests
"""

from types import SimpleNamespace

import pytest
import torch

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_qkv_to_meta_format,
    convert_hf_to_meta,
    convert_hf_to_meta_mllama,
    convert_meta_to_hf,
    map_hf_to_meta_keys_mllama,
    split_hf_keys,
    standardize_hf_keys_multimodal,
)


def _build_phi_hf_state_dict(dim: int = 32, hidden_dim: int = 64):
    torch.manual_seed(0)
    return {
        "model.embed_tokens.weight": torch.randn(128, dim),
        "lm_head.weight": torch.randn(128, dim),
        "model.layers.0.input_layernorm.weight": torch.randn(dim),
        "model.layers.0.input_layernorm.bias": torch.randn(dim),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(dim, dim),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(dim),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(dim, dim),
        "model.layers.0.self_attn.k_proj.bias": torch.randn(dim),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(dim, dim),
        "model.layers.0.self_attn.v_proj.bias": torch.randn(dim),
        "model.layers.0.self_attn.dense.weight": torch.randn(dim, dim),
        "model.layers.0.self_attn.dense.bias": torch.randn(dim),
        "model.layers.0.mlp.fc1.weight": torch.randn(hidden_dim, dim),
        "model.layers.0.mlp.fc1.bias": torch.randn(hidden_dim),
        "model.layers.0.mlp.fc2.weight": torch.randn(dim, hidden_dim),
        "model.layers.0.mlp.fc2.bias": torch.randn(dim),
        "model.final_layernorm.weight": torch.randn(dim),
        "model.final_layernorm.bias": torch.randn(dim),
    }


def test_phi_roundtrip_conversion_with_partial_rotary():
    head_dim = 8
    n_heads = 4
    n_kv_heads = 4
    rotary_dim = 4

    hf_state = _build_phi_hf_state_dict(dim=head_dim * n_heads)

    meta_state = convert_hf_to_meta(
        hf_state,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rotary_dim=rotary_dim,
        model_type="phi",
    )

    assert "layers.0.attention.wo.weight" in meta_state
    assert "layers.0.feed_forward.w1.weight" in meta_state
    assert "layers.0.feed_forward.w2.weight" in meta_state
    assert "norm.weight" in meta_state

    hf_roundtrip = convert_meta_to_hf(
        meta_state,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        model_type="phi",
    )

    for key, original in hf_state.items():
        assert key in hf_roundtrip
        assert torch.equal(hf_roundtrip[key], original), f"Mismatch for key: {key}"


def test_partial_rotary_only_transforms_rotary_slice():
    head_dim = 8
    n_heads = 2
    rotary_dim = 4
    in_dim = 10

    q_weight = torch.arange(n_heads * head_dim * in_dim, dtype=torch.float32).reshape(n_heads * head_dim, in_dim)
    k_bias = torch.arange(n_heads * head_dim, dtype=torch.float32)

    converted = convert_hf_qkv_to_meta_format(
        {
            "q_proj.weight": q_weight,
            "k_proj.bias": k_bias,
        },
        head_dim=head_dim,
        rotary_dim=rotary_dim,
    )

    q_in = q_weight.reshape(n_heads, head_dim, in_dim)
    q_out = converted["q_proj.weight"].reshape(n_heads, head_dim, in_dim)

    assert torch.equal(q_out[:, rotary_dim:, :], q_in[:, rotary_dim:, :])


def test_partial_rotary_dim_validation():
    with pytest.raises(ValueError, match="Invalid rotary_dim"):
        convert_hf_qkv_to_meta_format(
            {"q_proj.weight": torch.randn(16, 16)},
            head_dim=8,
            rotary_dim=3,
        )


def _make_mllama_config(num_hidden_layers=4, cross_attention_layers=None):
    """Build a minimal config object accepted by map_hf_to_meta_keys_mllama."""
    if cross_attention_layers is None:
        cross_attention_layers = [1]
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            cross_attention_layers=cross_attention_layers,
        )
    )


def _make_sample_mllama_state_dict():
    """Return a minimal HF-format state_dict that covers projector keys plus
    the embed_tokens and lm_head keys required by map_hf_to_meta_keys_mllama."""
    t = torch.zeros(1)
    return {
        "model.multi_modal_projector.weight": t,
        "model.multi_modal_projector.bias": t,
        "model.vision_model.layernorm_pre.weight": t,
        "model.vision_model.layernorm_pre.bias": t,
        "lm_head.weight": torch.zeros(16, 4),
        "model.embed_tokens.weight": torch.zeros(16, 4),
    }


class TestMllamaProjectorKeyRemap:
    """Ensure multimodal projector keys are remapped correctly."""

    def test_projector_keys_after_full_pipeline(self):
        state_dict = _make_sample_mllama_state_dict()
        config = _make_mllama_config()

        state_dict = standardize_hf_keys_multimodal(state_dict)
        state_dict = split_hf_keys(state_dict)
        state_dict = map_hf_to_meta_keys_mllama(state_dict, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
        assert not any("multi_modal_projector" in k for k in state_dict)

    def test_projector_keys_via_convert_hf_to_meta_mllama(self):
        state_dict = _make_sample_mllama_state_dict()
        config = _make_mllama_config()
        head_dim = 64

        state_dict = standardize_hf_keys_multimodal(state_dict)
        state_dict = convert_hf_to_meta_mllama(state_dict, head_dim, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
        assert not any("multi_modal_projector" in k for k in state_dict)

    def test_projector_keys_without_standardize(self):
        t = torch.zeros(1)
        state_dict = {
            "model.multi_modal_projector.weight": t,
            "model.multi_modal_projector.bias": t,
            "model.embed_tokens.weight": torch.zeros(16, 4),
        }
        config = _make_mllama_config()

        state_dict = split_hf_keys(state_dict)
        state_dict = map_hf_to_meta_keys_mllama(state_dict, config)

        assert "vision_model.vision_projection.weight" in state_dict
        assert "vision_model.vision_projection.bias" in state_dict
