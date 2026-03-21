# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_qkv_to_meta_format,
    convert_hf_to_meta,
    convert_meta_to_hf,
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

    # Non-rotary channels should be untouched by permutation.
    assert torch.equal(q_out[:, rotary_dim:, :], q_in[:, rotary_dim:, :])


def test_partial_rotary_dim_validation():
    with pytest.raises(ValueError, match="Invalid rotary_dim"):
        convert_hf_qkv_to_meta_format(
            {"q_proj.weight": torch.randn(16, 16)},
            head_dim=8,
            rotary_dim=3,
        )
