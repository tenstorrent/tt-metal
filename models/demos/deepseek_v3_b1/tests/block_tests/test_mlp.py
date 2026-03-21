# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MLP unit tests + comparison vs HF MLP module."""

import torch

from helpers import seed_all, assert_close, make_profiles, rms_norm
from mlp import mlp_torch, mlp_tt


def _make_mlp_weights(profile, dtype=torch.float32):
    h = profile["hidden_size"]
    mi = profile["ffn_intermediate"]
    return dict(
        w_gate=torch.randn(mi, h, dtype=dtype),
        w_up=torch.randn(mi, h, dtype=dtype),
        w_down=torch.randn(h, mi, dtype=dtype),
    )


def test_mlp_matches_per_profile_cpu_fp32():
    seed_all(42)
    profiles = make_profiles("tiny")
    b, seq = 2, 6
    for name, prof in profiles.items():
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_mlp_weights(prof)
        out_torch = mlp_torch(x, **weights)
        out_tt = mlp_tt(x, **weights)
        assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_mlp_batch1_seq1():
    seed_all(77)
    prof = make_profiles("tiny")["deepseek_v3"]
    h = prof["hidden_size"]
    x = torch.randn(1, 1, h)
    weights = _make_mlp_weights(prof)
    out_torch = mlp_torch(x, **weights)
    out_tt = mlp_tt(x, **weights)
    assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_mlp_odd_seq():
    seed_all(55)
    prof = make_profiles("tiny")["glm4_355b"]
    h = prof["hidden_size"]
    x = torch.randn(2, 7, h)
    weights = _make_mlp_weights(prof)
    out_torch = mlp_torch(x, **weights)
    out_tt = mlp_tt(x, **weights)
    assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_mlp_llama():
    """
    Compare mlp_torch against real HF LlamaMLP (SwiGLU).
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = LlamaDecoderLayer(config, layer_idx=0)
    layer.eval()

    normed = rms_norm(x, layer.post_attention_layernorm.weight.data, config.rms_norm_eps)

    hf_mlp_out = layer.mlp(normed)

    mlp = layer.mlp
    our_torch_out = mlp_torch(
        normed,
        w_gate=mlp.gate_proj.weight.data,
        w_up=mlp.up_proj.weight.data,
        w_down=mlp.down_proj.weight.data,
    )
    our_tt_out = mlp_tt(
        normed,
        w_gate=mlp.gate_proj.weight.data,
        w_up=mlp.up_proj.weight.data,
        w_down=mlp.down_proj.weight.data,
    )

    assert_close(hf_mlp_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mlp_torch vs HF Llama MLP mismatch"
    assert_close(hf_mlp_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mlp_tt vs HF Llama MLP mismatch"


@torch.no_grad()
def test_mlp_qwen2():
    """
    Compare mlp_torch against real HF Qwen2MLP (SwiGLU).
    """
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    config = Qwen2Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(70)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Qwen2DecoderLayer(config, layer_idx=0)
    layer.eval()

    normed = rms_norm(x, layer.post_attention_layernorm.weight.data, config.rms_norm_eps)

    hf_mlp_out = layer.mlp(normed)

    mlp = layer.mlp
    our_torch_out = mlp_torch(
        normed,
        w_gate=mlp.gate_proj.weight.data,
        w_up=mlp.up_proj.weight.data,
        w_down=mlp.down_proj.weight.data,
    )
    our_tt_out = mlp_tt(
        normed,
        w_gate=mlp.gate_proj.weight.data,
        w_up=mlp.up_proj.weight.data,
        w_down=mlp.down_proj.weight.data,
    )

    assert_close(hf_mlp_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mlp_torch vs HF Qwen2 MLP mismatch"
    assert_close(hf_mlp_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mlp_tt vs HF Qwen2 MLP mismatch"
