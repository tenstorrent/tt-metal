# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MLA unit tests + comparison vs HF attention."""

import torch

from helpers import seed_all, assert_close, make_profiles, rms_norm
from mla import mla_attention_torch, mla_attention_tt


ROPE_DIM = 4


def _make_mla_weights(profile, dtype=torch.float32):
    h = profile["hidden_size"]
    nq = profile["num_q_heads"]
    nkv = profile["num_kv_heads"]
    hd = profile["head_dim"]
    lat = profile["kv_latent_dim"]
    nope_dim = hd - ROPE_DIM
    v_head_dim = hd  # use same as head_dim for basic tests
    return dict(
        wq=torch.randn(nq * hd, h, dtype=dtype),
        w_kv_a=torch.randn(lat + ROPE_DIM, h, dtype=dtype),
        kv_a_layernorm_weight=torch.ones(lat, dtype=dtype),
        w_kv_b=torch.randn(nkv * (nope_dim + v_head_dim), lat, dtype=dtype),
        wo=torch.randn(h, nq * v_head_dim, dtype=dtype),
    )


def _make_rope(b, seq, rope_dim=ROPE_DIM):
    """Create simple cos/sin position embeddings for testing."""
    cos = torch.ones(b, seq, rope_dim)
    sin = torch.zeros(b, seq, rope_dim)
    return cos, sin


def test_mla_matches_hf_per_profile_cpu_fp32():
    seed_all(42)
    profiles = make_profiles("tiny")
    b, seq = 2, 6
    for name, prof in profiles.items():
        if not prof["use_mla"]:
            continue
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_mla_weights(prof)
        common = dict(
            num_q_heads=prof["num_q_heads"],
            num_kv_heads=prof["num_kv_heads"],
            kv_latent_dim=prof["kv_latent_dim"],
            qk_rope_head_dim=ROPE_DIM,
            position_embeddings=_make_rope(b, seq),
        )
        out_torch, _ = mla_attention_torch(x, **weights, **common)
        out_tt, _ = mla_attention_tt(x, **weights, **common)
        assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_mla_cache_chunking_equivalence():
    seed_all(123)
    prof = make_profiles("tiny")["deepseek_v3"]
    h = prof["hidden_size"]
    b, seq = 1, 8
    split = 4
    x = torch.randn(b, seq, h)
    weights = _make_mla_weights(prof)
    common = dict(
        num_q_heads=prof["num_q_heads"],
        num_kv_heads=prof["num_kv_heads"],
        kv_latent_dim=prof["kv_latent_dim"],
        qk_rope_head_dim=ROPE_DIM,
    )

    # Full pass with causal mask (additive)
    causal_full = torch.zeros(1, 1, seq, seq)
    causal_full.masked_fill_(~torch.tril(torch.ones(seq, seq, dtype=torch.bool)), float("-inf"))
    out_full, _ = mla_attention_torch(
        x, **weights, **common,
        position_embeddings=_make_rope(b, seq),
        attention_mask=causal_full,
    )

    # Chunk 1
    x1 = x[:, :split, :]
    causal1 = torch.zeros(1, 1, split, split)
    causal1.masked_fill_(~torch.tril(torch.ones(split, split, dtype=torch.bool)), float("-inf"))
    out1, cache = mla_attention_torch(
        x1, **weights, **common,
        position_embeddings=_make_rope(b, split),
        attention_mask=causal1, use_cache=True,
    )

    # Chunk 2
    x2 = x[:, split:, :]
    kv_len = seq
    causal2 = torch.zeros(1, 1, seq - split, kv_len)
    for i in range(seq - split):
        causal2[0, 0, i, split + i + 1:] = float("-inf")
    out2, _ = mla_attention_torch(
        x2, **weights, **common,
        position_embeddings=_make_rope(b, seq - split),
        attention_mask=causal2, past_key_value=cache, use_cache=True,
    )

    out_chunked = torch.cat([out1, out2], dim=1)
    assert_close(out_full, out_chunked, atol=1e-3, rtol=1e-3)


@torch.no_grad()
def test_mla_deepseek_v3():
    """
    Compare mla_attention_torch against HF DeepseekV3Attention
    with RoPE (nope+rope split) and position_ids.

    Adaptations:
    - q_lora_rank=None → direct Q projection (no LoRA), matches our wq
    """
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3RotaryEmbedding,
    )

    nope_dim = 8
    rope_dim = 4
    v_head_dim = 8
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=nope_dim,
        v_head_dim=v_head_dim,
        intermediate_size=128,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = DeepseekV3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)

    hf_attn_out, _ = layer.self_attn(
        normed, position_embeddings=(cos, sin), attention_mask=None,
    )

    attn = layer.self_attn

    mla_kwargs = dict(
        wq=attn.q_proj.weight.data,
        w_kv_a=attn.kv_a_proj_with_mqa.weight.data,
        kv_a_layernorm_weight=attn.kv_a_layernorm.weight.data,
        w_kv_b=attn.kv_b_proj.weight.data,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        kv_latent_dim=config.kv_lora_rank,
        position_embeddings=(cos, sin),
        qk_rope_head_dim=rope_dim,
    )

    our_torch_out, _ = mla_attention_torch(normed, **mla_kwargs)
    our_tt_out, _ = mla_attention_tt(normed, **mla_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_torch vs HF DeepSeek V3 attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_tt vs HF DeepSeek V3 attention mismatch"


@torch.no_grad()
def test_mla_deepseek_v3_full():
    """
    Full DeepSeek V3 MLA: Q LoRA (q_a_proj → q_a_layernorm → q_b_proj) +
    KV layernorm (kv_a_layernorm) — no monkey-patching.
    """
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3RotaryEmbedding,
    )

    nope_dim = 8
    rope_dim = 4
    v_head_dim = 8
    q_lora_rank = 16
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=nope_dim,
        v_head_dim=v_head_dim,
        intermediate_size=128,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = DeepseekV3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)

    hf_attn_out, _ = layer.self_attn(
        normed, position_embeddings=(cos, sin), attention_mask=None,
    )

    attn = layer.self_attn

    mla_kwargs = dict(
        # Q LoRA: q_a_proj → q_a_layernorm → q_b_proj
        w_q_a=attn.q_a_proj.weight.data,
        q_a_layernorm_weight=attn.q_a_layernorm.weight.data,
        q_a_layernorm_eps=config.rms_norm_eps,
        wq=attn.q_b_proj.weight.data,
        # KV path with layernorm
        w_kv_a=attn.kv_a_proj_with_mqa.weight.data,
        kv_a_layernorm_weight=attn.kv_a_layernorm.weight.data,
        kv_a_layernorm_eps=config.rms_norm_eps,
        w_kv_b=attn.kv_b_proj.weight.data,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        kv_latent_dim=config.kv_lora_rank,
        position_embeddings=(cos, sin),
        qk_rope_head_dim=rope_dim,
    )

    our_torch_out, _ = mla_attention_torch(normed, **mla_kwargs)
    our_tt_out, _ = mla_attention_tt(normed, **mla_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_torch vs HF DeepSeek V3 full MLA mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_tt vs HF DeepSeek V3 full MLA mismatch"


# NOTE: absorbed attention test removed — absorbed path is now a commented-out
# optimization example in mla.py.  See mla.py for the implementation.


@torch.no_grad()
def test_mla_kimi_k25():
    """
    Same as test_mla_deepseek_v3 but with Kimi K2.5 structural
    parameters (different rms_norm_eps, rope_theta).
    """
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3RotaryEmbedding,
    )

    nope_dim = 8
    rope_dim = 4
    v_head_dim = 8
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=nope_dim,
        v_head_dim=v_head_dim,
        intermediate_size=128,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        attention_bias=False,
        rms_norm_eps=1e-5,
        rope_theta=50000.0,
        max_position_embeddings=256,
        attn_implementation="eager",
    )

    seed_all(77)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = DeepseekV3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)
    hf_attn_out, _ = layer.self_attn(
        normed, position_embeddings=(cos, sin), attention_mask=None,
    )

    attn = layer.self_attn

    mla_kwargs = dict(
        wq=attn.q_proj.weight.data,
        w_kv_a=attn.kv_a_proj_with_mqa.weight.data,
        kv_a_layernorm_weight=attn.kv_a_layernorm.weight.data,
        w_kv_b=attn.kv_b_proj.weight.data,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        kv_latent_dim=config.kv_lora_rank,
        position_embeddings=(cos, sin),
        qk_rope_head_dim=rope_dim,
    )

    our_torch_out, _ = mla_attention_torch(normed, **mla_kwargs)
    our_tt_out, _ = mla_attention_tt(normed, **mla_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_torch vs HF Kimi K2.5 attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_tt vs HF Kimi K2.5 attention mismatch"
