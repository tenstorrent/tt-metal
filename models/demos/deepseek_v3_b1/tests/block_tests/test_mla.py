"""MLA unit tests + comparison vs HF attention."""

import torch

from helpers import seed_all, assert_close, make_profiles, rms_norm
from mla import mla_attention_torch, mla_attention_tt


def _make_mla_weights(profile, dtype=torch.float32):
    h = profile["hidden_size"]
    nq = profile["num_q_heads"]
    nkv = profile["num_kv_heads"]
    hd = profile["head_dim"]
    lat = profile["kv_latent_dim"]
    return dict(
        wq=torch.randn(nq * hd, h, dtype=dtype),
        w_kv_down=torch.randn(lat, h, dtype=dtype),
        w_k_up=torch.randn(nkv * hd, lat, dtype=dtype),
        w_v_up=torch.randn(nkv * hd, lat, dtype=dtype),
        wo=torch.randn(h, nq * hd, dtype=dtype),
    )


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
    )

    causal_full = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    out_full, _ = mla_attention_torch(x, **weights, **common, attention_mask=causal_full)

    x1 = x[:, :split, :]
    causal1 = torch.tril(torch.ones(split, split, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    out1, cache = mla_attention_torch(
        x1, **weights, **common, attention_mask=causal1, use_cache=True,
    )

    x2 = x[:, split:, :]
    kv_len = split + (seq - split)
    causal2 = torch.ones(seq - split, kv_len, dtype=torch.bool)
    for i in range(seq - split):
        causal2[i, split + i + 1:] = False
    causal2 = causal2.unsqueeze(0).unsqueeze(0)
    out2, _ = mla_attention_torch(
        x2, **weights, **common, attention_mask=causal2, past_key_value=cache, use_cache=True,
    )

    out_chunked = torch.cat([out1, out2], dim=1)
    assert_close(out_full, out_chunked, atol=1e-3, rtol=1e-3)


@torch.no_grad()
def test_mla_deepseek_v3():
    """
    Compare mla_attention_torch and mla_attention_tt from mla.py against the
    HF DeepseekV3Attention module.

    Adaptations to make architectures compatible:
    - q_lora_rank=None → direct Q projection (no LoRA), matches our wq
    - qk_rope_head_dim=0 → no RoPE dims, Q/K only have nope component
    - qk_nope_head_dim = v_head_dim → uniform head_dim for K and V
    - kv_a_layernorm replaced with nn.Identity → no norm in KV latent path
    - kv_b_proj weight split into separate w_k_up and w_v_up
    """
    import torch.nn as nn
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
    )

    head_dim = 8
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=0,
        qk_nope_head_dim=head_dim,
        v_head_dim=head_dim,
        intermediate_size=128,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer.eval()
    layer.self_attn.kv_a_layernorm = nn.Identity()

    cos = torch.empty(b, seq_len, 0)
    sin = torch.empty(b, seq_len, 0)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)

    hf_attn_out, _ = layer.self_attn(
        normed, position_embeddings=(cos, sin), attention_mask=None,
    )

    attn = layer.self_attn
    kv_b_w = attn.kv_b_proj.weight.data
    num_heads = config.num_attention_heads
    kv_b_reshaped = kv_b_w.view(num_heads, head_dim + head_dim, config.kv_lora_rank)
    w_k_up = kv_b_reshaped[:, :head_dim, :].reshape(num_heads * head_dim, config.kv_lora_rank)
    w_v_up = kv_b_reshaped[:, head_dim:, :].reshape(num_heads * head_dim, config.kv_lora_rank)

    mla_kwargs = dict(
        wq=attn.q_proj.weight.data,
        w_kv_down=attn.kv_a_proj_with_mqa.weight.data,
        w_k_up=w_k_up,
        w_v_up=w_v_up,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        kv_latent_dim=config.kv_lora_rank,
    )

    our_torch_out, _ = mla_attention_torch(normed, **mla_kwargs)
    our_tt_out, _ = mla_attention_tt(normed, **mla_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_torch vs HF DeepSeek V3 attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_tt vs HF DeepSeek V3 attention mismatch"


@torch.no_grad()
def test_mla_kimi_k25():
    """
    Same as test_mla_deepseek_v3 but with Kimi K2.5 structural
    parameters (different rms_norm_eps, rope_theta).
    """
    import torch.nn as nn
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
    )

    head_dim = 8
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=0,
        qk_nope_head_dim=head_dim,
        v_head_dim=head_dim,
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
    layer.self_attn.kv_a_layernorm = nn.Identity()

    cos = torch.empty(b, seq_len, 0)
    sin = torch.empty(b, seq_len, 0)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)
    hf_attn_out, _ = layer.self_attn(
        normed, position_embeddings=(cos, sin), attention_mask=None,
    )

    attn = layer.self_attn
    kv_b_w = attn.kv_b_proj.weight.data
    num_heads = config.num_attention_heads
    kv_b_reshaped = kv_b_w.view(num_heads, head_dim + head_dim, config.kv_lora_rank)
    w_k_up = kv_b_reshaped[:, :head_dim, :].reshape(num_heads * head_dim, config.kv_lora_rank)
    w_v_up = kv_b_reshaped[:, head_dim:, :].reshape(num_heads * head_dim, config.kv_lora_rank)

    mla_kwargs = dict(
        wq=attn.q_proj.weight.data,
        w_kv_down=attn.kv_a_proj_with_mqa.weight.data,
        w_k_up=w_k_up,
        w_v_up=w_v_up,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        kv_latent_dim=config.kv_lora_rank,
    )

    our_torch_out, _ = mla_attention_torch(normed, **mla_kwargs)
    our_tt_out, _ = mla_attention_tt(normed, **mla_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_torch vs HF Kimi K2.5 attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "mla_attention_tt vs HF Kimi K2.5 attention mismatch"
