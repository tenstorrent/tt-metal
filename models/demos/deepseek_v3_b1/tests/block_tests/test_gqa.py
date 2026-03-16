"""GQA unit tests + comparison vs HF attention."""

import torch

from helpers import seed_all, assert_close, make_profiles, rms_norm
from gqa import gqa_attention_torch, gqa_attention_tt


def _make_gqa_weights(profile, dtype=torch.float32):
    h = profile["hidden_size"]
    nq = profile["num_q_heads"]
    nkv = profile["num_kv_heads"]
    hd = profile["head_dim"]
    return dict(
        wq=torch.randn(nq * hd, h, dtype=dtype),
        wk=torch.randn(nkv * hd, h, dtype=dtype),
        wv=torch.randn(nkv * hd, h, dtype=dtype),
        wo=torch.randn(h, nq * hd, dtype=dtype),
    )


def test_gqa_matches_hf_per_profile_cpu_fp32():
    seed_all(42)
    profiles = make_profiles("tiny")
    b, seq = 2, 6
    for name, prof in profiles.items():
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_gqa_weights(prof)
        common = dict(
            num_q_heads=prof["num_q_heads"],
            num_kv_heads=prof["num_kv_heads"],
        )
        out_torch, _ = gqa_attention_torch(x, **weights, **common)
        out_tt, _ = gqa_attention_tt(x, **weights, **common)
        assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_gqa_cache_chunking_equivalence():
    """Full-seq vs chunk1+chunk2 with cache must give same final output."""
    seed_all(123)
    prof = make_profiles("tiny")["glm4_355b"]
    h = prof["hidden_size"]
    b, seq = 1, 8
    split = 4
    x = torch.randn(b, seq, h)
    weights = _make_gqa_weights(prof)
    common = dict(
        num_q_heads=prof["num_q_heads"],
        num_kv_heads=prof["num_kv_heads"],
    )

    causal_full = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    out_full, _ = gqa_attention_torch(x, **weights, **common, attention_mask=causal_full)

    x1 = x[:, :split, :]
    causal1 = torch.tril(torch.ones(split, split, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    out1, cache = gqa_attention_torch(
        x1, **weights, **common, attention_mask=causal1, use_cache=True,
    )

    x2 = x[:, split:, :]
    kv_len = split + (seq - split)
    causal2 = torch.ones(seq - split, kv_len, dtype=torch.bool)
    for i in range(seq - split):
        causal2[i, split + i + 1:] = False
    causal2 = causal2.unsqueeze(0).unsqueeze(0)
    out2, _ = gqa_attention_torch(
        x2, **weights, **common, attention_mask=causal2, past_key_value=cache, use_cache=True,
    )

    out_chunked = torch.cat([out1, out2], dim=1)
    assert_close(out_full, out_chunked, atol=1e-3, rtol=1e-3)


@torch.no_grad()
def test_gqa_glm4():
    """
    Compare gqa_attention_torch and gqa_attention_tt from gqa.py against the
    HF Glm4Attention module.

    Adaptations:
    - attention_bias=False (our functions don't support bias)
    - position_ids=0 for all tokens → RoPE becomes identity (cos=1, sin=0)
    """
    from transformers import Glm4Config
    from transformers.models.glm4.modeling_glm4 import (
        Glm4DecoderLayer,
        Glm4RotaryEmbedding,
    )

    config = Glm4Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        partial_rotary_factor=0.5,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Glm4DecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = Glm4RotaryEmbedding(config)
    position_ids = torch.zeros(b, seq_len, dtype=torch.long)
    cos, sin = rotary(x, position_ids)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)

    hf_attn_out, _ = layer.self_attn(
        normed, position_embeddings=(cos, sin), attention_mask=None,
    )

    attn = layer.self_attn
    gqa_kwargs = dict(
        wq=attn.q_proj.weight.data,
        wk=attn.k_proj.weight.data,
        wv=attn.v_proj.weight.data,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )

    our_torch_out, _ = gqa_attention_torch(normed, **gqa_kwargs)
    our_tt_out, _ = gqa_attention_tt(normed, **gqa_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), \
        "gqa_attention_torch vs HF GLM-4 attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), \
        "gqa_attention_tt vs HF GLM-4 attention mismatch"
