# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""GQA functional tests + comparison vs HF attention.

Test categories:
  1. torch vs tt equivalence across profiles
  2. KV-cache: full prefill matches chunk+decode
  3. KV-cache: token-by-token decode matches full prefill
  4. QK-Norm: with vs without produces different output
  5. Full-size weight shape validation
  6. Full-size single-token smoke test
  7. HF comparison: GLM-4 (with QK-Norm + partial RoPE)
  8. HF comparison: Llama (standard RoPE)
"""

import pytest
import torch
from gqa import gqa_attention_torch, gqa_attention_tt
from helpers import GQA_FULL_PROFILES, GQA_TINY_PROFILES, assert_close, make_profiles, rms_norm, seed_all

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _causal_mask(seq_len, kv_len=None, dtype=torch.float32):
    """Additive causal mask: 0 for allowed, -inf for masked."""
    kv_len = kv_len or seq_len
    mask = torch.full((seq_len, kv_len), float("-inf"), dtype=dtype)
    mask = torch.triu(mask, diagonal=kv_len - seq_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)


def _common_kwargs(prof):
    return dict(num_q_heads=prof["num_q_heads"], num_kv_heads=prof["num_kv_heads"])


# ---------------------------------------------------------------------------
# 1. torch vs tt equivalence across all tiny profiles
# ---------------------------------------------------------------------------


def test_gqa_torch_vs_tt_per_profile():
    seed_all(42)
    profiles = make_profiles("tiny")
    b, seq = 2, 6
    for name, prof in profiles.items():
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_gqa_weights(prof)
        common = _common_kwargs(prof)
        out_torch, _ = gqa_attention_torch(x, **weights, **common)
        out_tt, _ = gqa_attention_tt(x, **weights, **common)
        assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. KV-cache: full prefill matches chunk+decode
# ---------------------------------------------------------------------------


def test_gqa_cache_chunking_equivalence():
    """Full-seq vs chunk1+chunk2 with cache must give same final output."""
    seed_all(123)
    prof = make_profiles("tiny")["glm4_355b"]
    h = prof["hidden_size"]
    b, seq = 1, 8
    split = 4
    x = torch.randn(b, seq, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)

    causal_full = (
        torch.where(
            torch.tril(torch.ones(seq, seq)).bool(),
            0.0,
            float("-inf"),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    out_full, _ = gqa_attention_torch(x, **weights, **common, attention_mask=causal_full)

    x1 = x[:, :split, :]
    causal1 = (
        torch.where(
            torch.tril(torch.ones(split, split)).bool(),
            0.0,
            float("-inf"),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    out1, cache = gqa_attention_torch(
        x1,
        **weights,
        **common,
        attention_mask=causal1,
        use_cache=True,
    )

    x2 = x[:, split:, :]
    kv_len = split + (seq - split)
    causal2_bool = torch.ones(seq - split, kv_len, dtype=torch.bool)
    for i in range(seq - split):
        causal2_bool[i, split + i + 1 :] = False
    causal2 = torch.where(causal2_bool, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
    out2, _ = gqa_attention_torch(
        x2,
        **weights,
        **common,
        attention_mask=causal2,
        past_key_value=cache,
        use_cache=True,
    )

    out_chunked = torch.cat([out1, out2], dim=1)
    assert_close(out_full, out_chunked, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. KV-cache: token-by-token decode matches full prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_name", list(GQA_TINY_PROFILES.keys()))
def test_gqa_decode_matches_prefill(profile_name):
    """Incremental token-by-token decode must produce same output as full prefill."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h, hd = prof["hidden_size"], prof["head_dim"]
    nkv = prof["num_kv_heads"]
    b, seq_len = 1, 8
    x = torch.randn(b, seq_len, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)

    mask_full = _causal_mask(seq_len)
    with torch.no_grad():
        out_prefill, _ = gqa_attention_torch(x, **weights, **common, attention_mask=mask_full)

    kv_cache = (
        torch.zeros(b, nkv, 0, hd),
        torch.zeros(b, nkv, 0, hd),
    )
    decode_outputs = []
    with torch.no_grad():
        for t in range(seq_len):
            x_t = x[:, t : t + 1, :]
            mask_t = _causal_mask(1, t + 1)
            out_t, kv_cache = gqa_attention_torch(
                x_t,
                **weights,
                **common,
                attention_mask=mask_t,
                past_key_value=kv_cache,
                use_cache=True,
            )
            decode_outputs.append(out_t)

    out_decode = torch.cat(decode_outputs, dim=1)
    assert_close(out_prefill, out_decode, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. QK-Norm: with vs without produces different output
# ---------------------------------------------------------------------------

QK_NORM_MODELS = [k for k, v in GQA_TINY_PROFILES.items() if v.get("qk_norm")]
NO_QK_NORM_MODELS = [k for k, v in GQA_TINY_PROFILES.items() if not v.get("qk_norm")]


@pytest.mark.parametrize("profile_name", QK_NORM_MODELS)
def test_gqa_qk_norm_changes_output(profile_name):
    """QK-Norm (per-head RMSNorm on Q/K) should change the attention output."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h, hd = prof["hidden_size"], prof["head_dim"]
    x = torch.randn(1, 8, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = torch.zeros(1, 1, 8, 8)

    q_norm_w = torch.ones(hd)
    k_norm_w = torch.ones(hd)

    with torch.no_grad():
        out_no_norm, _ = gqa_attention_torch(x, **weights, **common, attention_mask=mask)
        out_with_norm, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            qk_norm_weights=(q_norm_w, k_norm_w),
        )
    assert not torch.allclose(out_no_norm, out_with_norm, atol=1e-5)


@pytest.mark.parametrize("profile_name", QK_NORM_MODELS)
def test_gqa_qk_norm_torch_vs_tt(profile_name):
    """torch and tt should agree when QK-Norm is enabled."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h, hd = prof["hidden_size"], prof["head_dim"]
    x = torch.randn(1, 8, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = _causal_mask(8)

    q_norm_w = torch.ones(hd)
    k_norm_w = torch.ones(hd)

    with torch.no_grad():
        out_torch, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            qk_norm_weights=(q_norm_w, k_norm_w),
        )
        out_tt, _ = gqa_attention_tt(
            x,
            **weights,
            **common,
            attention_mask=mask,
            qk_norm_weights=(q_norm_w, k_norm_w),
        )
    assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Full-size weight shape validation
# ---------------------------------------------------------------------------

EXPECTED_WEIGHT_SHAPES = {
    # (q_proj, k_proj, v_proj, o_proj) as PyTorch (out_features, in_features)
    "glm_4_7_355b": ((12288, 5120), (1024, 5120), (1024, 5120), (5120, 12288)),
    "gpt_oss_120b": ((4096, 2880), (512, 2880), (512, 2880), (2880, 4096)),
    "llama_guard_4": ((5120, 5120), (1024, 5120), (1024, 5120), (5120, 5120)),
    "qwen3_235b": ((8192, 4096), (512, 4096), (512, 4096), (4096, 8192)),
}


@pytest.mark.parametrize("model_name", list(EXPECTED_WEIGHT_SHAPES.keys()))
def test_gqa_full_size_weight_shapes(model_name):
    """Full-size projection shapes must match models_shapes.txt."""
    prof = GQA_FULL_PROFILES[model_name]
    h = prof["hidden_size"]
    nq, nkv, hd = prof["num_q_heads"], prof["num_kv_heads"], prof["head_dim"]
    q_expected, k_expected, v_expected, o_expected = EXPECTED_WEIGHT_SHAPES[model_name]
    assert (nq * hd, h) == q_expected, f"q_proj shape mismatch for {model_name}"
    assert (nkv * hd, h) == k_expected, f"k_proj shape mismatch for {model_name}"
    assert (nkv * hd, h) == v_expected, f"v_proj shape mismatch for {model_name}"
    assert (h, nq * hd) == o_expected, f"o_proj shape mismatch for {model_name}"


# ---------------------------------------------------------------------------
# 6. Full-size single-token smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", list(GQA_FULL_PROFILES.keys()))
def test_gqa_full_size_single_token_smoke(model_name):
    """Single-token forward pass with real model dimensions should be finite."""
    prof = GQA_FULL_PROFILES[model_name]
    h = prof["hidden_size"]
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    x = torch.randn(1, 1, h)
    with torch.no_grad():
        out, _ = gqa_attention_torch(x, **weights, **common)
    assert out.shape == (1, 1, h)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. HF comparison: GLM-4 (with QK-Norm + partial RoPE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_gqa_glm4():
    """
    Compare gqa_attention_torch and gqa_attention_tt from gqa.py against the
    HF Glm4Attention module with RoPE (GLM4 interleaved, partial).
    """
    from transformers import Glm4Config
    from transformers.models.glm4.modeling_glm4 import Glm4DecoderLayer, Glm4RotaryEmbedding

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
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)

    hf_attn_out, _ = layer.self_attn(
        normed,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )

    attn = layer.self_attn
    gqa_kwargs = dict(
        wq=attn.q_proj.weight.data,
        wk=attn.k_proj.weight.data,
        wv=attn.v_proj.weight.data,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        position_embeddings=(cos, sin),
        rope_variant="glm4",
    )

    our_torch_out, _ = gqa_attention_torch(normed, **gqa_kwargs)
    our_tt_out, _ = gqa_attention_tt(normed, **gqa_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), "gqa_attention_torch vs HF GLM-4 attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), "gqa_attention_tt vs HF GLM-4 attention mismatch"


# ---------------------------------------------------------------------------
# 8. HF comparison: Llama (standard RoPE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_gqa_llama():
    """
    Compare gqa_attention_torch against HF LlamaAttention
    with standard RoPE and position_ids.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = LlamaDecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = LlamaRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    normed = rms_norm(x, layer.input_layernorm.weight.data, config.rms_norm_eps)

    hf_attn_out, _ = layer.self_attn(
        normed,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )

    attn = layer.self_attn
    gqa_kwargs = dict(
        wq=attn.q_proj.weight.data,
        wk=attn.k_proj.weight.data,
        wv=attn.v_proj.weight.data,
        wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        position_embeddings=(cos, sin),
    )

    our_torch_out, _ = gqa_attention_torch(normed, **gqa_kwargs)
    our_tt_out, _ = gqa_attention_tt(normed, **gqa_kwargs)

    assert_close(hf_attn_out, our_torch_out, atol=1e-5, rtol=1e-5), "gqa_attention_torch vs HF Llama attention mismatch"
    assert_close(hf_attn_out, our_tt_out, atol=1e-5, rtol=1e-5), "gqa_attention_tt vs HF Llama attention mismatch"
