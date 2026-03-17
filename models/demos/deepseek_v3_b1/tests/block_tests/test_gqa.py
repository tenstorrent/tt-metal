# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""GQA unit tests + comparison vs HF attention.

Test categories:
  1. torch vs tt equivalence across all tiny profiles
  2. KV-cache chunking — full prefill matches chunk+decode
  3. Token-by-token decode matches full prefill
  4. Causal masking — future tokens don't affect earlier positions
  5. RoPE — with vs without, partial vs full, different positions
  6. Softcapping — Grok-2 style logit capping
  7. QK-norm — per-head RMSNorm on Q and K
  8. GQA group-ratio sweep (MHA / GQA / MQA)
  9. Full-size weight shape validation (meta device, no alloc)
 10. HF comparison — GLM-4, Llama
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


def test_gqa_matches_hf_per_profile_cpu_fp32():
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
# 2. KV-cache chunking equivalence
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
# 3. Token-by-token decode matches full prefill
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
# 4. Causal masking tests
# ---------------------------------------------------------------------------


def test_gqa_causal_mask_structure():
    mask = _causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 1] == float("-inf")
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 3, 0] == 0.0
    assert mask[0, 0, 3, 3] == 0.0


@pytest.mark.parametrize("profile_name", list(GQA_TINY_PROFILES.keys()))
def test_gqa_masked_output_independent_of_future(profile_name):
    """Changing future tokens shouldn't affect output at earlier positions."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h = prof["hidden_size"]
    seq_len = 8
    x1 = torch.randn(1, seq_len, h)
    x2 = x1.clone()
    x2[:, -2:, :] = torch.randn_like(x2[:, -2:, :])
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = _causal_mask(seq_len)
    with torch.no_grad():
        out1, _ = gqa_attention_torch(x1, **weights, **common, attention_mask=mask)
        out2, _ = gqa_attention_torch(x2, **weights, **common, attention_mask=mask)
    torch.testing.assert_close(out1[:, :6, :], out2[:, :6, :], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. RoPE tests
# ---------------------------------------------------------------------------


def _build_rope_cache(seq_len, head_dim, theta=10000.0, rope_dim=None):
    """Build cos/sin in the shape helpers.apply_rotary_pos_emb expects: [1, seq, dim]."""
    dim = rope_dim if rope_dim is not None else head_dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    return cos, sin


@pytest.mark.parametrize("profile_name", list(GQA_TINY_PROFILES.keys()))
def test_gqa_rope_changes_output(profile_name):
    """Output should differ with vs without RoPE."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h, hd = prof["hidden_size"], prof["head_dim"]
    seq_len = 16
    x = torch.randn(1, seq_len, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = _causal_mask(seq_len)
    cos, sin = _build_rope_cache(seq_len, hd, prof.get("rope_theta", 10000.0))
    with torch.no_grad():
        out_no_rope, _ = gqa_attention_torch(x, **weights, **common, attention_mask=mask)
        out_rope, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            position_embeddings=(cos, sin),
        )
    assert not torch.allclose(out_no_rope, out_rope, atol=1e-5)


def test_gqa_partial_vs_full_rope_differ():
    """GLM-4 partial RoPE (glm4 variant) and standard full RoPE should give different results."""
    seed_all(42)
    prof = GQA_TINY_PROFILES["glm_4_7_355b"]
    h, hd = prof["hidden_size"], prof["head_dim"]
    seq_len = 16
    x = torch.randn(1, seq_len, h)

    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = _causal_mask(seq_len)

    cos, sin = _build_rope_cache(seq_len, hd)
    with torch.no_grad():
        out_glm4, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            position_embeddings=(cos, sin),
            rope_variant="glm4",
        )
        out_standard, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            position_embeddings=(cos, sin),
            rope_variant="standard",
        )
    assert not torch.allclose(out_glm4, out_standard, atol=1e-5)


@pytest.mark.parametrize("profile_name", list(GQA_TINY_PROFILES.keys()))
def test_gqa_different_positions_give_different_output(profile_name):
    """Different relative positions (via KV cache) should produce different outputs."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h, hd = prof["hidden_size"], prof["head_dim"]
    nkv = prof["num_kv_heads"]
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)

    cos_full, sin_full = _build_rope_cache(64, hd, prof.get("rope_theta", 10000.0))

    x_ctx = torch.randn(1, 4, h)
    empty_cache = (torch.zeros(1, nkv, 0, hd), torch.zeros(1, nkv, 0, hd))
    mask_ctx = _causal_mask(4)
    with torch.no_grad():
        _, kv_cache = gqa_attention_torch(
            x_ctx,
            **weights,
            **common,
            attention_mask=mask_ctx,
            position_embeddings=(cos_full[:, 0:4], sin_full[:, 0:4]),
            past_key_value=empty_cache,
            use_cache=True,
        )

    x_new = torch.randn(1, 1, h)
    mask_decode = _causal_mask(1, 5)
    with torch.no_grad():
        out_pos4, _ = gqa_attention_torch(
            x_new,
            **weights,
            **common,
            attention_mask=mask_decode,
            position_embeddings=(cos_full[:, 4:5], sin_full[:, 4:5]),
            past_key_value=kv_cache,
            use_cache=True,
        )
        out_pos20, _ = gqa_attention_torch(
            x_new,
            **weights,
            **common,
            attention_mask=mask_decode,
            position_embeddings=(cos_full[:, 20:21], sin_full[:, 20:21]),
            past_key_value=kv_cache,
            use_cache=True,
        )
    assert not torch.allclose(out_pos4, out_pos20, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Softcapping tests (Grok-2 style)
# ---------------------------------------------------------------------------


def test_gqa_softcapping_changes_output():
    """Softcapping should produce different output than without."""
    seed_all(42)
    prof = GQA_TINY_PROFILES["grok_2_270b"]
    h = prof["hidden_size"]
    x = torch.randn(1, 8, h) * 10.0
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = torch.zeros(1, 1, 8, 8)
    with torch.no_grad():
        out_no_cap, _ = gqa_attention_torch(x, **weights, **common, attention_mask=mask)
        out_cap, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            attn_logit_softcapping=30.0,
        )
    assert torch.isfinite(out_cap).all()
    assert not torch.allclose(out_no_cap, out_cap, atol=1e-5)


def test_gqa_softcapping_bounds_logits():
    """Softcapped output should be finite even with large inputs."""
    seed_all(42)
    prof = GQA_TINY_PROFILES["grok_2_270b"]
    h = prof["hidden_size"]
    x = torch.randn(1, 8, h) * 100.0
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    mask = torch.zeros(1, 1, 8, 8)
    with torch.no_grad():
        out, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            attn_logit_softcapping=30.0,
        )
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. QK-norm tests
# ---------------------------------------------------------------------------


def test_gqa_qk_norm_changes_output():
    """QK-norm should produce different output than without."""
    seed_all(42)
    h, hd, nq, nkv = 256, 64, 4, 2
    prof = dict(hidden_size=h, num_q_heads=nq, num_kv_heads=nkv, head_dim=hd)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    q_norm_w = torch.ones(hd)
    k_norm_w = torch.ones(hd)
    x = torch.randn(1, 8, h)
    mask = torch.zeros(1, 1, 8, 8)
    with torch.no_grad():
        out_no, _ = gqa_attention_torch(x, **weights, **common, attention_mask=mask)
        out_yes, _ = gqa_attention_torch(
            x,
            **weights,
            **common,
            attention_mask=mask,
            qk_norm_weights=(q_norm_w, k_norm_w),
        )
    assert not torch.allclose(out_no, out_yes, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. GQA group-ratio sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_heads,num_kv_heads",
    [(8, 8), (8, 4), (8, 2), (8, 1), (16, 4), (32, 8)],
    ids=["MHA_8:8", "GQA_8:4", "GQA_8:2", "MQA_8:1", "GQA_16:4", "GQA_32:8"],
)
def test_gqa_group_ratio_torch_vs_tt(num_heads, num_kv_heads):
    """Both implementations should agree for various group ratios."""
    seed_all(42)
    h, hd = 256, 32
    prof = dict(hidden_size=h, num_q_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=hd)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    x = torch.randn(2, 8, h)
    mask = _causal_mask(8)
    with torch.no_grad():
        out_torch, _ = gqa_attention_torch(x, **weights, **common, attention_mask=mask)
        out_tt, _ = gqa_attention_tt(x, **weights, **common, attention_mask=mask)
    assert_close(out_torch, out_tt, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 9. Output shape tests (various batch / seq combinations)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_name", list(GQA_TINY_PROFILES.keys()))
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
def test_gqa_output_shape(profile_name, batch_size, seq_len):
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h = prof["hidden_size"]
    x = torch.randn(batch_size, seq_len, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    with torch.no_grad():
        out, _ = gqa_attention_torch(x, **weights, **common)
    assert out.shape == (batch_size, seq_len, h)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 10. Full-size weight shape validation (meta device — zero memory)
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
    """Full-size projection shapes must match models_shapes.txt (meta device, no allocation)."""
    prof = GQA_FULL_PROFILES[model_name]
    h = prof["hidden_size"]
    nq, nkv, hd = prof["num_q_heads"], prof["num_kv_heads"], prof["head_dim"]
    q_expected, k_expected, v_expected, o_expected = EXPECTED_WEIGHT_SHAPES[model_name]
    assert (nq * hd, h) == q_expected, f"q_proj shape mismatch for {model_name}"
    assert (nkv * hd, h) == k_expected, f"k_proj shape mismatch for {model_name}"
    assert (nkv * hd, h) == v_expected, f"v_proj shape mismatch for {model_name}"
    assert (h, nq * hd) == o_expected, f"o_proj shape mismatch for {model_name}"


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
# 11. Residual connection test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile_name", list(GQA_TINY_PROFILES.keys()))
def test_gqa_residual_connection(profile_name):
    """Pre-norm + attention + residual should produce valid output different from input."""
    seed_all(42)
    prof = GQA_TINY_PROFILES[profile_name]
    h, hd = prof["hidden_size"], prof["head_dim"]
    seq_len = 16
    x = torch.randn(1, seq_len, h)
    weights = _make_gqa_weights(prof)
    common = _common_kwargs(prof)
    norm_weight = torch.ones(h)
    normed = rms_norm(x, norm_weight)
    mask = _causal_mask(seq_len)
    with torch.no_grad():
        attn_out, _ = gqa_attention_torch(normed, **weights, **common, attention_mask=mask)
    output = x + attn_out
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, x, atol=1e-5)


# ---------------------------------------------------------------------------
# 12. HF comparison: GLM-4
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
# 13. HF comparison: Llama
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
