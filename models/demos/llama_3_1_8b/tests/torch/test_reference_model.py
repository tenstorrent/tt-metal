# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1/M1: CPU reference model self-consistency and golden cache round-trip."""

import pytest
import torch

from models.demos.llama_3_1_8b.config import Llama31Config
from models.demos.llama_3_1_8b.reference.model import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)


@pytest.fixture
def small_config():
    return Llama31Config(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=64,
        vocab_size=1024,
    )


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


def test_rmsnorm_bf16(small_config, seed):
    norm = LlamaRMSNorm(small_config.hidden_size).to(torch.bfloat16)
    x = torch.randn(1, 32, small_config.hidden_size, dtype=torch.bfloat16)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16
    var = out.float().pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=0.1)


def test_mlp_bf16(small_config, seed):
    mlp = LlamaMLP(small_config).to(torch.bfloat16)
    x = torch.randn(1, 32, small_config.hidden_size, dtype=torch.bfloat16)
    out = mlp(x)
    assert out.shape == x.shape


def test_attention_returns_kv(small_config, seed):
    attn = LlamaAttention(small_config, layer_idx=0).to(torch.bfloat16)
    x = torch.randn(1, 32, small_config.hidden_size, dtype=torch.bfloat16)
    rope = LlamaRotaryEmbedding(small_config)
    pos_ids = torch.arange(32).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    out, k, v = attn(x, (cos, sin))
    assert out.shape == x.shape
    assert k.shape == (1, small_config.num_key_value_heads, 32, small_config.head_dim)
    assert v.shape == k.shape


def test_decoder_layer_residual(small_config, seed):
    layer = LlamaDecoderLayer(small_config, layer_idx=0).to(torch.bfloat16)
    x = torch.randn(1, 32, small_config.hidden_size, dtype=torch.bfloat16)
    rope = LlamaRotaryEmbedding(small_config)
    pos_ids = torch.arange(32).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    out, k, v = layer(x, (cos, sin))
    assert out.shape == x.shape
    assert not torch.allclose(out, x, atol=1e-3)


def test_full_model_forward(small_config, seed):
    model = LlamaForCausalLM(small_config).to(torch.bfloat16)
    tokens = torch.randint(0, small_config.vocab_size, (1, 32))
    logits, kv_cache = model(tokens)
    assert logits.shape == (1, 32, small_config.vocab_size)
    assert len(kv_cache) == small_config.num_hidden_layers
    for layer_idx in range(small_config.num_hidden_layers):
        assert "k" in kv_cache[layer_idx]
        assert "v" in kv_cache[layer_idx]
        assert kv_cache[layer_idx]["k"].shape == (1, small_config.num_key_value_heads, 32, small_config.head_dim)


def test_model_deterministic(small_config, seed):
    model = LlamaForCausalLM(small_config).to(torch.bfloat16)
    tokens = torch.randint(0, small_config.vocab_size, (1, 16))
    logits1, kv1 = model(tokens)
    logits2, kv2 = model(tokens)
    assert torch.equal(logits1, logits2)
    for L in range(small_config.num_hidden_layers):
        assert torch.equal(kv1[L]["k"], kv2[L]["k"])
        assert torch.equal(kv1[L]["v"], kv2[L]["v"])


def test_rope_hf_convention(small_config, seed):
    rope = LlamaRotaryEmbedding(small_config)
    x = torch.randn(1, 16, small_config.hidden_size, dtype=torch.bfloat16)
    pos_ids = torch.arange(16).unsqueeze(0)
    cos, sin = rope(x, pos_ids)
    assert cos.shape == (1, 16, small_config.head_dim)
    assert sin.shape == (1, 16, small_config.head_dim)
    # Position 0 should have cos=1 for all freqs (cos(0)=1)
    assert torch.allclose(
        cos[0, 0, : small_config.head_dim // 2], torch.ones(small_config.head_dim // 2, dtype=cos.dtype), atol=1e-2
    )


def test_repeat_kv():
    x = torch.randn(1, 2, 8, 64)
    out = repeat_kv(x, 4)
    assert out.shape == (1, 8, 8, 64)
    # Each group of 4 output heads should be identical
    for g in range(2):
        for r in range(4):
            assert torch.equal(out[0, g * 4 + r], x[0, g])


def test_golden_cache_round_trip(small_config, seed, tmp_path):
    import os

    os.environ["TT_LLAMA31_8B_REF_CACHE"] = str(tmp_path)
    from models.demos.llama_3_1_8b.reference.cache import (
        ReferenceCacheKey,
        check_reference_cache_exists,
        load_reference_cache,
        save_reference_cache,
    )

    key = ReferenceCacheKey(
        model_name="llama_3_1_8b",
        hidden_size=small_config.hidden_size,
        num_layers=small_config.num_hidden_layers,
        num_kv_heads=small_config.num_key_value_heads,
        head_dim=small_config.head_dim,
        seq_len=16,
        dtype="bfloat16",
        seed=42,
    )
    model = LlamaForCausalLM(small_config).to(torch.bfloat16)
    tokens = torch.randint(0, small_config.vocab_size, (1, 16))
    logits, kv = model(tokens)

    assert not check_reference_cache_exists(key)
    save_reference_cache(key, kv, logits)
    assert check_reference_cache_exists(key)

    loaded_kv, loaded_logits = load_reference_cache(key)
    for L in range(small_config.num_hidden_layers):
        assert torch.equal(loaded_kv[L]["k"], kv[L]["k"])
        assert torch.equal(loaded_kv[L]["v"], kv[L]["v"])
    assert torch.equal(loaded_logits, logits)

    # Changed key should miss
    key2 = ReferenceCacheKey(
        model_name="llama_3_1_8b",
        hidden_size=small_config.hidden_size,
        num_layers=small_config.num_hidden_layers,
        num_kv_heads=small_config.num_key_value_heads,
        head_dim=small_config.head_dim,
        seq_len=32,
        dtype="bfloat16",
        seed=42,
    )
    assert not check_reference_cache_exists(key2)
