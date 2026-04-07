# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Embedding sanity tests for Qwen3.5-27B (PyTorch only, no TTNN)."""

import torch

from .conftest import (
    get_config_attr,
    skip_no_transformers,
)


# ──────────────────────────────────────────────────────────────────────
# Test: Embedding output shape
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_embedding_output_shape(model_1_layer):
    """Verify embedding output shape is [B, T, hidden_size]."""
    model, config = model_1_layer
    embed = model.model.embed_tokens

    batch_size, seq_len = 1, 16
    hidden_size = get_config_attr(config, "hidden_size")
    # Use valid token IDs (within vocab range)
    vocab_size = get_config_attr(config, "vocab_size")
    input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))

    with torch.no_grad():
        out = embed(input_ids)

    assert out.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), got {out.shape}"
    print(f"  Embedding output shape: {out.shape}")


# ──────────────────────────────────────────────────────────────────────
# Test: Embedding output dtype
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_embedding_output_dtype(model_1_layer):
    """Verify embedding outputs bfloat16 (model is cast to bfloat16)."""
    model, config = model_1_layer
    embed = model.model.embed_tokens

    vocab_size = get_config_attr(config, "vocab_size")
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, 8))

    with torch.no_grad():
        out = embed(input_ids)

    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"
    print(f"  Embedding output dtype: {out.dtype}")


# ──────────────────────────────────────────────────────────────────────
# Test: Embedding no NaN
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_embedding_no_nan(model_1_layer):
    """Verify no NaN values for valid token IDs."""
    model, config = model_1_layer
    embed = model.model.embed_tokens

    vocab_size = get_config_attr(config, "vocab_size")
    input_ids = torch.randint(0, min(vocab_size, 1000), (1, 32))

    with torch.no_grad():
        out = embed(input_ids)

    assert not torch.isnan(out).any(), "NaN detected in embedding output"
    assert not torch.isinf(out).any(), "Inf detected in embedding output"
    print("  Embedding: no NaN/Inf detected")
