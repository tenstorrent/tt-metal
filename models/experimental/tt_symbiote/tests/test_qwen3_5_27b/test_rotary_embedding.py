# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding generation sanity tests for Qwen3.5-27B (PyTorch only, no TTNN).

Tests the Qwen3_5TextRotaryEmbedding module that generates cos/sin
position embeddings for the attention layers.
"""

import torch

from .conftest import (
    get_config_attr,
    skip_no_transformers,
)


# ──────────────────────────────────────────────────────────────────────
# Test: Rotary embedding output shapes
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_rotary_embedding_output_shape(model_4_layers):
    """Verify cos/sin shapes from rotary embedding.

    Qwen3_5TextRotaryEmbedding.forward(x, position_ids) returns (cos, sin)
    with shape matching the position encoding dimensions.
    """
    model, config = model_4_layers
    rotary_emb = model.model.rotary_emb

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")

    # Create a dummy input tensor (rotary_emb uses it only for device/dtype)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Position IDs: standard sequential positions
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        cos, sin = rotary_emb(x, position_ids)

    # cos and sin should have compatible shapes
    assert cos.shape == sin.shape, f"cos shape {cos.shape} != sin shape {sin.shape}"
    # Should contain the sequence length dimension
    assert cos.shape[-2] == seq_len, f"Expected seq_len={seq_len} in dim -2, got {cos.shape}"
    print(f"  Rotary embedding cos shape: {cos.shape}, sin shape: {sin.shape}")


# ──────────────────────────────────────────────────────────────────────
# Test: Rotary embedding values bounded in [-1, 1]
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_rotary_embedding_values_bounded(model_4_layers):
    """Verify cos/sin values are bounded in [-1, 1]."""
    model, config = model_4_layers
    rotary_emb = model.model.rotary_emb

    batch_size, seq_len = 1, 128
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        cos, sin = rotary_emb(x, position_ids)

    cos_float = cos.float()
    sin_float = sin.float()

    # Allow small tolerance for bfloat16 precision
    tolerance = 0.01
    assert cos_float.min() >= -1.0 - tolerance, f"cos min {cos_float.min()} < -1.0"
    assert cos_float.max() <= 1.0 + tolerance, f"cos max {cos_float.max()} > 1.0"
    assert sin_float.min() >= -1.0 - tolerance, f"sin min {sin_float.min()} < -1.0"
    assert sin_float.max() <= 1.0 + tolerance, f"sin max {sin_float.max()} > 1.0"
    print(f"  cos range: [{cos_float.min():.4f}, {cos_float.max():.4f}]")
    print(f"  sin range: [{sin_float.min():.4f}, {sin_float.max():.4f}]")


# ──────────────────────────────────────────────────────────────────────
# Test: Different positions produce different embeddings
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_rotary_embedding_different_positions(model_4_layers):
    """Verify that different positions produce different cos/sin embeddings."""
    model, config = model_4_layers
    rotary_emb = model.model.rotary_emb

    batch_size, seq_len = 1, 64
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        cos, sin = rotary_emb(x, position_ids)

    # Check that position 0 and position 1 produce different embeddings
    # (squeeze batch dim if present)
    cos_squeezed = cos.squeeze(0) if cos.dim() > 2 else cos
    assert not torch.allclose(
        cos_squeezed[0], cos_squeezed[1], atol=1e-6
    ), "Position 0 and 1 produced identical cos embeddings"

    sin_squeezed = sin.squeeze(0) if sin.dim() > 2 else sin
    assert not torch.allclose(
        sin_squeezed[0], sin_squeezed[1], atol=1e-6
    ), "Position 0 and 1 produced identical sin embeddings"
    print("  Different positions produce different embeddings: verified")
