# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gated RMSNorm sanity tests for Qwen3.5-27B (PyTorch only, no TTNN).

Qwen3_5RMSNormGated lives inside the GatedDeltaNet layer (linear attention).
It applies RMSNorm then gates with SiLU(gate).
"""

import torch

from .conftest import (
    skip_no_transformers,
)


# ──────────────────────────────────────────────────────────────────────
# Test: Output shape preservation
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_rmsnorm_gated_output_shape(model_1_layer):
    """Verify Qwen3_5RMSNormGated preserves input shape."""
    model, config = model_1_layer
    # GatedDeltaNet is at layer 0 (linear attention layer)
    gated_deltanet = model.model.layers[0].linear_attn
    norm = gated_deltanet.norm

    # norm operates on head_v_dim sized inputs
    head_v_dim = gated_deltanet.head_v_dim
    batch_size, seq_len = 1, 32
    x = torch.randn(batch_size, seq_len, head_v_dim, dtype=torch.bfloat16)
    gate = torch.randn(batch_size, seq_len, head_v_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out = norm(x, gate=gate)

    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    print(f"  RMSNormGated output shape: {out.shape}")


# ──────────────────────────────────────────────────────────────────────
# Test: No NaN/Inf in output
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_rmsnorm_gated_no_nan(model_1_layer):
    """Verify no NaN or Inf in Qwen3_5RMSNormGated output."""
    model, config = model_1_layer
    gated_deltanet = model.model.layers[0].linear_attn
    norm = gated_deltanet.norm

    head_v_dim = gated_deltanet.head_v_dim
    batch_size, seq_len = 1, 32
    x = torch.randn(batch_size, seq_len, head_v_dim, dtype=torch.bfloat16)
    gate = torch.randn(batch_size, seq_len, head_v_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out = norm(x, gate=gate)

    assert not torch.isnan(out).any(), "NaN detected in RMSNormGated output"
    assert not torch.isinf(out).any(), "Inf detected in RMSNormGated output"
    print("  RMSNormGated: no NaN/Inf detected")
