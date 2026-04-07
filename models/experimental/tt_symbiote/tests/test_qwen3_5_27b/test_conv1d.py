# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Depthwise Conv1d sanity tests for Qwen3.5-27B (PyTorch only, no TTNN).

The Conv1d is inside GatedDeltaNet (layer 0, linear attention).
It is a depthwise causal convolution over the QKV projection.
"""

import torch

from .conftest import (
    skip_no_transformers,
)


# ──────────────────────────────────────────────────────────────────────
# Test: Conv1d output shape
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_conv1d_output_shape(model_1_layer):
    """Verify Conv1d output shape after causal convolution.

    The conv1d operates on (batch, conv_dim, seq_len) and should produce
    the same shape output (causal padding preserves length).
    """
    model, config = model_1_layer
    gated_deltanet = model.model.layers[0].linear_attn
    conv1d = gated_deltanet.conv1d

    # conv_dim = key_dim * 2 + value_dim
    conv_dim = gated_deltanet.conv_dim
    batch_size, seq_len = 1, 32

    # Conv1d expects (batch, channels, length)
    x = torch.randn(batch_size, conv_dim, seq_len, dtype=torch.bfloat16)

    with torch.no_grad():
        out = conv1d(x)

    # Causal conv: output length = seq_len + kernel_size - 1 (with padding=kernel_size-1)
    # We take only [:, :, :seq_len] for causal output
    expected_len = seq_len + gated_deltanet.conv_kernel_size - 1
    assert out.shape == (
        batch_size,
        conv_dim,
        expected_len,
    ), f"Expected shape ({batch_size}, {conv_dim}, {expected_len}), got {out.shape}"
    # After truncation to causal:
    causal_out = out[:, :, :seq_len]
    assert causal_out.shape == (batch_size, conv_dim, seq_len), f"Causal output shape mismatch: {causal_out.shape}"
    print(f"  Conv1d output shape: {out.shape}, causal truncated: {causal_out.shape}")


# ──────────────────────────────────────────────────────────────────────
# Test: Conv1d causal property
# ──────────────────────────────────────────────────────────────────────


@skip_no_transformers
def test_conv1d_causal_property(model_1_layer):
    """Verify output at position t depends only on inputs [0..t].

    If we change input at position t+1, the output at position t should
    remain unchanged (causal property).
    """
    model, config = model_1_layer
    gated_deltanet = model.model.layers[0].linear_attn
    conv1d = gated_deltanet.conv1d

    conv_dim = gated_deltanet.conv_dim
    batch_size, seq_len = 1, 32
    t = 15  # test position

    x1 = torch.randn(batch_size, conv_dim, seq_len, dtype=torch.bfloat16)
    x2 = x1.clone()
    # Modify positions after t
    x2[:, :, t + 1 :] = torch.randn_like(x2[:, :, t + 1 :])

    with torch.no_grad():
        out1 = conv1d(x1)[:, :, :seq_len]
        out2 = conv1d(x2)[:, :, :seq_len]

    # Output at position t and before should be identical
    assert torch.allclose(
        out1[:, :, : t + 1], out2[:, :, : t + 1], atol=1e-6
    ), "Causal property violated: output at position <= t changed when future inputs changed"
    print(f"  Conv1d causal property verified at position t={t}")
