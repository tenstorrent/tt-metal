# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for scaled_dot_product_attention (Flash Attention).

This is the immutable spec — the implementer must not modify this file.
It tests the operation against a PyTorch reference using PCC tolerances
keyed by dtype.

Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention.py
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from .reference import flash_attention_reference


# PCC tolerances keyed by dtype — same thresholds as the golden suite.
PCC_TOLERANCES = {
    ttnn.bfloat16: 0.995,
    ttnn.float32: 0.999,
    ttnn.bfloat8_b: 0.99,
}


def make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16):
    """Broadcast (B, 1, S_q, S_kv) additive causal mask: -inf above diagonal."""
    mask = torch.zeros(B, 1, S_q, S_kv, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask


# =============================================================================
# Shape parametrization
# =============================================================================

SHAPES = [
    # (label, Q_shape, K_shape, V_shape)
    # --- single tile ---
    ("single_tile", (1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    # --- multi-tile ---
    ("multi_tile_128x128", (1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128)),
    ("multi_tile_256x64", (1, 1, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),
    # --- non-square (Q dims differ) ---
    ("non_square", (1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    # --- multi-batch ---
    ("multi_batch", (2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
    # --- multi-head ---
    ("multi_head", (1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)),
    # --- long context ---
    ("long_context_512", (1, 4, 512, 64), (1, 4, 512, 64), (1, 4, 512, 64)),
    # --- cross-attention (S_q != S_kv) ---
    ("cross_attn", (1, 4, 64, 64), (1, 4, 128, 64), (1, 4, 128, 64)),
]

# Call patterns: (label, is_causal, use_mask, scale)
CALL_PATTERNS = [
    ("no_mask_auto_scale", False, False, None),
    ("no_mask_explicit_scale", False, False, 0.125),
    ("custom_mask_auto_scale", False, True, None),
    ("custom_mask_explicit_scale", False, True, 0.1),
]


@pytest.mark.parametrize(
    "label, q_shape, k_shape, v_shape",
    SHAPES,
    ids=[s[0] for s in SHAPES],
)
@pytest.mark.parametrize(
    "label_pattern, is_causal, use_mask, scale",
    CALL_PATTERNS,
    ids=[s[0] for s in CALL_PATTERNS],
)
def test_scaled_dot_product_attention(
    device, label, q_shape, k_shape, v_shape, label_pattern, is_causal, use_mask, scale
):
    """Test scaled_dot_product_attention against PyTorch reference."""
    torch.manual_seed(42)

    # Create torch tensors
    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    # Create mask if needed
    if use_mask:
        B, H, S_q, _D = q_shape
        S_kv = k_shape[-2]
        torch_mask = make_causal_mask(B, S_q, S_kv, dtype=torch.bfloat16)
    else:
        torch_mask = None

    # PyTorch reference (fp32 internally)
    expected = flash_attention_reference(
        torch_Q,
        torch_K,
        torch_V,
        attn_mask=torch_mask,
        is_causal=is_causal,
        scale=scale,
    )

    # Convert to TTNN
    ttnn_Q = ttnn.from_torch(
        torch_Q,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_K = ttnn.from_torch(
        torch_K,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_V = ttnn.from_torch(
        torch_V,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_mask = (
        ttnn.from_torch(
            torch_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if torch_mask is not None
        else None
    )

    # Run the operation
    ttnn_output = scaled_dot_product_attention(
        ttnn_Q,
        ttnn_K,
        ttnn_V,
        attn_mask=ttnn_mask,
        is_causal=is_causal,
        scale=scale,
    )

    # Convert back to torch
    torch_output = ttnn.to_torch(ttnn_output)

    # Check shape
    assert list(ttnn_output.shape) == list(
        q_shape
    ), f"Output shape {list(ttnn_output.shape)} != expected {list(q_shape)}"

    # Compare with reference using PCC
    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]

    output_f32 = torch_output.float()
    expected_f32 = expected.float()

    # Compute PCC (Pearson Correlation Coefficient)
    output_flat = output_f32.flatten()
    expected_flat = expected_f32.flatten()

    if output_flat.numel() > 1:
        output_centered = output_flat - output_flat.mean()
        expected_centered = expected_flat - expected_flat.mean()

        numerator = (output_centered * expected_centered).sum()
        denominator = torch.sqrt((output_centered**2).sum()) * torch.sqrt((expected_centered**2).sum())

        if denominator > 0:
            pcc = (numerator / denominator).item()
        else:
            pcc = 1.0 if torch.allclose(output_flat, expected_flat) else 0.0
    else:
        pcc = 1.0 if torch.allclose(output_flat, expected_flat, atol=1e-2) else 0.0

    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} < threshold {pcc_threshold} for {label}/{label_pattern}"


def test_scaled_dot_product_attention_basic(device):
    """Minimal smoke test — single tile, no mask, auto scale."""
    torch.manual_seed(42)

    Q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    K = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    V = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

    expected = flash_attention_reference(Q, K, V)

    ttnn_Q = ttnn.from_torch(
        Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)

    assert list(output.shape) == [1, 1, 32, 32]
    torch_output = ttnn.to_torch(output)
    assert torch_output.dtype == torch.bfloat16

    # Basic correctness check
    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    output_f32 = torch_output.float()
    expected_f32 = expected.float()

    output_flat = output_f32.flatten()
    expected_flat = expected_f32.flatten()
    output_centered = output_flat - output_flat.mean()
    expected_centered = expected_flat - expected_flat.mean()
    numerator = (output_centered * expected_centered).sum()
    denominator = torch.sqrt((output_centered**2).sum()) * torch.sqrt((expected_centered**2).sum())
    pcc = (numerator / denominator).item() if denominator > 0 else 1.0

    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} < threshold {pcc_threshold}"


def test_scaled_dot_product_attention_gqa(device):
    """GQA: H_q != H_kv (4:1 ratio)."""
    torch.manual_seed(42)

    q_shape = (1, 8, 128, 64)
    k_shape = (1, 2, 128, 64)
    v_shape = (1, 2, 128, 64)

    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = flash_attention_reference(Q, K, V)

    ttnn_Q = ttnn.from_torch(
        Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)

    assert list(output.shape) == list(q_shape)
    torch_output = ttnn.to_torch(output)

    pcc_threshold = PCC_TOLERANCES[ttnn.bfloat16]
    output_f32 = torch_output.float()
    expected_f32 = expected.float()
    output_flat = output_f32.flatten()
    expected_flat = expected_f32.flatten()
    output_centered = output_flat - output_flat.mean()
    expected_centered = expected_flat - expected_flat.mean()
    numerator = (output_centered * expected_centered).sum()
    denominator = torch.sqrt((output_centered**2).sum()) * torch.sqrt((expected_centered**2).sum())
    pcc = (numerator / denominator).item() if denominator > 0 else 1.0

    assert pcc >= pcc_threshold, f"GQA PCC {pcc:.6f} < threshold {pcc_threshold}"
