# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 tests: causal masking for scaled_dot_product_attention.

Tests that is_causal=True generates a triangular -inf mask on-device and
produces output matching the PyTorch reference. Covers:
- Various shapes (single tile, multi-tile, multi-head, multi-batch, long context)
- Causal mask correctness (lower triangle attends, upper triangle masked)
- Self-attention only (cross+causal is EXCLUDED)
- All three dtypes (bf16, fp32, bf8b)
- Comparison with custom additive mask (same result via different path)
- Non-tile-aligned shapes with causal masking

Run with:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_causal.py
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


def _compute_pcc(output_f32, expected_f32):
    output_flat = output_f32.flatten()
    expected_flat = expected_f32.flatten()
    if output_flat.numel() > 1:
        output_centered = output_flat - output_flat.mean()
        expected_centered = expected_flat - expected_flat.mean()
        numerator = (output_centered * expected_centered).sum()
        denominator = torch.sqrt((output_centered**2).sum()) * torch.sqrt((expected_centered**2).sum())
        if denominator > 0:
            return (numerator / denominator).item()
        return 1.0 if torch.allclose(output_flat, expected_flat) else 0.0
    return 1.0 if torch.allclose(output_flat, expected_flat, atol=1e-2) else 0.0


# =============================================================================
# Shape parametrization
# =============================================================================

CAUSAL_SHAPES = [
    # (label, Q_shape, K_shape, V_shape) — all self-attention (S_q == S_kv)
    ("single_tile", (1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
    ("two_tiles", (1, 1, 64, 32), (1, 1, 64, 32), (1, 1, 64, 32)),
    ("multi_tile_128x64", (1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),
    ("multi_tile_128x128", (1, 1, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128)),
    ("multi_tile_256x64", (1, 1, 256, 64), (1, 1, 256, 64), (1, 1, 256, 64)),
    ("multi_batch", (2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
    ("multi_head", (1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64)),
    ("long_context", (1, 4, 512, 64), (1, 4, 512, 64), (1, 4, 512, 64)),
    ("non_pow2_heads", (1, 3, 96, 64), (1, 3, 96, 64), (1, 3, 96, 64)),
    ("large_model", (1, 32, 128, 128), (1, 32, 128, 128), (1, 32, 128, 128)),
    # Non-tile-aligned with causal
    ("h_non_aligned", (1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64)),
    ("w_non_aligned", (1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50)),
    ("both_non_aligned", (1, 1, 50, 50), (1, 1, 50, 50), (1, 1, 50, 50)),
    ("non_aligned_multi_head", (1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50)),
]

SCALE_MODES = [("auto", None), ("explicit", 0.125)]


@pytest.mark.parametrize(
    "label, q_shape, k_shape, v_shape",
    CAUSAL_SHAPES,
    ids=[s[0] for s in CAUSAL_SHAPES],
)
@pytest.mark.parametrize(
    "scale_label, scale",
    SCALE_MODES,
    ids=[s[0] for s in SCALE_MODES],
)
def test_causal_mask_bf16(device, label, q_shape, k_shape, v_shape, scale_label, scale):
    """Test is_causal=True with bfloat16 against PyTorch reference."""
    torch.manual_seed(42)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = flash_attention_reference(torch_Q, torch_K, torch_V, is_causal=True, scale=scale)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True, scale=scale)
    torch_output = ttnn.to_torch(output)

    assert list(output.shape) == list(q_shape)
    assert not torch.any(torch.isnan(torch_output)), f"NaN in output for {label}"
    assert not torch.any(torch.isinf(torch_output)), f"Inf in output for {label}"

    pcc = _compute_pcc(torch_output.float(), expected.float())
    assert (
        pcc >= PCC_TOLERANCES[ttnn.bfloat16]
    ), f"Causal PCC {pcc:.6f} < {PCC_TOLERANCES[ttnn.bfloat16]} for {label}/{scale_label}"


@pytest.mark.parametrize(
    "label, q_shape, k_shape, v_shape",
    CAUSAL_SHAPES,
    ids=[s[0] for s in CAUSAL_SHAPES],
)
def test_causal_mask_fp32(device, label, q_shape, k_shape, v_shape):
    """Test is_causal=True with float32."""
    torch.manual_seed(42)

    torch_Q = torch.randn(q_shape, dtype=torch.float32)
    torch_K = torch.randn(k_shape, dtype=torch.float32)
    torch_V = torch.randn(v_shape, dtype=torch.float32)

    expected = flash_attention_reference(torch_Q, torch_K, torch_V, is_causal=True)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True, compute_kernel_config=cfg)
    torch_output = ttnn.to_torch(output)

    assert list(output.shape) == list(q_shape)
    assert not torch.any(torch.isnan(torch_output)), f"NaN in output for {label}"

    pcc = _compute_pcc(torch_output.float(), expected.float())
    assert (
        pcc >= PCC_TOLERANCES[ttnn.float32]
    ), f"Causal fp32 PCC {pcc:.6f} < {PCC_TOLERANCES[ttnn.float32]} for {label}"


@pytest.mark.parametrize(
    "label, q_shape, k_shape, v_shape",
    CAUSAL_SHAPES,
    ids=[s[0] for s in CAUSAL_SHAPES],
)
def test_causal_mask_bf8b(device, label, q_shape, k_shape, v_shape):
    """Test is_causal=True with bfloat8_b."""
    torch.manual_seed(42)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = flash_attention_reference(torch_Q, torch_K, torch_V, is_causal=True)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True, compute_kernel_config=cfg)
    torch_output = ttnn.to_torch(output)

    assert list(output.shape) == list(q_shape)
    assert not torch.any(torch.isnan(torch_output)), f"NaN in output for {label}"

    pcc = _compute_pcc(torch_output.float(), expected.float())
    assert (
        pcc >= PCC_TOLERANCES[ttnn.bfloat8_b]
    ), f"Causal bf8b PCC {pcc:.6f} < {PCC_TOLERANCES[ttnn.bfloat8_b]} for {label}"


def test_causal_equals_custom_mask(device):
    """is_causal=True should produce the same result as an equivalent custom mask."""
    torch.manual_seed(42)

    shape = (1, 4, 128, 64)
    torch_Q = torch.randn(shape, dtype=torch.bfloat16)
    torch_K = torch.randn(shape, dtype=torch.bfloat16)
    torch_V = torch.randn(shape, dtype=torch.bfloat16)

    # Custom mask: upper triangular -inf (same as causal)
    B, H, S, D = shape
    custom_mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    custom_mask.masked_fill_(
        torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_mask = ttnn.from_torch(
        custom_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_causal = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True)
    output_custom = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask)

    torch_causal = ttnn.to_torch(output_causal).float()
    torch_custom = ttnn.to_torch(output_custom).float()

    pcc = _compute_pcc(torch_causal, torch_custom)
    assert pcc >= 0.999, f"Causal vs custom mask PCC {pcc:.6f} — should be near-identical"


def test_causal_gqa(device):
    """GQA with causal masking."""
    torch.manual_seed(42)

    q_shape = (1, 8, 128, 64)
    k_shape = (1, 2, 128, 64)
    v_shape = (1, 2, 128, 64)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = flash_attention_reference(torch_Q, torch_K, torch_V, is_causal=True)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True)
    torch_output = ttnn.to_torch(output)

    assert list(output.shape) == list(q_shape)
    pcc = _compute_pcc(torch_output.float(), expected.float())
    assert pcc >= PCC_TOLERANCES[ttnn.bfloat16], f"GQA causal PCC {pcc:.6f}"


def test_causal_mqa(device):
    """MQA with causal masking."""
    torch.manual_seed(42)

    q_shape = (1, 8, 128, 64)
    k_shape = (1, 1, 128, 64)
    v_shape = (1, 1, 128, 64)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = flash_attention_reference(torch_Q, torch_K, torch_V, is_causal=True)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True)
    torch_output = ttnn.to_torch(output)

    assert list(output.shape) == list(q_shape)
    pcc = _compute_pcc(torch_output.float(), expected.float())
    assert pcc >= PCC_TOLERANCES[ttnn.bfloat16], f"MQA causal PCC {pcc:.6f}"


def test_causal_single_tile_diagonal(device):
    """Single-tile causal: the diagonal-straddling case only.
    For a single 32x32 tile, the causal mask is entirely diagonal-straddling.
    """
    torch.manual_seed(42)

    shape = (1, 1, 32, 32)
    torch_Q = torch.randn(shape, dtype=torch.bfloat16)
    torch_K = torch.randn(shape, dtype=torch.bfloat16)
    torch_V = torch.randn(shape, dtype=torch.bfloat16)

    expected = flash_attention_reference(torch_Q, torch_K, torch_V, is_causal=True)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True)
    torch_output = ttnn.to_torch(output)

    pcc = _compute_pcc(torch_output.float(), expected.float())
    assert pcc >= PCC_TOLERANCES[ttnn.bfloat16], f"Single-tile causal PCC {pcc:.6f}"


def test_causal_exclusion_cross_attention(device):
    """is_causal=True with cross-attention should raise ExcludedCell."""
    q_shape = (1, 4, 64, 64)
    k_shape = (1, 4, 128, 64)
    v_shape = (1, 4, 128, 64)

    torch_Q = torch.randn(q_shape, dtype=torch.bfloat16)
    torch_K = torch.randn(k_shape, dtype=torch.bfloat16)
    torch_V = torch.randn(v_shape, dtype=torch.bfloat16)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with pytest.raises((NotImplementedError, RuntimeError)):
        scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, is_causal=True)


def test_causal_and_attn_mask_mutually_exclusive(device):
    """is_causal=True and attn_mask is not None should raise ValueError."""
    shape = (1, 1, 32, 32)
    torch_Q = torch.randn(shape, dtype=torch.bfloat16)
    torch_K = torch.randn(shape, dtype=torch.bfloat16)
    torch_V = torch.randn(shape, dtype=torch.bfloat16)
    mask = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)

    ttnn_Q = ttnn.from_torch(
        torch_Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_K = ttnn.from_torch(
        torch_K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_V = ttnn.from_torch(
        torch_V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_mask = ttnn.from_torch(
        mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    with pytest.raises(ValueError):
        scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask, is_causal=True)
