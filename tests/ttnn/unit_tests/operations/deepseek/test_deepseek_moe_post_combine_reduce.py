# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for deepseek_moe_post_combine_reduce fused kernel.

Validates correctness against:
- PyTorch reference (weighted sum across experts)
- Old implementation from tt_moe.py (to_layout + mul + sum)

Tests structured data, random data, and sparse weights.
Shape: [1, 3200, 8, 7168] - DeepSeek-V3 dimensions.
"""

import pytest
import torch
import ttnn
from loguru import logger


NUM_TOKENS = 3200
NUM_EXPERTS = 8
EMB_DIM = 7168
EXPERT_DIM = 2
PCC_THRESHOLD = 0.999


def pytorch_reference(combine, weights):
    """PyTorch reference: weighted sum across experts."""
    return (combine * weights.expand(-1, -1, -1, combine.shape[-1])).sum(dim=EXPERT_DIM)


def old_implementation(combine_tt, weights_tt):
    """Old implementation as used in tt_moe.py: to_layout(TILE) + mul + sum."""
    combine_tiled = ttnn.to_layout(combine_tt, ttnn.TILE_LAYOUT)
    weights_tiled = ttnn.to_layout(weights_tt, ttnn.TILE_LAYOUT)
    weighted = ttnn.mul(combine_tiled, weights_tiled)
    return ttnn.sum(weighted, dim=EXPERT_DIM)


def new_implementation(combine_tt, weights_tt):
    """Fused kernel: reads ROW_MAJOR, produces TILE output."""
    return ttnn.experimental.deepseek_moe_post_combine_reduce(
        combine_tt, weights_tt, expert_dim=EXPERT_DIM, output_memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def compute_pcc(a, b):
    """Compute PCC between two tensors."""
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def to_device(tensor, device):
    return ttnn.from_torch(tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def assert_pcc(result, expected, threshold=PCC_THRESHOLD, label=""):
    nan_count = torch.isnan(result).sum().item()
    assert nan_count == 0, f"{label}: got {nan_count} NaN elements"
    pcc = compute_pcc(result, expected)
    logger.info(f"  {label}: PCC={pcc:.6f}")
    assert pcc > threshold, f"{label}: PCC {pcc:.6f} below {threshold}"
    return pcc


# ============================================================================
# Structured data test
# ============================================================================


def test_structured_data(device):
    """Constant-per-tile activations with sequential weights [1..8].
    This pattern is easy to verify manually and catches tile ordering bugs."""
    combine = torch.zeros(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    tile_value = 0.1
    for t in range(NUM_TOKENS):
        for e in range(NUM_EXPERTS):
            for tile in range(EMB_DIM // 1024):
                combine[0, t, e, tile * 1024 : (tile + 1) * 1024] = tile_value
                tile_value += 0.1

    weights = torch.zeros(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)
    for e in range(NUM_EXPERTS):
        weights[0, :, e, 0] = e + 1.0

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(new_implementation(to_device(combine, device), to_device(weights, device)))
    assert_pcc(result, ref, threshold=0.998, label="structured")


# ============================================================================
# Random data tests
# ============================================================================


def test_random_data(device):
    """Random activations and weights, compared to PyTorch reference."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(new_implementation(to_device(combine, device), to_device(weights, device)))
    assert_pcc(result, ref, label="random")


def test_vs_old_implementation(device):
    """Fused kernel vs old implementation (to_layout + mul + sum) with random data."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    ref = pytorch_reference(combine, weights)
    combine_tt = to_device(combine, device)
    weights_tt = to_device(weights, device)

    old_result = ttnn.to_torch(old_implementation(combine_tt, weights_tt))
    new_result = ttnn.to_torch(new_implementation(combine_tt, weights_tt))

    assert_pcc(old_result, ref, label="old_vs_ref")
    assert_pcc(new_result, ref, label="new_vs_ref")
    assert_pcc(old_result, new_result, label="old_vs_new")


# ============================================================================
# Sparse weight tests
# ============================================================================


@pytest.mark.parametrize("k_active", [6, 4, 2, 1])
def test_sparse_weights(device, k_active):
    """Fused kernel with sparse weights (k_active out of 8 experts non-zero per token)."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.zeros(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)
    for t in range(NUM_TOKENS):
        active = torch.randperm(NUM_EXPERTS)[:k_active]
        weights[0, t, active, 0] = torch.randn(k_active, dtype=torch.bfloat16)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(new_implementation(to_device(combine, device), to_device(weights, device)))
    assert_pcc(result, ref, label=f"sparse_{k_active}/{NUM_EXPERTS}")


# ============================================================================
# Output format test
# ============================================================================


def test_output_layout(device):
    """Verify output is TILE layout with correct shape."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    result_tt = new_implementation(to_device(combine, device), to_device(weights, device))
    assert result_tt.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {result_tt.layout}"
    assert list(result_tt.shape) == [1, NUM_TOKENS, EMB_DIM], f"Wrong shape: {result_tt.shape}"


# ============================================================================
# Entry point
# ============================================================================


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        logger.info("=== Structured data ===")
        test_structured_data(device)

        logger.info("\n=== Random data ===")
        test_random_data(device)

        logger.info("\n=== vs old implementation ===")
        test_vs_old_implementation(device)

        for k in [6, 4, 2, 1]:
            logger.info(f"\n=== Sparse {k}/{NUM_EXPERTS} ===")
            test_sparse_weights(device, k)

        logger.info("\n=== Output layout ===")
        test_output_layout(device)

        logger.info("\nAll tests passed!")
    finally:
        ttnn.close_device(device)
