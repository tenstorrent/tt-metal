# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


def test_lgamma_exhaustive_bfloat16(device):
    """Exhaustive bfloat16 test: all 2^16 bit patterns within a valid range.

    Uses allclose because lgamma passes through zero at x=1 and x=2,
    making ULP comparison unreliable for near-zero outputs.
    """
    input_tensor = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    input_tensor = flush_subnormal_values_to_zero(input_tensor)

    input_f32 = input_tensor.to(torch.float32)

    # lgamma Lanczos approx is valid for x > 0; exclude x <= 0, overflow, inf/nan
    mask = (input_f32 > 0.0) & (input_f32 <= 60.0) & torch.isfinite(input_f32)
    input_tensor = input_tensor[mask]

    # Pad to tile-aligned shape (multiple of 32)
    n = input_tensor.numel()
    pad = (32 - n % 32) % 32
    if pad > 0:
        input_tensor = torch.cat([input_tensor, torch.ones(pad, dtype=torch.bfloat16)])

    input_tensor = input_tensor.reshape(1, 1, -1, 32)

    golden = torch.lgamma(input_tensor.to(torch.float32))

    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.lgamma(tt_input)
    result = ttnn.to_torch(tt_result)

    golden_flat = golden.flatten()[:n]
    result_flat = result.to(torch.float32).flatten()[:n]

    torch.testing.assert_close(result_flat, golden_flat, rtol=1.6e-2, atol=1e-2)


def test_lgamma_ulp_bfloat16(device):
    """ULP test on values where lgamma output is large enough for meaningful ULP comparison.

    Excludes the region near x=1 and x=2 where lgamma approaches zero.
    """
    input_tensor = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    input_tensor = flush_subnormal_values_to_zero(input_tensor)

    input_f32 = input_tensor.to(torch.float32)

    # Valid positive range, exclude overflow
    mask = (input_f32 > 0.0) & (input_f32 <= 60.0) & torch.isfinite(input_f32)

    # Exclude regions where lgamma output is near zero (ULP comparison breaks down)
    # lgamma has zeros at x=1 and x=2, and a minimum near x=1.46 (~-0.12)
    lgamma_f32 = torch.lgamma(input_f32)
    mask &= torch.abs(lgamma_f32) > 0.5

    input_tensor = input_tensor[mask]

    n = input_tensor.numel()
    pad = (32 - n % 32) % 32
    if pad > 0:
        input_tensor = torch.cat([input_tensor, torch.full((pad,), 5.0, dtype=torch.bfloat16)])

    input_tensor = input_tensor.reshape(1, 1, -1, 32)

    golden = torch.lgamma(input_tensor.to(torch.float32))

    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.lgamma(tt_input)
    result = ttnn.to_torch(tt_result)

    golden_flat = golden.flatten()[:n]
    result_flat = result.flatten()[:n]

    assert_with_ulp(golden_flat, result_flat, ulp_threshold=3)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (3, 4, 64, 32),
    ],
)
def test_lgamma_bfloat16_random(device, shapes):
    """Random bfloat16 inputs in the positive range."""
    torch.manual_seed(42)
    # Avoid the near-zero regions of lgamma (near x=1 and x=2)
    torch_input = torch.empty(shapes, dtype=torch.bfloat16).uniform_(3.0, 50.0)

    golden = torch.lgamma(torch_input.to(torch.float32))

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.lgamma(tt_input)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, ulp_threshold=3)


def test_lgamma_special_values(device):
    """Test lgamma at known special values: lgamma(1)=0, lgamma(2)=0."""
    special_vals = torch.tensor([1.0, 2.0, 1.0, 2.0] * 8, dtype=torch.bfloat16).reshape(1, 1, 1, 32)

    golden = torch.lgamma(special_vals.to(torch.float32))

    tt_input = ttnn.from_torch(
        special_vals,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.lgamma(tt_input)
    result = ttnn.to_torch(tt_result).to(torch.float32)

    torch.testing.assert_close(result, golden, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shapes",
    [
        (1, 1, 32, 32),
        (3, 4, 64, 32),
        (128, 128),
    ],
)
def test_lgamma_fp32(device, shapes):
    """Test lgamma with float32 precision.

    Note: the SFPU kernel uses 1 Newton-Raphson iteration for reciprocal,
    which limits precision to ~bfloat16 level even with float32 inputs.
    """
    torch.manual_seed(42)
    # Avoid near-zero lgamma region (x near 1 and 2)
    torch_input = torch.empty(shapes, dtype=torch.float32).uniform_(3.0, 50.0)

    golden = torch.lgamma(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.lgamma(tt_input)
    result = ttnn.to_torch(tt_result).to(torch.float32)

    torch.testing.assert_close(result, golden, rtol=1.6e-2, atol=1e-2)
