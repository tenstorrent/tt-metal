# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ttnn.rrelu (Randomized Leaky ReLU) SFPU operation.

RReLU(x) = x              if x >= 0
          = a * x          if x < 0
  - Eval mode  (seed == 0): a = (lower + upper) / 2   (deterministic)
  - Train mode (seed != 0): a ~ Uniform(lower, upper)  (random per element)

Two test groups:
  1. Eval mode  -- exhaustive bfloat16 bitpatterns, exact golden comparison
  2. Train mode -- verify output is in valid range for negative inputs
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


# ---------------------------------------------------------------------------
# 1. Eval-mode test (deterministic) -- exhaustive bfloat16, bfloat16 + fp32
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 0.333333),  # default
        (0.01, 0.5),
        (0.2, 0.2),  # lower == upper => fixed slope, identical to leaky_relu
    ],
)
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval(device, lower, upper, is_fp32):
    """Eval mode (seed=0): slope = (lower + upper) / 2, fully deterministic."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Golden: eval-mode RReLU is just leaky_relu with slope = midpoint
    midpoint = (lower + upper) / 2.0
    input_f32 = torch_input.float()
    torch_output = torch.where(input_f32 >= 0, input_f32, input_f32 * midpoint)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device -- seed=0 means eval mode (deterministic midpoint slope)
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, seed=0)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter NaN/Inf
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )

    if is_fp32:
        expected_finite = expected[finite_mask].float().reshape(1, -1)
        actual_finite = actual[finite_mask].float().reshape(1, -1)
        # Flush subnormal float32 artifacts before comparison
        expected_finite = flush_subnormal_values_to_zero(expected_finite)
        actual_finite = flush_subnormal_values_to_zero(actual_finite)
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        # Keep as bfloat16 for ULP comparison so ULP is measured in bfloat16 granularity.
        # Converting to float32 inflates ULP by 2^16 (65536) because float32 has 16 more
        # mantissa bits than bfloat16.
        expected_finite = expected[finite_mask].to(torch.bfloat16).reshape(1, -1)
        actual_finite = actual[finite_mask].to(torch.bfloat16).reshape(1, -1)
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        # allclose in float32 for better precision reporting
        assert_allclose(expected_finite.float(), actual_finite.float(), rtol=1.6e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 2. Training-mode test (random slopes) -- range checking
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_training(device, is_fp32):
    """Training mode (seed != 0): verify positive passthrough and negative range."""
    lower = 0.125
    upper = 0.333333

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, seed=42)
    actual = ttnn.to_torch(tt_output)

    # Flush subnormals from both input and output for comparison.
    # Hardware flushes subnormal inputs to zero; match this in the reference.
    input_f32 = flush_subnormal_values_to_zero(torch_input.float())
    actual_f32 = flush_subnormal_values_to_zero(actual.float())

    # --- Positive values: output == input (exact passthrough) ---
    # Use strictly positive to avoid zero-point artifacts
    pos_mask = input_f32 > 0.0
    pos_input = input_f32[pos_mask]
    pos_actual = actual_f32[pos_mask]

    # Filter to finite values only
    finite_pos = torch.isfinite(pos_input) & torch.isfinite(pos_actual)
    pos_input_f = pos_input[finite_pos]
    pos_actual_f = pos_actual[finite_pos]
    # Positive passthrough should be exact (or within 1 ULP for fp32 rounding)
    assert torch.allclose(
        pos_input_f, pos_actual_f, rtol=0, atol=0
    ), f"Positive passthrough failed: max diff = {(pos_input_f - pos_actual_f).abs().max().item()}"

    # --- Negative values: output in [upper * x, lower * x] ---
    # (since x < 0, upper * x < lower * x)
    neg_mask = input_f32 < 0.0
    neg_input = input_f32[neg_mask]
    neg_actual = actual_f32[neg_mask]

    finite_neg = torch.isfinite(neg_input) & torch.isfinite(neg_actual)
    neg_input_f = neg_input[finite_neg]
    neg_actual_f = neg_actual[finite_neg]

    # For negative x: lower_bound = upper * x (more negative), upper_bound = lower * x (less negative)
    neg_lower_bound = upper * neg_input_f  # more negative
    neg_upper_bound = lower * neg_input_f  # less negative

    # Allow small tolerance for floating point rounding
    tol = 1e-2 if not is_fp32 else 1e-4
    in_range = (neg_actual_f >= neg_lower_bound - tol) & (neg_actual_f <= neg_upper_bound + tol)
    violations = (~in_range).sum().item()
    total_neg = neg_input_f.numel()
    assert violations == 0, (
        f"Training mode range check failed: {violations}/{total_neg} negative values out of range "
        f"[upper*x, lower*x] = [{upper}*x, {lower}*x]"
    )
