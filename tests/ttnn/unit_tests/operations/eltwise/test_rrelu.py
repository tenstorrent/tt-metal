# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the rrelu (Randomized Leaky ReLU) SFPU operation.

Two test modes:
  - Eval mode (training=False): deterministic, slope = (lower + upper) / 2.
    Exhaustive comparison over all bfloat16 bit patterns.
  - Training mode (training=True): non-deterministic random slope in [lower, upper].
    Verifies positive passthrough and that negative outputs fall in the valid range.
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
# Eval mode test (deterministic) -- exhaustive bfloat16 + fp32
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval(device, is_fp32):
    """Test rrelu in eval mode (training=False) with all bfloat16 bit patterns."""
    lower = 0.125
    upper = 1.0 / 3.0

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Eval-mode golden: slope = (lower + upper) / 2
    slope = (lower + upper) / 2.0
    torch_output = torch.where(
        torch_input.float() >= 0,
        torch_input.float(),
        torch_input.float() * slope,
    )
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    if is_fp32:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Training mode test (non-deterministic) -- range check
# ---------------------------------------------------------------------------
def test_rrelu_training(device):
    """
    Test rrelu in training mode (training=True).

    For x >= 0:  output == input  (identity)
    For x < 0:   lower * x <= output <= upper * x
        (Note: since x < 0, upper*x is more negative, lower*x is less negative,
         so the bounds flip: upper*x <= output <= lower*x)
    """
    lower = 0.125
    upper = 1.0 / 3.0

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    # Flush subnormals in both input and output to match hardware behavior
    torch_input_f32 = flush_subnormal_values_to_zero(torch_input.float())
    actual_f32 = flush_subnormal_values_to_zero(actual.float())

    # Only check finite, non-zero values (zero inputs can produce tiny subnormal
    # artifacts due to PRNG multiplication; these get flushed to zero above)
    finite_mask = torch.isfinite(torch_input_f32) & torch.isfinite(actual_f32)
    input_finite = torch_input_f32[finite_mask]
    output_finite = actual_f32[finite_mask]

    # Positive values: output == input (identity)
    # Use strictly positive mask (exclude zero, which can have -0.0 vs 0.0 issues)
    pos_mask = input_finite > 0
    pos_input = input_finite[pos_mask]
    pos_output = output_finite[pos_mask]
    assert torch.equal(
        pos_output.to(torch.bfloat16), pos_input.to(torch.bfloat16)
    ), "Training mode: positive values should pass through unchanged"

    # Zero inputs: output should also be zero (or flushed subnormal)
    zero_mask = input_finite == 0
    zero_output = output_finite[zero_mask]
    assert (zero_output == 0).all(), (
        f"Training mode: zero inputs should produce zero output, " f"got {zero_output[zero_output != 0][:5].tolist()}"
    )

    # Negative values: output should be in [upper*x, lower*x]
    # (upper*x is more negative because upper > lower and x < 0)
    neg_mask = input_finite < 0
    neg_input = input_finite[neg_mask]
    neg_output = output_finite[neg_mask]

    lower_bound = neg_input * upper  # more negative (smaller)
    upper_bound = neg_input * lower  # less negative (larger)

    # Allow a small tolerance for bfloat16 rounding
    tol = 1e-2
    violations_low = (neg_output < lower_bound - tol).sum().item()
    violations_high = (neg_output > upper_bound + tol).sum().item()
    total_neg = neg_input.numel()

    assert violations_low == 0, (
        f"Training mode: {violations_low}/{total_neg} negative outputs below lower_bound (upper*x). "
        f"Worst: output={neg_output[neg_output < lower_bound - tol][:5].tolist()}, "
        f"bound={lower_bound[neg_output < lower_bound - tol][:5].tolist()}"
    )
    assert violations_high == 0, (
        f"Training mode: {violations_high}/{total_neg} negative outputs above upper_bound (lower*x). "
        f"Worst: output={neg_output[neg_output > upper_bound + tol][:5].tolist()}, "
        f"bound={upper_bound[neg_output > upper_bound + tol][:5].tolist()}"
    )

    # Verify that the random slopes are not all the same (i.e., actual randomness)
    # Compute implied slope: output / input for negative values (avoid div by zero)
    nonzero_neg_mask = neg_input.abs() > 1e-6
    if nonzero_neg_mask.sum() > 10:
        slopes = neg_output[nonzero_neg_mask] / neg_input[nonzero_neg_mask]
        unique_slopes = slopes.unique()
        assert unique_slopes.numel() > 1, (
            "Training mode: all negative slopes are identical -- expected randomness. "
            f"Got slope = {unique_slopes.tolist()}"
        )
