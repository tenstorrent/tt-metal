# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval(device, is_fp32):
    """Test RReLU in evaluation mode (deterministic slope = (lower + upper) / 2)."""
    lower = 0.125
    upper = 1.0 / 3.0

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference: eval mode uses fixed slope
    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    slope = (lower + upper) / 2.0
    torch_output = torch.where(golden_input >= 0, golden_input, golden_input * slope)
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
    actual = flush_subnormal_values_to_zero(actual)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    nonzero_mask = torch.abs(expected_finite.float()) > 1e-30
    expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
    actual_nz = actual_finite[nonzero_mask].reshape(1, -1)

    if is_fp32:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval_custom_bounds(device, is_fp32):
    """Test RReLU eval mode with non-default lower/upper bounds."""
    lower = 0.05
    upper = 0.5

    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    slope = (lower + upper) / 2.0
    torch_output = torch.where(golden_input >= 0, golden_input, golden_input * slope)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    actual = ttnn.to_torch(tt_output)
    actual = flush_subnormal_values_to_zero(actual)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    nonzero_mask = torch.abs(expected_finite.float()) > 1e-30
    expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
    actual_nz = actual_finite[nonzero_mask].reshape(1, -1)

    if is_fp32:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=3, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_train(device, is_fp32):
    """Test RReLU in training mode (stochastic slope per element).

    Verifies:
    1. Positive values pass through unchanged.
    2. Negative values are scaled by a factor in [lower, upper].
    """
    lower = 0.125
    upper = 1.0 / 3.0

    # Use a small tile-aligned shape for training test
    torch_input = torch.randn(1, 1, 32, 64)
    if is_fp32:
        torch_input = flush_subnormal_values_to_zero(torch_input.float())
    else:
        torch_input = torch_input.to(torch.bfloat16)

    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output)

    input_float = torch_input.float()
    actual_float = actual.float()

    # 1. Positive values should pass through unchanged
    pos_mask = input_float >= 0
    if pos_mask.any():
        expected_pos = input_float[pos_mask]
        actual_pos = actual_float[pos_mask]
        if is_fp32:
            assert_allclose(expected_pos.reshape(1, -1), actual_pos.reshape(1, -1), rtol=1e-5, atol=1e-6)
        else:
            assert_allclose(expected_pos.reshape(1, -1), actual_pos.reshape(1, -1), rtol=1e-2, atol=1e-2)

    # 2. Negative values: actual = slope * input, where slope in [lower, upper]
    neg_mask = input_float < 0
    if neg_mask.any():
        neg_input = input_float[neg_mask]
        neg_actual = actual_float[neg_mask]
        # Compute the implied slope: slope = actual / input
        slopes = neg_actual / neg_input
        # Slopes should be in [lower, upper] with some tolerance for bfloat16 precision
        tolerance = 0.05  # generous tolerance for bfloat16 quantization + PRNG approximation
        assert slopes.min() >= lower - tolerance, (
            f"Min slope {slopes.min():.4f} below lower bound {lower} - {tolerance}"
        )
        assert slopes.max() <= upper + tolerance, (
            f"Max slope {slopes.max():.4f} above upper bound {upper} + {tolerance}"
        )


def test_rrelu_positive_passthrough(device):
    """Test that positive values are always unchanged regardless of mode."""
    torch_input = torch.abs(torch.randn(1, 1, 32, 32)) + 0.01  # All positive

    tt_input = ttnn.from_torch(torch_input.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    # Eval mode
    tt_out_eval = ttnn.rrelu(tt_input, training=False)
    actual_eval = ttnn.to_torch(tt_out_eval)

    # Train mode
    tt_out_train = ttnn.rrelu(tt_input, training=True)
    actual_train = ttnn.to_torch(tt_out_train)

    expected = torch_input.to(torch.bfloat16)
    assert_allclose(expected.reshape(1, -1), actual_eval.reshape(1, -1), rtol=1e-2, atol=1e-2)
    assert_allclose(expected.reshape(1, -1), actual_train.reshape(1, -1), rtol=1e-2, atol=1e-2)
