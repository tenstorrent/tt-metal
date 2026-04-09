# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    """Test rrelu in evaluation mode (training=False) with default lower=1/8, upper=1/3."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference: rrelu in eval mode uses fixed slope = (lower + upper) / 2
    lower = 0.125
    upper = 1.0 / 3.0
    neg_slope = (lower + upper) / 2.0

    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    # Manual computation: x if x >= 0, else neg_slope * x
    expected = torch.where(golden_input >= 0, golden_input, golden_input * neg_slope)
    expected = flush_subnormal_values_to_zero(expected)
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

    # Filter NaN/Inf
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
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=4, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_eval_custom_bounds(device, is_fp32):
    """Test rrelu eval mode with custom lower/upper bounds."""
    torch.manual_seed(42)
    torch_input = torch.randn((64, 128), dtype=torch.bfloat16)

    lower = 0.05
    upper = 0.5
    neg_slope = (lower + upper) / 2.0

    if is_fp32:
        torch_input = torch_input.float()

    golden_input = torch_input.float().clone()
    expected = torch.where(golden_input >= 0, golden_input, golden_input * neg_slope)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    actual = ttnn.to_torch(tt_output)

    finite_mask = torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    if is_fp32:
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


def test_rrelu_training_mode(device):
    """Test rrelu in training mode - slopes should be in [lower, upper] range."""
    torch.manual_seed(42)
    # Use negative values to ensure slopes are actually applied
    torch_input = -torch.abs(torch.randn((64, 128), dtype=torch.bfloat16))

    lower = 0.125
    upper = 1.0 / 3.0

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output)

    # In training mode, for negative x: output = slope * x where slope in [lower, upper]
    # So output / x should give slope, and it should be in [lower, upper]
    # Use float for division
    input_float = torch_input.float()
    actual_float = actual.float()

    # Only check where input is significantly negative (avoid division by near-zero)
    neg_mask = input_float < -0.01
    if neg_mask.any():
        slopes = actual_float[neg_mask] / input_float[neg_mask]
        # Check slopes are in reasonable range (with some tolerance for bfloat16)
        assert slopes.min() >= lower - 0.05, f"Min slope {slopes.min()} < {lower - 0.05}"
        assert slopes.max() <= upper + 0.05, f"Max slope {slopes.max()} > {upper + 0.05}"

    # For positive values, output should equal input
    pos_mask = torch_input > 0.01
    if pos_mask.any():
        assert_allclose(
            torch_input[pos_mask].reshape(1, -1).float(),
            actual[pos_mask].reshape(1, -1).float(),
            rtol=1e-2,
            atol=1e-2,
        )
