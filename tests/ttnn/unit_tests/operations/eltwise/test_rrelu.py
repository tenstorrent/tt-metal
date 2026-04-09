# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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


def rrelu_golden(x, lower=0.125, upper=1.0 / 3.0):
    """RReLU in evaluation mode: output = x if x >= 0, else slope * x where slope = (lower + upper) / 2."""
    slope = (lower + upper) / 2.0
    return torch.where(x >= 0, x, slope * x)


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rrelu_default(device, is_fp32):
    """Test rrelu with default parameters (lower=0.125, upper=1/3)."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32
    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    torch_output = rrelu_golden(golden_input)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input)
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

    # Exclude near-zero from ULP check
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


@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.0, 0.0),  # acts like relu
        (1.0, 1.0),  # acts like identity
        (0.01, 0.99),  # wide range
        (0.125, 0.3333333432674408),  # default
    ],
    ids=["relu_like", "identity_like", "wide_range", "default"],
)
def test_rrelu_params(device, lower, upper):
    """Test rrelu with various lower/upper parameters (bfloat16 only)."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    torch_output = rrelu_golden(golden_input, lower=lower, upper=upper)
    expected = flush_subnormal_values_to_zero(torch_output).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
    actual = ttnn.to_torch(tt_output)
    actual = flush_subnormal_values_to_zero(actual).to(torch.bfloat16)

    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    nonzero_mask = torch.abs(expected_finite.float()) > 1e-30
    expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
    actual_nz = actual_finite[nonzero_mask].reshape(1, -1)

    if expected_nz.numel() > 0:
        assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
