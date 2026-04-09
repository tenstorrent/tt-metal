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
def test_rrelu_default(device, is_fp32):
    """Test rrelu with default parameters (lower=0.125, upper=1/3) over all bfloat16 bitpatterns."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32
    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    torch_output = torch.nn.functional.rrelu(golden_input, lower=0.125, upper=1.0 / 3.0, training=False)
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

    # Filter NaN/Inf
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    # Exclude near-zero from ULP check (ULP metric breaks down near zero)
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
def test_rrelu_custom_params(device, is_fp32):
    """Test rrelu with explicit lower/upper parameters."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)

    lower = 0.125
    upper = 0.333333

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    torch_output = torch.nn.functional.rrelu(golden_input, lower=lower, upper=upper, training=False)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
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
