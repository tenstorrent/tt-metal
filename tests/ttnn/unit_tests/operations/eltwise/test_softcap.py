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


def softcap_reference(x, cap):
    """Reference: cap * tanh(x / cap), computed in float64 for precision."""
    x64 = x.double()
    return (cap * torch.tanh(x64 / cap)).float()


@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0], ids=["cap1", "cap10", "cap50"])
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_softcap(device, is_fp32, cap):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float64, then convert to target precision
    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    expected = softcap_reference(golden_input, cap)
    expected = flush_subnormal_values_to_zero(expected)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.softcap(tt_input, cap=cap)
    actual = ttnn.to_torch(tt_output)
    actual = flush_subnormal_values_to_zero(actual)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    # ULP metric breaks down near zero — exclude near-zero expected values from ULP check
    nonzero_mask = torch.abs(expected_finite.float()) > 1e-30
    expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
    actual_nz = actual_finite[nonzero_mask].reshape(1, -1)

    if is_fp32:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=10, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
