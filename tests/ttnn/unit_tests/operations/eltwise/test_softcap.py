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


def torch_softcap(x, cap):
    """Reference: softcap(x) = cap * tanh(x / cap)"""
    return cap * torch.tanh(x / cap)


@pytest.mark.parametrize("cap", [1.0, 10.0, 50.0], ids=["cap1", "cap10", "cap50"])
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_softcap(device, is_fp32, cap):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute golden reference in float32, flush subnormals to match hardware
    golden_input = flush_subnormal_values_to_zero(torch_input.float().clone())
    torch_output = torch_softcap(golden_input, cap)
    expected = flush_subnormal_values_to_zero(torch_output)
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

    # Filter NaN/Inf (tanh(inf/cap) is valid but inf/cap can produce NaN for 0*inf)
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    # ULP breaks down near zero — split into nonzero (ULP) and near-zero (allclose) regions
    nonzero_mask = torch.abs(expected_finite.float()) > 1e-30
    expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
    actual_nz = actual_finite[nonzero_mask].reshape(1, -1)

    # Strict thresholds from activation_function_error_thresholds.md
    # softcap = cap * tanh(x/cap): tanh is the only transcendental (2 ULP),
    # division and multiplication are 0 ULP on SFPU.
    if is_fp32:
        # FP32 Strict: ULP ≤ 2, rtol = 1.3e-6, atol = 1e-5
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1.3e-6, atol=1e-5)
    else:
        # FP16B Strict: ULP ≤ 2, rtol = 1.6e-2, atol = 1e-3
        if expected_nz.numel() > 0:
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-3)
