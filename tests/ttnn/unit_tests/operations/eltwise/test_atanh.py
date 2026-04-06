# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
def test_atanh(device, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Filter to valid domain: |x| < 1 (strict inequality)
    # Replace out-of-domain values with 0.0 (a safe in-domain value)
    mask = torch_input.float().abs() < 1.0
    torch_input = torch.where(mask, torch_input, torch.zeros_like(torch_input))

    # Compute reference in float32, flush subnormals to match hardware behavior
    torch_output = torch.atanh(torch_input.float())
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.atanh(tt_input)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    # allclose covers the full domain including small values near zero
    # where the ln-based kernel has reduced precision due to catastrophic cancellation
    # (computing ln(1+x) - ln(1-x) for small x subtracts near-equal values).
    # fp32 uses wider atol because the SFPU cubic polynomial for ln provides ~2-3
    # decimal digits of accuracy regardless of accumulation format.
    if is_fp32:
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=2e-3)
    else:
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)

    # ULP check only for bfloat16 where it is meaningful given the polynomial accuracy.
    # fp32 ULP is too fine-grained for the cubic polynomial's ~10-bit effective precision.
    if not is_fp32:
        large_mask = expected_finite.float().abs() > 0.25
        if large_mask.any():
            expected_large = expected_finite[large_mask].reshape(1, -1)
            actual_large = actual_finite[large_mask].reshape(1, -1)
            assert_with_ulp(expected_large, actual_large, ulp_threshold=4)
