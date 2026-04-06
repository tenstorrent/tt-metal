# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    flush_subnormal_values_to_zero,
)


def generate_rpow_test_inputs(base, dtype=torch.bfloat16):
    """
    Generate test inputs appropriate for rpow (base^x).
    We use a range of exponents that won't cause extreme overflow/underflow.
    For base > 1, large positive x causes overflow; large negative x causes underflow.
    For 0 < base < 1, it's the opposite.
    """
    # Use a moderate range of exponents to avoid overflow/underflow
    if base > 1.0:
        # For base > 1, limit exponent range to avoid overflow
        # base^x overflows when x * log2(base) > ~127
        import math

        max_exp = min(10.0, 80.0 / max(math.log2(base), 0.01))
        min_exp = max(-10.0, -80.0 / max(math.log2(base), 0.01))
    elif base > 0.0:
        import math

        log2_inv_base = -math.log2(base) if base > 0 else 1.0
        max_exp = min(10.0, 80.0 / max(log2_inv_base, 0.01))
        min_exp = -max_exp
    else:
        max_exp = 5.0
        min_exp = -5.0

    # Generate a grid of test values
    n = 32 * 32  # one tile
    torch_input = torch.linspace(float(min_exp), float(max_exp), n, dtype=torch.float32)
    torch_input = torch_input.reshape(1, 1, 32, 32)

    if dtype == torch.bfloat16:
        torch_input = torch_input.to(torch.bfloat16)

    return torch_input


@pytest.mark.parametrize(
    "base",
    [
        2.0,
        0.5,
        3.0,
        10.0,
        1.5,
    ],
    ids=["base_2", "base_0.5", "base_3", "base_10", "base_1.5"],
)
@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_rpow(device, base, is_fp32):
    torch_input = generate_rpow_test_inputs(base, dtype=torch.bfloat16)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference: base^x
    torch_output = torch.pow(torch.tensor(base, dtype=torch.float32), torch_input.float())
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.rpow(tt_input, base)
    actual = ttnn.to_torch(tt_output)
    if not is_fp32:
        actual = actual.to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = (
        torch.isfinite(torch_input.float()) & torch.isfinite(expected.float()) & torch.isfinite(actual.float())
    )
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    if expected_finite.numel() == 0:
        pytest.skip("No finite values to compare")

    if is_fp32:
        # fp32 ULP threshold is higher due to polynomial approximation in exp_21f
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=32, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=5e-2, atol=1e-2)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=4)
        assert_allclose(expected_finite, actual_finite, rtol=5e-2, atol=1e-2)


@pytest.mark.parametrize("base", [1.0], ids=["base_1"])
def test_rpow_base_one(device, base):
    """base=1 should always produce 1.0 for any exponent."""
    torch_input = torch.linspace(-10.0, 10.0, 32 * 32, dtype=torch.bfloat16).reshape(1, 1, 32, 32)
    expected = torch.ones_like(torch_input)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rpow(tt_input, base)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)
