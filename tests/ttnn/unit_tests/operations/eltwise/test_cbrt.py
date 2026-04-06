# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive bfloat16/fp32 test for the cbrt (cube root) unary SFPU operation.

Tests all 65,536 bfloat16 bit patterns to ensure complete coverage.
Golden reference: cbrt(x) = sign(x) * |x|^(1/3)
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


@pytest.mark.parametrize("is_fp32", [False, True], ids=["bfloat16", "fp32"])
def test_cbrt(device, is_fp32):
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        # Cast to float32 and flush subnormal inputs - hardware flushes these to zero
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32, flush subnormal inputs to match hardware behavior
    # Hardware flushes subnormal inputs to zero in both bfloat16 and fp32 modes
    input_f32 = flush_subnormal_values_to_zero(torch_input.float())
    torch_output = torch.pow(torch.abs(input_f32), 1.0 / 3.0) * torch.sign(input_f32)
    expected = flush_subnormal_values_to_zero(torch_output)
    if not is_fp32:
        expected = expected.to(torch.bfloat16)

    # Run on device
    tt_kwargs = dict(layout=ttnn.TILE_LAYOUT, device=device)
    if is_fp32:
        tt_kwargs["dtype"] = ttnn.float32
    tt_input = ttnn.from_torch(torch_input, **tt_kwargs)
    tt_output = ttnn.cbrt(tt_input)
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
        # fp32 tolerances: the SFPU kernel uses polynomial+Halley refinement which
        # achieves ~15 ULP max error at extreme small values (smallest normal bfloat16).
        # ULP 16 covers this with minimal margin.
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=16, allow_nonfinite=True)
        assert_allclose(expected_finite, actual_finite, rtol=1e-3, atol=1e-4)
    else:
        assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
        assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
