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
@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.125, 1.0 / 3.0),
        (0.0, 0.5),
        (0.1, 0.1),
    ],
    ids=["default", "wide", "constant"],
)
def test_rrelu_eval(device, is_fp32, lower, upper):
    """Test RReLU in eval mode (deterministic: slope = (lower + upper) / 2 for negative inputs)."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    if is_fp32:
        torch_input = torch_input.float()
        torch_input = flush_subnormal_values_to_zero(torch_input)

    # Compute reference in float32 using eval mode (training=False)
    torch_output = torch.nn.functional.rrelu(torch_input.float(), lower=lower, upper=upper, training=False)
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

    # allclose check
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)

    # ULP check for bfloat16
    if not is_fp32:
        nonzero_mask = expected_finite.float().abs() > 0.0
        if nonzero_mask.any():
            expected_nz = expected_finite[nonzero_mask].reshape(1, -1)
            actual_nz = actual_finite[nonzero_mask].reshape(1, -1)
            assert_with_ulp(expected_nz, actual_nz, ulp_threshold=2)
