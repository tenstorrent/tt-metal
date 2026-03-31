# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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


@pytest.mark.parametrize("lower, upper", [(0.125, 1.0 / 3.0)])
def test_rrelu_eval(device, lower, upper):
    """Test evaluation mode with ALL 65536 bfloat16 bit patterns."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Compute reference in float32, flush subnormals to match hardware behavior
    x_f32 = torch_input.float()
    midpoint = (lower + upper) / 2.0
    torch_output = torch.where(x_f32 >= 0, x_f32, midpoint * x_f32)
    expected = flush_subnormal_values_to_zero(torch_output).to(torch.bfloat16)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = torch.isfinite(torch_input) & torch.isfinite(expected) & torch.isfinite(actual)
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)
