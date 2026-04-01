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


@pytest.mark.parametrize("lower, upper", [(0.125, 1.0 / 3.0)])
def test_rrelu_training(device, lower, upper):
    """Test training mode: random slopes from U(lower, upper) for negative elements."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 64)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    x_f32 = torch_input.float()
    actual_f32 = actual.float()

    # Positive elements should pass through unchanged
    pos_mask = x_f32 >= 0
    if pos_mask.any():
        assert_allclose(
            actual_f32[pos_mask].reshape(1, -1),
            x_f32[pos_mask].reshape(1, -1),
            rtol=0,
            atol=0,
        )

    # Negative elements: verify slope is in [lower, upper]
    neg_mask = x_f32 < 0
    if neg_mask.any():
        neg_x = x_f32[neg_mask]
        neg_out = actual_f32[neg_mask]
        slopes = neg_out / neg_x
        # Slopes must be within [lower, upper] (with tolerance for bfloat16 precision)
        assert (slopes >= lower - 0.05).all(), f"Some slopes below lower bound: {slopes.min()}"
        assert (slopes <= upper + 0.05).all(), f"Some slopes above upper bound: {slopes.max()}"

    # Two calls in training mode should produce different outputs (stochastic)
    tt_input2 = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output2 = ttnn.rrelu(tt_input2, lower=lower, upper=upper, training=True)
    actual2 = ttnn.to_torch(tt_output2).to(torch.bfloat16)

    if neg_mask.any():
        # With random seeds, outputs for negative values should differ
        neg_out1 = actual.float()[neg_mask]
        neg_out2 = actual2.float()[neg_mask]
        assert not torch.allclose(neg_out1, neg_out2), "Training mode should produce stochastic outputs"
