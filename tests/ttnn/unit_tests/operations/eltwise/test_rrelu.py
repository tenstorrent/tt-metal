import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),  # default PyTorch values
    ],
)
def test_rrelu_eval(device, lower, upper):
    """Test RReLU in eval/inference mode with fixed slope = (lower + upper) / 2."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Compute reference in float32 -- eval mode uses fixed slope = (lower + upper) / 2
    slope = (lower + upper) / 2.0
    inp_f32 = torch_input.float()
    torch_output = torch.where(inp_f32 >= 0, inp_f32, slope * inp_f32)
    expected = flush_subnormal_values_to_zero(torch_output).to(torch.bfloat16)

    # Run on device
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    # Filter out NaN/Inf for meaningful comparison
    finite_mask = torch.isfinite(torch_input) & torch.isfinite(expected) & torch.isfinite(actual)
    expected_finite = expected[finite_mask].reshape(1, -1)
    actual_finite = actual[finite_mask].reshape(1, -1)

    assert_with_ulp(expected_finite, actual_finite, ulp_threshold=2)
    assert_allclose(expected_finite, actual_finite, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),  # default PyTorch values
    ],
)
def test_rrelu_training(device, lower, upper):
    """Test RReLU in training mode with random per-element slopes in [lower, upper)."""
    torch_input = generate_all_bfloat16_bitpatterns(dtype=torch.bfloat16)  # (256, 256)

    # Run on device with training=True
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)

    inp_f32 = torch_input.float()
    actual_f32 = actual.float()

    # For positive inputs: output should equal input (accounting for subnormal flushing)
    # Hardware flushes subnormals to zero, so we compare after flushing
    positive_mask = torch.isfinite(torch_input) & (torch_input > 0)
    if positive_mask.any():
        expected_pos = flush_subnormal_values_to_zero(torch_input[positive_mask].float()).to(torch.bfloat16)
        actual_pos = actual[positive_mask].to(torch.bfloat16)
        assert torch.equal(
            expected_pos, actual_pos
        ), "Positive inputs should pass through unchanged (after subnormal flush)"

    # For negative inputs: output should be in range [upper * input, lower * input]
    # (since input is negative, multiplying by a larger slope gives a MORE negative value)
    negative_mask = torch.isfinite(torch_input) & (torch_input < 0) & torch.isfinite(actual)
    if negative_mask.any():
        neg_input = inp_f32[negative_mask]
        neg_actual = actual_f32[negative_mask]

        # For negative x: lower * x <= a * x <= upper * x
        # But since x < 0, upper * x <= a * x <= lower * x
        lower_bound = upper * neg_input  # more negative (upper slope * negative input)
        upper_bound = lower * neg_input  # less negative (lower slope * negative input)

        # Allow small tolerance for bfloat16 rounding
        tolerance = 1e-2 + 1e-2 * torch.abs(neg_input)
        assert (neg_actual >= lower_bound - tolerance).all(), (
            f"Some outputs below lower bound: " f"min_diff={float((neg_actual - lower_bound + tolerance).min())}"
        )
        assert (neg_actual <= upper_bound + tolerance).all(), (
            f"Some outputs above upper bound: " f"max_diff={float((neg_actual - upper_bound - tolerance).max())}"
        )

    # For zero inputs: output should be zero
    zero_mask = torch_input == 0
    if zero_mask.any():
        assert (actual[zero_mask] == 0).all(), "Zero inputs should produce zero outputs"


@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),
    ],
)
def test_rrelu_training_randomness(device, lower, upper):
    """Verify that training mode produces different outputs across runs (non-deterministic)."""
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    # Make all values negative to ensure the random slope is applied
    torch_input = -torch.abs(torch_input)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # Run twice with training=True -- results should differ (different random seeds)
    tt_output_1 = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual_1 = ttnn.to_torch(tt_output_1)

    tt_output_2 = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    actual_2 = ttnn.to_torch(tt_output_2)

    # With high probability, two random runs should produce different results
    # (probability of identical results is vanishingly small for 32x32 random slopes)
    assert not torch.equal(
        actual_1, actual_2
    ), "Training mode should produce different outputs across runs due to random slopes"
