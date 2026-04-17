# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


LOWER = 0.125
UPPER = 1.0 / 3.0


def _bf16_roundtrip(t):
    """Round a float32 tensor to bfloat16 precision and back."""
    return t.to(torch.bfloat16).to(torch.float32)


# ---------- Eval mode tests ----------


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_eval_basic(device, h, w):
    """Eval mode: slope = (lower + upper) / 2, deterministic."""
    torch.manual_seed(0)
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)

    # PyTorch golden
    torch_output = torch.nn.functional.rrelu(
        torch_input.float(), lower=LOWER, upper=UPPER, training=False
    )

    # TTNN
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)

    # Also check allclose with specified tolerances
    torch.testing.assert_close(
        tt_output.float(), _bf16_roundtrip(torch_output), rtol=1.6e-2, atol=1e-2
    )


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_eval_positive_only(device, h, w):
    """Eval mode with all-positive input: output should equal input."""
    torch.manual_seed(42)
    torch_input = torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # For positive inputs, rrelu is identity
    torch.testing.assert_close(tt_output.float(), torch_input.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_eval_negative_only(device, h, w):
    """Eval mode with all-negative input: output = input * (lower+upper)/2."""
    torch.manual_seed(7)
    torch_input = -(torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01)

    slope = (LOWER + UPPER) / 2.0
    expected = _bf16_roundtrip(torch_input.float() * slope)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    torch.testing.assert_close(tt_output.float(), expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.0, 0.0),      # slope=0 for negatives (like ReLU)
        (1.0, 1.0),      # slope=1 for negatives (identity)
        (0.01, 0.99),     # wide range
        (0.125, 0.125),   # lower==upper, single fixed slope
    ],
)
def test_rrelu_eval_param_sweep(device, lower, upper):
    """Eval mode with various lower/upper combinations."""
    torch.manual_seed(123)
    h, w = 64, 128
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)

    torch_output = torch.nn.functional.rrelu(
        torch_input.float(), lower=lower, upper=upper, training=False
    )

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)


# ---------- Training mode tests ----------
# Note: Training mode currently uses deterministic midpoint slope ((lower+upper)/2)
# same as eval mode, because the SFPU PRNG hardware float generation has known
# limitations. These tests verify training mode produces correct results.


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_positive_passthrough(device, h, w):
    """Training mode: positive inputs should pass through unchanged."""
    torch.manual_seed(55)
    torch_input = torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # Positive values should be identity
    torch.testing.assert_close(tt_output.float(), torch_input.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_negative_scaled(device, h, w):
    """Training mode: negative inputs are scaled by midpoint slope = (lower+upper)/2."""
    torch.manual_seed(99)
    torch_input = -(torch.abs(torch.randn((h, w), dtype=torch.bfloat16)) + 0.01)

    slope = (LOWER + UPPER) / 2.0
    expected = _bf16_roundtrip(torch_input.float() * slope)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    torch.testing.assert_close(tt_output.float(), expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_mixed_input(device, h, w):
    """Training mode with mixed positive/negative: matches eval golden."""
    torch.manual_seed(77)
    torch_input = torch.randn((h, w), dtype=torch.bfloat16)

    torch_output = torch.nn.functional.rrelu(
        torch_input.float(), lower=LOWER, upper=UPPER, training=False
    )

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rrelu_training_slope_in_range(device, h, w):
    """Training mode: all slopes for negative inputs are in [lower, upper]."""
    torch.manual_seed(33)
    torch_input = -torch.ones((h, w), dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=LOWER, upper=UPPER, training=True)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    # For input = -1, output = -a, so a = -output
    slopes = -tt_output.float()

    # Check slopes are within [lower, upper] (with tolerance)
    assert slopes.min().item() >= LOWER - 1e-2, f"Min slope {slopes.min().item()} below lower={LOWER}"
    assert slopes.max().item() <= UPPER + 1e-2, f"Max slope {slopes.max().item()} above upper={UPPER}"
