# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_allclose


# ======================== Evaluation mode tests ========================


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_eval_default_params(device, shape, dtype):
    """RReLU eval mode with default lower=1/8, upper=1/3."""
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)

    lower, upper = 1.0 / 8.0, 1.0 / 3.0
    slope = (lower + upper) / 2.0
    torch_output = torch.where(torch_input >= 0, torch_input, torch_input * slope)
    if torch_dtype == torch.bfloat16:
        torch_output = torch_output.to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_threshold = 0.999 if dtype == ttnn.bfloat16 else 0.9999
    assert_with_pcc(torch_output, tt_output_torch, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "lower,upper",
    [
        (0.0, 1.0),
        (0.1, 0.3),
        (0.0, 0.0),     # zero slope for negatives (like ReLU)
        (0.5, 0.5),     # fixed slope = 0.5 (like leaky relu with 0.5)
        (0.01, 0.99),
    ],
)
def test_rrelu_eval_various_params(device, shape, dtype, lower, upper):
    """RReLU eval mode with various lower/upper parameter combinations."""
    torch.manual_seed(42)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)

    slope = (lower + upper) / 2.0
    torch_output = torch.where(torch_input >= 0, torch_input, torch_input * slope)
    if torch_dtype == torch.bfloat16:
        torch_output = torch_output.to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_threshold = 0.999 if dtype == ttnn.bfloat16 else 0.9999
    # Params pass through bfloat16 on device, so fp32 also has bf16-level precision for slopes
    atol = 1e-2 if dtype == ttnn.bfloat16 else 5e-3
    rtol = 1.6e-2 if dtype == ttnn.bfloat16 else 5e-3
    assert_with_pcc(torch_output, tt_output_torch, pcc=pcc_threshold)
    assert_allclose(torch_output, tt_output_torch, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_eval_positive_passthrough(device, shape, dtype):
    """Verify that positive values pass through unchanged in eval mode."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.abs(torch.randn(shape, dtype=torch_dtype)) + 0.01  # all positive

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, training=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    atol = 1e-2 if dtype == ttnn.bfloat16 else 1e-5
    assert_allclose(torch_input, tt_output_torch, rtol=0, atol=atol)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_eval_negative_scaling(device, shape, dtype):
    """Verify that negative values are scaled by (lower + upper) / 2 in eval mode."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = -(torch.abs(torch.randn(shape, dtype=torch_dtype)) + 0.01)  # all negative

    lower, upper = 0.125, 1.0 / 3.0
    slope = (lower + upper) / 2.0
    expected = torch_input * slope

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=False)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_threshold = 0.999 if dtype == ttnn.bfloat16 else 0.9999
    assert_with_pcc(expected, tt_output_torch, pcc=pcc_threshold)


# ======================== Training mode tests ========================


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_train_positive_passthrough(device, shape, dtype):
    """In training mode, positive values should still pass through unchanged."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.abs(torch.randn(shape, dtype=torch_dtype)) + 0.01

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, training=True)
    tt_output_torch = ttnn.to_torch(tt_output)

    atol = 1e-2 if dtype == ttnn.bfloat16 else 1e-5
    assert_allclose(torch_input, tt_output_torch, rtol=0, atol=atol)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_train_negative_bounds(device, shape, dtype):
    """In training mode, negative outputs should be within [lower*x, upper*x] bounds."""
    torch.manual_seed(123)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = -(torch.abs(torch.randn(shape, dtype=torch_dtype)) + 0.1)  # all negative

    lower, upper = 0.125, 1.0 / 3.0

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    tt_output_torch = ttnn.to_torch(tt_output)

    # For negative x, output = a * x where a in [lower, upper]
    # So output should be in [upper * x, lower * x] (since x < 0, multiplying by larger a gives more negative)
    torch_input_float = torch_input.float()
    lower_bound = upper * torch_input_float  # more negative (larger slope * negative)
    upper_bound = lower * torch_input_float  # less negative (smaller slope * negative)

    tt_float = tt_output_torch.float()
    atol = 0.1 if dtype == ttnn.bfloat16 else 0.01
    assert (tt_float >= lower_bound - atol).all(), "Output below lower bound"
    assert (tt_float <= upper_bound + atol).all(), "Output above upper bound"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 3, 64, 64],
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_train_randomness(device, shape, dtype):
    """Verify training mode produces different random slopes (not all identical)."""
    torch.manual_seed(7)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = -(torch.ones(shape, dtype=torch_dtype))  # all -1.0

    lower, upper = 0.0, 1.0

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.rrelu(tt_input, lower=lower, upper=upper, training=True)
    tt_output_torch = ttnn.to_torch(tt_output).float()

    # For input = -1.0, output = -a where a in [0, 1]
    # If random, we should see variation in the output
    unique_vals = tt_output_torch.unique().numel()
    assert unique_vals > 1, f"Expected varied random slopes, got {unique_vals} unique values"
