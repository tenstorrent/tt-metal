# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (1, 3, 128, 128),
    ],
)
@pytest.mark.parametrize(
    "lower, upper",
    [
        (0.125, 1.0 / 3.0),
        (0.0, 0.5),
        (0.1, 0.3),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_eval(device, shape, lower, upper, dtype):
    """Test RReLU in eval mode (training=False). Output is deterministic: slope = (lower+upper)/2."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)

    # PyTorch reference in eval mode
    slope = (lower + upper) / 2.0
    torch_output = torch.where(torch_input >= 0, torch_input, torch_input * slope)

    tt_dtype = dtype
    x_tt = ttnn.from_torch(torch_input, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.rrelu(x_tt, lower=lower, upper=upper, training=False)
    tt_out = ttnn.to_torch(y_tt)

    assert_with_pcc(torch_output, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_eval_default_params(device, shape, dtype):
    """Test RReLU in eval mode with default parameters (lower=1/8, upper=1/3)."""
    torch.manual_seed(123)
    torch_input = torch.randn(shape, dtype=torch.float32)

    lower, upper = 0.125, 1.0 / 3.0
    slope = (lower + upper) / 2.0
    torch_output = torch.where(torch_input >= 0, torch_input, torch_input * slope)

    x_tt = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.rrelu(x_tt)  # all defaults
    tt_out = ttnn.to_torch(y_tt)

    assert_with_pcc(torch_output, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_training(device, shape, dtype):
    """Test RReLU in training mode. Verify outputs are in valid range for negative inputs."""
    torch.manual_seed(99)
    torch_input = torch.randn(shape, dtype=torch.float32)

    lower, upper = 0.125, 1.0 / 3.0

    x_tt = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.rrelu(x_tt, lower=lower, upper=upper, training=True)
    tt_out = ttnn.to_torch(y_tt).float()

    # For x >= 0: output should equal x
    pos_mask = torch_input >= 0
    if pos_mask.any():
        # Reconvert input through bfloat16 round-trip for fair comparison
        x_roundtrip = ttnn.to_torch(
            ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ).float()
        pos_close = torch.allclose(tt_out[pos_mask], x_roundtrip[pos_mask], atol=0.05, rtol=0.05)
        assert pos_close, "For x >= 0, output should equal input"

    # For x < 0: output should be in range [upper * x, lower * x] (note: x < 0, so upper*x < lower*x)
    neg_mask = torch_input < 0
    if neg_mask.any():
        neg_x = tt_out[neg_mask]
        x_neg = torch_input[neg_mask]
        # For negative x: slope * x where slope in [lower, upper]
        # Since x < 0: upper * x <= output <= lower * x
        lower_bound = upper * x_neg - 0.05  # small tolerance
        upper_bound = lower * x_neg + 0.05
        in_range = (neg_x >= lower_bound).all() and (neg_x <= upper_bound).all()
        assert in_range, f"For x < 0, output should be in [upper*x, lower*x] range"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_positive_passthrough(device, dtype):
    """Test that positive values pass through unchanged."""
    torch_input = torch.abs(torch.randn(1, 1, 32, 32, dtype=torch.float32)) + 0.01

    x_tt = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.rrelu(x_tt, lower=0.1, upper=0.3, training=False)
    tt_out = ttnn.to_torch(y_tt)

    assert_with_pcc(torch_input, tt_out, pcc=0.999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rrelu_zero_slope(device, dtype):
    """Test RReLU with lower=0, upper=0 (equivalent to ReLU)."""
    torch.manual_seed(42)
    torch_input = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    torch_output = torch.relu(torch_input)

    x_tt = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.rrelu(x_tt, lower=0.0, upper=0.0, training=False)
    tt_out = ttnn.to_torch(y_tt)

    assert_with_pcc(torch_output, tt_out, pcc=0.999)
