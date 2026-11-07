# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose

pytestmark = pytest.mark.use_module_device


def test_add_int32(device):
    x_torch = torch.tensor([[11, 23, 0, -23, -1, -100]], dtype=torch.int32)
    y_torch = torch.tensor([[78, 99, 34, -33, -1, 100]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn.add)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_add)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


def test_pow_fp32(device):
    x_torch = torch.tensor([[1.55, 2.25, -3.6]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, -2.2]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn.pow(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_pow)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_hypot_multi_dtype(device, dtype):
    # Map ttnn dtypes to torch dtypes
    torch_dtype_map = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
    torch_dtype = torch_dtype_map[dtype]

    x_torch = torch.tensor([[3.0, 5.0, 3.14159, 0.5773, 1.41421, 2.71828, 0.001, 100.25]], dtype=torch_dtype)
    y_torch = torch.tensor([[4.0, 12.0, 2.71828, 1.73205, 1.73205, 3.14159, 999.999, 0.333]], dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.hypot)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_hypot = ttnn.hypot(x_tt, y_tt)

    if dtype == ttnn.float32:
        assert_allclose(z_torch, z_tt_hypot, rtol=1e-05, atol=1e-05)
    else:
        assert_with_ulp(expected_result=z_torch, actual_result=z_tt_hypot, ulp_threshold=1)


def test_add_fp32_activ(device):
    x_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32)
    y_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32) * 4
    z_torch = torch.square(x_torch + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt, activations=[ttnn.UnaryOpType.SQUARE])
    tt_out = ttnn.to_torch(z_tt_add)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 16, 16],
        [1, 1, 80, 80],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
def test_add_fp32_input_activ(device, shape):
    x_torch = torch.ones(shape, dtype=torch.float32) * 2
    y_torch = torch.ones(shape, dtype=torch.float32) * 4
    z_torch = torch.square(torch.nn.functional.silu(x_torch) + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        activations=[ttnn.UnaryOpType.SQUARE],
    )
    tt_out = ttnn.to_torch(z_tt_add)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


def test_logaddexp_fp32(device):
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.logaddexp(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


def test_logaddexp2_fp32(device):
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.logaddexp2)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.logaddexp2(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


def test_ldexp_fp32(device):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.ldexp)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.ldexp(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


def test_bias_gelu_fp32(device):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.bias_gelu)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.bias_gelu(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


def test_squared_difference_fp32(device):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2.009, 3.11, 4.22, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.squared_difference)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.squared_difference(x_tt, y_tt)

    assert_with_ulp(expected_result=z_torch, actual_result=z_tt_out, ulp_threshold=2)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
def test_bitwise(device, ttnn_function):
    x_torch = torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.int32)
    y_torch = torch.tensor([[9, 3, 0, 1, 7, 0]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    assert torch.equal(z_torch, tt_out)


def test_bitwise_left_shift(device):
    x_torch = torch.tensor([[99, 3, 100, 1, 72, 0, -100, 22, 12, 1000]], dtype=torch.int32)
    y_torch = torch.tensor([[1, 2, 31, 4, 5, 0, -20, 1, -3, -25]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn.bitwise_left_shift)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.bitwise_left_shift(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    assert torch.equal(z_torch, tt_out)


def test_bitwise_right_shift(device):
    x_torch = torch.tensor([[19, 3, 101, 21, 47, 0]], dtype=torch.int32)
    y_torch = torch.tensor([[5, 2, 31, 4, 5, 0]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn.bitwise_right_shift)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.bitwise_right_shift(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    assert torch.equal(z_torch, tt_out)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sub,
        ttnn.add,
        ttnn.rsub,
        ttnn.mul,
        ttnn.divide,
    ],
)
def test_ng_scalar_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = 0.00030171126
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = y_torch
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("input_shape", [torch.Size([1, 1, 32, 64]), torch.Size([1, 1, 32, 40])])
def test_bias_gelu_scalar_activations(input_shape, ttnn_dtype, device):
    x_torch = (torch.rand(input_shape) * 4.0 - 2.0).to(torch.float32)
    bias = -2.0

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # The reference has to see what the device saw, so use the host-quantised input.
    x_host = ttnn.to_torch(x_tt).float()
    rtol, atol = (1e-2, 1e-2) if ttnn_dtype == ttnn.bfloat16 else (1e-5, 1e-5)

    # No activation: control.
    assert_allclose(torch.nn.functional.gelu(x_host + bias), ttnn.bias_gelu(x_tt, bias), rtol=rtol, atol=atol)

    # activations=[relu] must be applied after the gelu.
    assert_allclose(
        torch.relu(torch.nn.functional.gelu(x_host + bias)),
        ttnn.bias_gelu(x_tt, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("input_shape", [torch.Size([1, 1, 32, 64]), torch.Size([1, 1, 32, 40])])
def test_bias_gelu_scalar_input_a_activations(input_shape, ttnn_dtype, device):
    x_torch = (torch.rand(input_shape) * 4.0 - 2.0).to(torch.float32)
    bias = -2.0

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # The reference has to see what the device saw, so use the host-quantised input.
    x_host = ttnn.to_torch(x_tt).float()
    rtol, atol = (1e-2, 1e-2) if ttnn_dtype == ttnn.bfloat16 else (1e-5, 1e-5)

    # No activation: control.
    assert_allclose(torch.nn.functional.gelu(x_host + bias), ttnn.bias_gelu(x_tt, bias), rtol=rtol, atol=atol)

    # input_tensor_a_activations=[sigmoid] must be applied to the operand before the add.
    assert_allclose(
        torch.nn.functional.gelu(torch.sigmoid(x_host) + bias),
        ttnn.bias_gelu(x_tt, bias, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)]),
        rtol=rtol,
        atol=atol,
    )
