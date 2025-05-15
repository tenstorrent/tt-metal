# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sub,
        ttnn.rsub,
        ttnn.add,
    ],
)
def test_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_sub = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_sub)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
        ttnn.sub,
    ],
)
def test_int32(device, ttnn_function):
    x_torch = torch.tensor([[11, 23, 0, -23, -1, -100]], dtype=torch.int32)
    y_torch = torch.tensor([[78, 99, 34, -33, -1, 100]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_add)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.mul,
    ],
)
def test_mul_fp32(device, ttnn_function):
    x_torch = torch.tensor([[2]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.divide,
    ],
)
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0=nan; in fp32  tile
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0=0; in chained (mul * recip) div op
def test_div_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.00030171126, -3, 16, -5, 14, -12, 0, 0, 1, 15]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, -4, -5, 0, 0, 0, 1, 0, 10]], dtype=torch.float32)
    # torch out in ttnn TorchTensor([[ 0.500150859355927, -1.000000000000000, -4.000000000000000,  1.000000000000000,                inf,               -inf,                nan,  0.000000000000000,                inf,
    #            1.500000000000000]])
    # tt out in torch TorchTensor([[ 0.500150859355927, -1.000000000000000, -4.000000000000000,  1.000000000000000,                inf,               -inf,                nan,  0.000000000000000,                inf,
    #            1.499999880790710]])
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.divide,
    ],
)
# Torch: num/ 0 = inf and 0/0  nan;
# TT: num/ 0 = inf but 0/0= 0 not nan and 1/0 is 170141183460469231731687303715884105728.000000000000000 not inf;
# input_b must be non-zero
def test_div_bf16(device, ttnn_function):
    x_torch = torch.tensor(
        [
            [
                1.00030171126,
                -3,
                16,
                -5,
                14,
                -12,
                0,
                0,
                15,
            ]
        ],
        dtype=torch.bfloat16,
    )
    y_torch = torch.tensor(
        [
            [
                2,
                3,
                -4,
                -5,
                0,
                0,
                0,
                1,
                10,
            ]
        ],
        dtype=torch.bfloat16,
    )
    # torch out in ttnn TorchTensor([[ 0.500000000000000, -1.000000000000000, -4.000000000000000,  1.000000000000000,                inf,               -inf,                nan,  0.000000000000000,  1.500000000000000]],
    #         dtype=torch.bfloat16)
    # tt out in torch TorchTensor([[ 0.500000000000000, -1.000000000000000, -4.000000000000000,  1.000000000000000,                inf,               -inf,  0.000000000000000,  0.000000000000000,  1.500000000000000]],
    #         dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn_function(x_tt, y_tt)  # bf16 runs FPU
    tt_out = ttnn.to_torch(z_tt_div)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.pow,
    ],
)
def test_pow_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.55, 2.25, -3.6]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, -2.2]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_pow)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


def test_squared_sum_fp32_activ(device):
    x_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32)
    y_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32) * 4
    z_torch = torch.square(x_torch + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)])
    tt_out = ttnn.to_torch(z_tt_add)

    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 16, 16],
        [1, 1, 80, 80],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
def test_add_fp32_input_activ(device, ttnn_function, shape):
    x_torch = torch.ones(shape, dtype=torch.float32) * 2
    y_torch = torch.ones(shape, dtype=torch.float32) * 4
    z_torch = torch.square(torch.nn.functional.silu(x_torch) + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)],
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    )
    tt_out = ttnn.to_torch(z_tt_add)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logaddexp,
    ],
)
def test_logaddexp_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.logaddexp(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logaddexp2,
    ],
)
def test_logaddexp2_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.logaddexp2(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.ldexp,
    ],
)
def test_ldexp_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bias_gelu,
    ],
)
def test_bias_gelu_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.squared_difference,
    ],
)
def test_squared_difference_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2.009, 3.11, 4.22, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
    ],
)
def test_logical_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.509009, 2, 3.33, 4, 0, -11]], dtype=torch.float32)
    y_torch = torch.tensor([[0, 3, 4, 5, 0, -9999]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.gt,
        ttnn.ge,
        ttnn.lt,
        ttnn.le,
    ],
)
def test_relational_fp32(device, ttnn_function):
    x_torch = torch.tensor([[1.99999999991, 0, 345.1234568999130, -1]], dtype=torch.float32)
    y_torch = torch.tensor([[1.99999999990, 0, 345.1234568999131, -1]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


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

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_left_shift,
    ],
)
def test_bitwise_left_shift(device, ttnn_function):
    x_torch = torch.tensor([[99, 3, 100, 1, 72, 0, -100, 22, 12, 1000]], dtype=torch.int32)
    y_torch = torch.tensor([[1, 2, 31, 4, 5, 0, -20, 1, -3, -25]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_right_shift,
    ],
)
def test_bitwise_right_shift(device, ttnn_function):
    x_torch = torch.tensor([[19, 3, 101, 21, 47, 0]], dtype=torch.int32)
    y_torch = torch.tensor([[5, 2, 31, 4, 5, 0]], dtype=torch.int32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status
