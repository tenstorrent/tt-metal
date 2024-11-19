# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sub,
    ],
)
def test_sub_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    z_torch = x_torch - y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_sub = ttnn.subtract(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_sub)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.rsub,
    ],
)
def test_rsub_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_sub = ttnn.rsub(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_sub)

    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
    ],
)
def test_add_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    z_torch = x_torch + y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_add)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.mul,
    ],
)
def test_mul_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[2]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    z_torch = x_torch * y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.mul(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status

    # currently failing as div sfpu tile is performing multiplication


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.div,
    ],
)
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0  inf; in fp32  tile
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0  0; in chained op
def test_div_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.00030171126, -3, 16, -5, 14, -12, 0]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, -4, -5, 0, 0, 0]], dtype=torch.float32)
    z_torch = x_torch / y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.divide(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)
    # print("inputs a, b", x_torch, "\n", y_torch)
    # print(z_torch, ttnn.to_torch(z_tt), tt_out)
    # print("torch out", z_torch, )
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)

    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


#     # status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
#     # assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.div,
    ],
)
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0  inf;
# Torch num/ 0 = inf and 0/0  nan; TT num/ 0 = inf and 0/0  0; in chained op
def test_div_bf16(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.00030171126, -3, 16, -5, 14, -12, 0]], dtype=torch.bfloat16)
    y_torch = torch.tensor([[2, 3, -4, -5, 0, 0, 0]], dtype=torch.bfloat16)
    z_torch = x_torch / y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.divide(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)
    # print("inputs a, b", x_torch, "\n", y_torch)
    # print(z_torch, ttnn.to_torch(z_tt), tt_out)
    # print("torch out", z_torch, )
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)

    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.pow,
    ],
)
def test_pow_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.55, 2.25]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3]], dtype=torch.float32)
    z_torch = torch.pow(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn.pow(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_pow)

    print("inputs a, b", x_torch, y_torch)
    print("torch out in ttnn", ttnn.to_torch(z_tt))
    print("tt out in torch", tt_out)

    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 80, 80],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
    ],
)
def test_add_fp32_activ(device, ttnn_function, shape):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32)
    y_torch = torch.ones([1, 1, 64, 64], dtype=torch.float32) * 4
    z_torch = torch.square(x_torch + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)])
    tt_out = ttnn.to_torch(z_tt_add)
    # print("inputs a, b", x_torch, y_torch)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
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
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.ones(shape, dtype=torch.float32) * 2
    y_torch = torch.ones(shape, dtype=torch.float32) * 4
    z_torch = torch.square(torch.nn.functional.silu(x_torch) + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.POWER, 2)],
        input_tensor_a_activation=ttnn.UnaryOpType.SILU,
    )
    tt_out = ttnn.to_torch(z_tt_add)
    # print("inputs a, b", x_torch, y_torch)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logaddexp,
    ],
)
def test_logaddexp_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    z_torch = torch.logaddexp(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.logaddexp(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logaddexp2,
    ],
)
def test_logaddexp2_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    z_torch = torch.logaddexp2(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.logaddexp2(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.ldexp,
    ],
)
def test_ldexp_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    z_torch = torch.ldexp(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.ldexp(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bias_gelu,
    ],
)
def test_bias_gelu_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2, 3, 4, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.bias_gelu(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.squared_difference,
    ],
)
def test_squared_difference_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.5, 2, 3.33, 4]], dtype=torch.float32)
    y_torch = torch.tensor([[2.009, 3.11, 4.22, 5]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.squared_difference(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logical_or,
    ],
)
def test_logical_or_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.509009, 2, 3.33, 4, 0, -11]], dtype=torch.float32)
    y_torch = torch.tensor([[0, 3, 4, 5, 0, -9999]], dtype=torch.float32)
    z_torch = torch.logical_or(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.logical_or(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logical_xor,
    ],
)
def test_logical_xor_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.509009, 2, 3.33, 4, 0, -11]], dtype=torch.float32)
    y_torch = torch.tensor([[0, 3, 4, 5, 0, -9999]], dtype=torch.float32)
    z_torch = torch.logical_xor(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.logical_xor(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logical_and,
    ],
)
def test_logical_and_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.509009, 2, 3.33, 4, 0, -11]], dtype=torch.float32)
    y_torch = torch.tensor([[0, 3, 4, 5, 0, -9999]], dtype=torch.float32)
    z_torch = torch.logical_and(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn.logical_and(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
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
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1.99999999991, 0, 345.1234568999130, -1]], dtype=torch.float32)
    y_torch = torch.tensor([[1.99999999990, 0, 345.1234568999131, -1]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_mul = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_mul)
    # print("torch out in ttnn", ttnn.to_torch(z_tt))
    # print("tt out in torch", tt_out)
    status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


# @skip_for_grayskull("Unsupported dtype for Grayskull")
# @pytest.mark.parametrize(
#     "ttnn_function",
#     [
#         ttnn.silu,
#     ],
# )
# @pytest.mark.parametrize(
#     "shape",
#     [
#         [1, 1, 16, 16],
#     ],
# )
# def test_silu_activ(device, ttnn_function, shape):
#     torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
#     x_torch = torch.ones(shape, dtype=torch.float32)
#     z_torch = torch.nn.functional.silu(x_torch)
#     x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
#     z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
#     z_tt_out = ttnn.silu(x_tt)
#     tt_out = ttnn.to_torch(z_tt_out)
#     # print("inputs a, b", x_torch, y_torch)
#     print("torch out in ttnn", ttnn.to_torch(z_tt)) # 0.731058597564697
#     print("tt out in torch", tt_out) # 0.728999972343445
#     # status = ttnn.ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
#     # assert status
#     status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
#     assert status
