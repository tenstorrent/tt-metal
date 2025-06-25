# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.neg,
    ],
)
def test_neg_fp32(device, ttnn_function):
    x_torch = torch.tensor([[0.00001]], dtype=torch.float32)
    y_torch = -x_torch

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sin,
    ],
)
def test_sin_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.sin(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.cos,
    ],
)
def test_cos_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.cos(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.tan,
    ],
)
def test_tan_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.tan(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.relu,
    ],
)
def test_relu_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.relu(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


def run_unary_test(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.float32)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp(device, h, w):
    run_unary_test(device, h, w, ttnn.exp, pcc=0.9998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanh(device, h, w):
    run_unary_test(device, h, w, ttnn.tanh, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_unary_test(device, h, w, ttnn.gelu, pcc=0.9996)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsqrt(device, h, w):
    run_unary_test(device, h, w, ttnn.rsqrt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_silu(device, h, w):
    run_unary_test(device, h, w, ttnn.silu)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log(device, h, w):
    run_unary_test(device, h, w, ttnn.log)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asin(device, h, w):
    run_unary_test(device, h, w, ttnn.asin, pcc=0.998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acos(device, h, w):
    run_unary_test(device, h, w, ttnn.acos, pcc=0.998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atan(device, h, w):
    run_unary_test(device, h, w, ttnn.atan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sinh(device, h, w):
    run_unary_test(device, h, w, ttnn.sinh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cosh(device, h, w):
    run_unary_test(device, h, w, ttnn.cosh, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acosh(device, h, w):
    run_unary_test(device, h, w, ttnn.acosh)


@pytest.mark.skip("The current version doesn’t work with float32, but this will be fixed in issue #231689.")
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atanh(device, h, w):
    run_unary_test(device, h, w, ttnn.atanh, pcc=0.997)
