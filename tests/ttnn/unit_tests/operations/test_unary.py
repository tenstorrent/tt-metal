# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_grayskull


def run_unary_test(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_identity_test(device, h, w, data_type, pcc=0.9999):
    torch.manual_seed(0)

    int_format = data_type == ttnn.uint32 or data_type == ttnn.uint16
    if int_format:
        torch_input_tensor = torch.randint(0, 10000, (1, 1, h, w), dtype=torch.int32)
    else:
        torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
    if int_format:
        output_tensor = ttnn.experimental.tensor.identity_uint32(input_tensor)
    else:
        output_tensor = ttnn.experimental.tensor.identity(input_tensor)

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint32, ttnn.float32])
@skip_for_grayskull("Grayskull doesn't support uint32 / fp32 formats and fp32 dest")
def test_fp32_uint32(device, h, w, dtype):
    run_identity_test(device, h, w, dtype, pcc=0.9998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp(device, h, w):
    run_unary_test(device, h, w, ttnn.exp, torch.exp, pcc=0.9998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanh(device, h, w):
    run_unary_test(device, h, w, ttnn.tanh, torch.tanh, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_unary_test(device, h, w, ttnn.gelu, torch.nn.functional.gelu, pcc=0.9997)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu(device, h, w):
    run_unary_test(device, h, w, ttnn.relu, torch.relu)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsqrt(device, h, w):
    run_unary_test(device, h, w, ttnn.rsqrt, torch.rsqrt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_silu(device, h, w):
    run_unary_test(device, h, w, ttnn.silu, torch.nn.functional.silu)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log(device, h, w):
    run_unary_test(device, h, w, ttnn.log, torch.log)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sin(device, h, w):
    run_unary_test(device, h, w, ttnn.sin, torch.sin)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asin(device, h, w):
    run_unary_test(device, h, w, ttnn.asin, torch.asin, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cos(device, h, w):
    run_unary_test(device, h, w, ttnn.cos, torch.cos, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acos(device, h, w):
    run_unary_test(device, h, w, ttnn.acos, torch.acos, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tan(device, h, w):
    run_unary_test(device, h, w, ttnn.tan, torch.tan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atan(device, h, w):
    run_unary_test(device, h, w, ttnn.atan, torch.atan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sinh(device, h, w):
    run_unary_test(device, h, w, ttnn.sinh, torch.sinh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asinh(device, h, w):
    run_unary_test(device, h, w, ttnn.asinh, torch.asinh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cosh(device, h, w):
    run_unary_test(device, h, w, ttnn.cosh, torch.cosh, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acosh(device, h, w):
    run_unary_test(device, h, w, ttnn.acosh, torch.acosh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atanh(device, h, w):
    run_unary_test(device, h, w, ttnn.atanh, torch.atanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_not(device, h, w):
    run_unary_test(device, h, w, ttnn.logical_not, torch.logical_not)


def run_unary_test_range(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)
    low = -100
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_signbit(device, h, w):
    run_unary_test_range(device, h, w, ttnn.signbit, torch.signbit, pcc=0.99)


def run_unary_test_with_float(device, h, w, scalar, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor, scalar)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logit(device, h, w, scalar):
    run_unary_test_with_float(device, h, w, scalar, ttnn.logit, torch.logit)
