# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import torch_random, skip_for_grayskull, skip_for_wormhole_b0


def run_unary_test(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_unary_test_fixed(device, h, w, fill_value, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.full((h, w), fill_value, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_identity_test(device, h, w, data_type, pcc=0.9999):
    torch.manual_seed(0)
    ttnn_function = ttnn.identity
    if data_type == ttnn.uint8:
        # init value
        torch_input_tensor = torch.randint(0, 245, (1, 1, h, w), dtype=torch.uint8)
        bias = 10

        # run torch
        torch_input_tensor = torch_input_tensor + bias
        golden_function = ttnn.get_golden_function(ttnn_function)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_equal(torch_output_tensor, output_tensor)
    elif data_type == ttnn.uint16:
        # init value
        torch_input_tensor = torch.randint(0, 60000, (1, 1, h, w), dtype=torch.uint16)
        bias = 2000

        # run torch
        torch_input_tensor = torch_input_tensor + bias
        golden_function = ttnn.get_golden_function(ttnn_function)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_equal(torch_output_tensor, output_tensor)

    elif data_type == ttnn.uint32:
        # init value
        torch_input_tensor = torch.randint(0, 2047483648, (1, 1, h, w), dtype=torch.int32)
        bias = 2000

        # run torch
        torch_input_tensor = torch_input_tensor + bias
        golden_function = ttnn.get_golden_function(ttnn_function)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_equal(torch_output_tensor, output_tensor)

    elif data_type == ttnn.int32:
        # init value
        torch_input_tensor = torch.randint(-2047483648, 2047483648, (1, 1, h, w), dtype=torch.int32)
        bias = 2000

        # run torch
        torch_input_tensor = torch_input_tensor + bias
        golden_function = ttnn.get_golden_function(ttnn_function)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_equal(torch_output_tensor, output_tensor)

    else:
        # init value
        torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)

        # run torch
        torch_input_tensor = torch_input_tensor
        golden_function = ttnn.get_golden_function(ttnn_function)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint8, ttnn.uint32, ttnn.int32, ttnn.float32])
@skip_for_grayskull("Grayskull doesn't support uint32 / fp32 formats and fp32 dest")
def test_fp32_uint32(device, h, w, dtype):
    run_identity_test(device, h, w, dtype, pcc=0.9998)


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
def test_relu(device, h, w):
    run_unary_test(device, h, w, ttnn.relu)


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
def test_sin(device, h, w):
    run_unary_test(device, h, w, ttnn.sin)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asin(device, h, w):
    run_unary_test(device, h, w, ttnn.asin, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cos(device, h, w):
    run_unary_test(device, h, w, ttnn.cos, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acos(device, h, w):
    run_unary_test(device, h, w, ttnn.acos, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tan(device, h, w):
    run_unary_test(device, h, w, ttnn.tan)


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
def test_asinh(device, h, w):
    run_unary_test(device, h, w, ttnn.asinh, pcc=0.9997)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cosh(device, h, w):
    run_unary_test(device, h, w, ttnn.cosh, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_wormhole_b0("Issue #6991: Failing on wormhole_b0 PCC issue")
def test_acosh(device, h, w):
    run_unary_test(device, h, w, ttnn.acosh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atanh(device, h, w):
    run_unary_test(device, h, w, ttnn.atanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_not(device, h, w):
    run_unary_test(device, h, w, ttnn.logical_not)


def run_unary_test_range(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)
    low = -100
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_signbit(device, h, w):
    run_unary_test_range(device, h, w, ttnn.signbit, pcc=0.99)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
def test_floor(device, h, w):
    run_unary_test_range(device, h, w, ttnn.floor, pcc=0.99)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
def test_ceil(device, h, w):
    run_unary_test_range(device, h, w, ttnn.ceil, pcc=0.99)


def run_unary_test_with_float(device, h, w, scalar, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, scalar)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_wormhole_b0("Issue #6991: Failing on wormhole_b0 PCC issue")
def test_logit(device, h, w, scalar):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.logit)
    torch_output_tensor = golden_function(torch_input_tensor_a, eps=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.logit(input_tensor_a, eps=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("scalar", [0, 1.0, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_pow(device, h, w, scalar):
    run_unary_test_with_float(device, h, w, scalar, ttnn.pow, pcc=0.9)


@pytest.mark.parametrize("lower_limit", [0, 1.0, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu_min(device, h, w, lower_limit):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.relu_min)
    torch_output_tensor = golden_function(torch_input_tensor, lower_limit=lower_limit)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.relu_min(input_tensor, lower_limit)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("upper_limit", [0, 1.0, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu_max(device, h, w, upper_limit):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.relu_max)
    torch_output_tensor = golden_function(torch_input_tensor, upper_limit=upper_limit)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.relu_max(input_tensor, upper_limit)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("scalar", [1.5, 2.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
def test_remainder(device, h, w, scalar):
    run_unary_test_with_float(device, h, w, scalar, ttnn.remainder)


@pytest.mark.parametrize("scalar", [1.5, 2.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_grayskull("Op not supported for Grayskull, supported for wormhole_b0")
def test_fmod(device, h, w, scalar):
    run_unary_test_with_float(device, h, w, scalar, ttnn.fmod)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asin_fixed(device, h, w):
    run_unary_test_fixed(device, h, w, 90, ttnn.asin, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acos_fixed(device, h, w):
    run_unary_test_fixed(device, h, w, 90, ttnn.acos, pcc=0.999)
