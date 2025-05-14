# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import torch_random, is_wormhole_b0, is_blackhole


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


def run_unary_with_approx_mode_test(device, h, w, ttnn_function, vector_mode, approx_mode, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, vector_mode=vector_mode, fast_and_approximate_mode=approx_mode)
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
        torch_output_tensor = golden_function(torch_input_tensor)

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
        torch_output_tensor = golden_function(torch_input_tensor)

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
        torch_output_tensor = golden_function(torch_input_tensor)

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
        torch_output_tensor = golden_function(torch_input_tensor)

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
        torch_output_tensor = golden_function(torch_input_tensor)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint8, ttnn.uint32, ttnn.int32, ttnn.float32])
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


@pytest.mark.parametrize("h", [0])
@pytest.mark.parametrize("w", [1])
def test_01_volume_sin(device, h, w):
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


@pytest.mark.parametrize("h", [2048 * 128])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("approx_mode", [True, False])
@pytest.mark.parametrize("vector_mode", [4])
def test_sigmoid(device, h, w, vector_mode, approx_mode):
    run_unary_with_approx_mode_test(
        device, h, w, ttnn.sigmoid, vector_mode=vector_mode, approx_mode=approx_mode, pcc=0.999
    )


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
    torch_output_tensor = golden_function(torch_input_tensor)

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
def test_floor(device, h, w):
    run_unary_test_range(device, h, w, ttnn.floor, pcc=0.99)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ceil(device, h, w):
    run_unary_test_range(device, h, w, ttnn.ceil, pcc=0.99)


def run_unary_test_with_float(device, h, w, scalar, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_unary_test_with_float_remainder(device, h, w, scalar, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

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
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.logit)
    torch_output_tensor = golden_function(torch_input_tensor_a, eps=scalar, device=device)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.logit(input_tensor_a, eps=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)


@pytest.mark.parametrize("scalar", [0, 1.0, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_pow(device, h, w, scalar):
    run_unary_test_with_float(device, h, w, scalar, ttnn.pow, pcc=0.999)


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
def test_remainder(device, h, w, scalar):
    run_unary_test_with_float_remainder(device, h, w, scalar, ttnn.remainder)


@pytest.mark.parametrize("scalar", [1.5, 2.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
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


def run_unary_test_bitwise_not(device, h, w, fill_value, ttnn_function):
    torch.manual_seed(0)

    torch_input_tensor = torch.full(size=(h, w), fill_value=fill_value).to(torch.int32)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("fill_value", [-2147483647, 2147483648, 7534, 225, 97, 3])
def test_bitwise_not(device, h, w, fill_value):
    run_unary_test_bitwise_not(device, h, w, fill_value, ttnn.bitwise_not)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_floor(input_shapes, device):
    in_data1 = torch.empty(input_shapes, dtype=torch.float32).uniform_(-43566, 43565)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.floor(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.floor)
    golden_tensor = golden_function(in_data1)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(golden_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_ceil(input_shapes, device):
    in_data1 = torch.empty(input_shapes, dtype=torch.float32).uniform_(-43566, 43565)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.ceil(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.ceil)
    golden_tensor = golden_function(in_data1)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(golden_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_alt_complex_rotate90(device, h: int, w: int, dtype: ttnn.DataType):
    ttnn_function = ttnn.alt_complex_rotate90
    golden_function = ttnn.get_golden_function(ttnn_function)

    torch.manual_seed(0)

    tt_input = ttnn.from_torch(torch.randn([h, w]), device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    torch_input = ttnn.to_torch(tt_input)

    torch_output = golden_function(torch_input, device=device)
    tt_output = ttnn_function(tt_input)

    assert torch.equal(torch_output, ttnn.to_torch(tt_output))


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 5),  # Small range
    ],
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eqz,
        ttnn.nez,
        ttnn.ltz,
        ttnn.lez,
        ttnn.gtz,
        ttnn.gez,
    ],
)
def test_unary_zero_comp_ttnn(input_shapes, low, high, ttnn_function, device):
    in_data = torch.randint(low, high, input_shapes, dtype=torch.int32)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    cq_id = 0
    output_tensor = ttnn_function(input_tensor, queue_id=cq_id)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data)

    output_tensor = ttnn.to_torch(output_tensor)
    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc == 1


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eqz,
        ttnn.nez,
        ttnn.ltz,
        ttnn.lez,
        ttnn.gtz,
        ttnn.gez,
    ],
)
def test_unary_zero_comp_edge_case(input_shapes, ttnn_function, device):
    torch.manual_seed(213919)

    # Generate a uniform range of values across the valid int32 range
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    uniform_values = torch.linspace(-2147483647, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor([0, 1, -1, 2147483647, -2147483647], dtype=torch.int32)
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data)

    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc == 1


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([])),
        (torch.Size([128])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", [-100, -54, -1, 0, 1, 13, 29])
@pytest.mark.parametrize("ttnn_op", [ttnn.ne, ttnn.eq])
@pytest.mark.parametrize("use_legacy", [True, False])
def test_unary_comp_ops(input_shapes, scalar, ttnn_op, use_legacy, device):
    torch.manual_seed(213919)

    # Generate a uniform range of values across the valid int32 range
    num_elements = int(torch.prod(torch.tensor(input_shapes)).item())
    uniform_values = torch.linspace(-2147483647, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor([0, 1, -1, 2147483647, -2147483647, -100, -54, 13, 29], dtype=torch.int32)
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    if is_wormhole_b0() and use_legacy == False and ((in_data - scalar) < -2147483647).any():
        pytest.xfail("Overflow occurs as in case of binary_ng, sub_tile is called")

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_op(input_tensor, scalar, use_legacy=use_legacy)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(in_data, scalar)

    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(golden_tensor, output_tensor)
