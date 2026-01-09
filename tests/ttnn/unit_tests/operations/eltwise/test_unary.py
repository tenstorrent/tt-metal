# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, assert_with_ulp, assert_allclose
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_pcc,
)
from models.common.utility_functions import torch_random, is_wormhole_b0, is_blackhole


def create_full_range_tensor(input_shapes, dtype):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()

    large_negatives = torch.linspace(-3.3e38, -1e30, steps=num_elements // 5, dtype=dtype)
    medium_negatives = torch.linspace(-1e5, -1e-3, steps=num_elements // 5, dtype=dtype)
    near_zero = torch.linspace(-1e-5, 1e-5, steps=num_elements // 5, dtype=dtype)
    medium_positives = torch.linspace(1e-3, 1e5, steps=num_elements // 5, dtype=dtype)
    large_positives = torch.linspace(1e30, 3.3e38, steps=num_elements // 5, dtype=dtype)

    in_data = torch.cat([large_negatives, medium_negatives, near_zero, medium_positives, large_positives])

    corner_cases = torch.tensor([0.0], dtype=dtype)
    in_data = torch.cat([in_data, corner_cases])

    in_data = in_data[:num_elements]
    if in_data.numel() < num_elements:  # add some random noise to the tensor to make it full range
        in_data = torch.cat([in_data, torch.randn(num_elements - in_data.numel(), dtype=dtype)])
    in_data = in_data.reshape(input_shapes)

    return in_data


def run_unary_test(device, h, w, ttnn_function, layout=ttnn.TILE_LAYOUT, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_unary_with_approx_mode_test(
    device, h, w, ttnn_function, vector_mode, approx_mode, layout=ttnn.TILE_LAYOUT, pcc=0.9999
):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor, vector_mode=vector_mode, fast_and_approximate_mode=approx_mode)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_unary_test_fixed(device, h, w, fill_value, ttnn_function, layout=ttnn.TILE_LAYOUT, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.full((h, w), fill_value, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_identity_test(device, h, w, data_type):
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

    elif data_type == ttnn.bfloat16:
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
        assert_equal(torch_output_tensor, output_tensor)

    else:
        # init value
        torch_input_tensor = torch.rand((1, 1, h, w))

        # run torch
        torch_input_tensor = torch_input_tensor
        golden_function = ttnn.get_golden_function(ttnn_function)
        torch_output_tensor = golden_function(torch_input_tensor)

        # run tt
        input_tensor = ttnn.from_torch(torch_input_tensor, data_type, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.identity(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        # compare result
        assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint8, ttnn.uint32, ttnn.int32, ttnn.float32])
def test_fp32_uint32(device, h, w, dtype):
    run_identity_test(device, h, w, dtype)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.exp, layout=layout, pcc=0.9998)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.gelu, layout=layout, pcc=0.9996)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.relu, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_silu(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.silu, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.log, layout=layout)


def test_log_edge_cases(device):
    in_data = torch.tensor(
        [-10.0, 0.0, -float("inf"), +float("inf"), +float("nan"), -float("nan")], dtype=torch.float32
    )
    input_tensor = ttnn.from_torch(in_data, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.log(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.log)
    golden_tensor = golden_function(in_data, device=device)
    assert torch.allclose(ttnn.to_torch(output_tensor), golden_tensor, equal_nan=True)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([5, 3, 145, 72])),
        (torch.Size([4, 9, 52, 182])),
    ),
)
@pytest.mark.parametrize(
    "log_function, torch_dtype, ttnn_dtype, low_range, high_range",
    [
        # for negative input values torch output is nan
        # for ttnn bfloat8_b due to shared exponent inf/nan values will result in incorrect results due to flushing
        (ttnn.log, torch.bfloat16, ttnn.bfloat8_b, 1, 100),
        # Hence ignoring the negative input for bfloat8_b
        (ttnn.log2, torch.bfloat16, ttnn.bfloat8_b, 1, 100),
        (ttnn.log10, torch.bfloat16, ttnn.bfloat8_b, 1, 100),
        # for ttnn bfloat16 nan is packed as inf (doesn't match with torch behavior).
        # hence ignoring the negative input for bfloat16 as well
        (ttnn.log, torch.bfloat16, ttnn.bfloat16, 1, 100),
        (ttnn.log2, torch.bfloat16, ttnn.bfloat16, 1, 100),
        (ttnn.log10, torch.bfloat16, ttnn.bfloat16, 1, 100),
        # TODO: add float32 once https://github.com/tenstorrent/tt-metal/pull/26675 is merged
    ],
)
@pytest.mark.parametrize(
    "data_seed",
    [4171614],
)
# Related to issue 8634 for log based functions with different dtypes
def test_unary_log_operations_ttnn(
    input_shapes, log_function, torch_dtype, ttnn_dtype, low_range, high_range, data_seed, device
):
    """Test logarithm functions (log, log2, log10)"""
    torch.manual_seed(data_seed)
    in_data = torch.Tensor(size=input_shapes).uniform_(low_range, high_range).to(torch_dtype)

    # Only use pad_value=1.0 for bfloat8_b to avoid log(0) issues with shared exponent flushing
    tensor_kwargs = {
        "dtype": ttnn_dtype,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    if ttnn_dtype == ttnn.bfloat8_b:
        tensor_kwargs["pad_value"] = 1.0

    input_tensor = ttnn.from_torch(in_data, **tensor_kwargs)
    output_tensor = log_function(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    golden_function = ttnn.get_golden_function(log_function)
    # for bfloat8_b precision
    input_torch_converted = ttnn.to_torch(input_tensor)
    golden_tensor = golden_function(input_torch_converted, device=device)
    tt_result = ttnn.to_torch(output_tensor)

    assert_with_pcc(tt_result, golden_tensor, pcc=0.99)
    assert torch.allclose(tt_result, golden_tensor, rtol=4e-2, atol=4e-2)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sin(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.sin, layout=layout)


@pytest.mark.parametrize("h", [0])
@pytest.mark.parametrize("w", [1])
def test_01_volume_sin(device, h, w):
    run_unary_test(device, h, w, ttnn.sin)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asin(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.asin, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cos(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.cos, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acos(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.acos, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tan(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.tan, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atan(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.atan, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sinh(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.sinh, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [2048 * 128])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("approx_mode", [True, False])
@pytest.mark.parametrize("vector_mode", [4])
def test_sigmoid(device, h, w, vector_mode, approx_mode, layout):
    run_unary_with_approx_mode_test(
        device, h, w, ttnn.sigmoid, vector_mode=vector_mode, approx_mode=approx_mode, layout=layout, pcc=0.999
    )


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_not(device, h, w, layout):
    run_unary_test(device, h, w, ttnn.logical_not, layout=layout)


def run_unary_test_range(device, h, w, ttnn_function, layout=ttnn.TILE_LAYOUT, pcc=0.9999):
    torch.manual_seed(0)
    low = -100
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_floor(device, h, w, layout):
    run_unary_test_range(device, h, w, ttnn.floor, layout=layout, pcc=0.99)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ceil(device, h, w, layout):
    run_unary_test_range(device, h, w, ttnn.ceil, layout=layout, pcc=0.99)


def run_unary_test_with_float(device, h, w, scalar, ttnn_function, layout=ttnn.TILE_LAYOUT, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
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


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("scalar", [0, 1.0, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_pow(device, h, w, scalar, layout):
    run_unary_test_with_float(device, h, w, scalar, ttnn.pow, layout=layout, pcc=0.999)


@pytest.mark.parametrize("lower_limit", [0, 1.0, 2, -5.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_relu_min(device, h, w, lower_limit, dtype):
    torch.manual_seed(0)

    if dtype == ttnn.bfloat16:
        torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    elif dtype == ttnn.int32:
        torch_input_tensor = torch.randint(
            torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (h, w), dtype=torch.int32
        )
        lower_limit = int(lower_limit)

    golden_function = ttnn.get_golden_function(ttnn.relu_min)
    torch_output_tensor = golden_function(torch_input_tensor, lower_limit=lower_limit)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.relu_min(input_tensor, lower_limit)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("upper_limit", [0, 1.0, 2, -5.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_relu_max(device, h, w, upper_limit, dtype):
    torch.manual_seed(0)

    if dtype == ttnn.bfloat16:
        torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    elif dtype == ttnn.int32:
        torch_input_tensor = torch.randint(
            torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (h, w), dtype=torch.int32
        )
        upper_limit = int(upper_limit)

    golden_function = ttnn.get_golden_function(ttnn.relu_max)
    torch_output_tensor = golden_function(torch_input_tensor, upper_limit=upper_limit)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.relu_max(input_tensor, upper_limit)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(torch_output_tensor, output_tensor)


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

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "low, high, torch_dtype, ttnn_dtype, input_shapes",
    [
        (0, 2, torch.int32, ttnn.uint16, torch.Size([32, 32])),  # Small range
        (0, 65535, torch.int32, ttnn.uint16, torch.Size([1, 3, 320, 384])),  # Full uint16 range
        (0, 2, torch.uint32, ttnn.uint32, torch.Size([32, 32])),  # Small range
        (0, 4294967295, torch.uint32, ttnn.uint32, torch.Size([1, 3, 320, 384])),  # Full uint32 range
    ],
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eqz,
        ttnn.nez,
    ],
)
def test_unary_zero_comp_uint_ttnn(input_shapes, low, high, torch_dtype, ttnn_dtype, ttnn_function, device):
    in_data = torch.randint(low, high, input_shapes, dtype=torch_dtype)
    zeroize_prob = 0.50
    if zeroize_prob > 0:
        zero_mask = torch.rand(in_data.shape) < zeroize_prob
        in_data = in_data.to(torch.int64) if ttnn_dtype == ttnn.uint32 else in_data
        in_data = in_data.masked_fill(zero_mask, 0)
        in_data = in_data.to(torch.uint32) if ttnn_dtype == ttnn.uint32 else in_data
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cq_id = 0
    output_tensor = ttnn_function(input_tensor, queue_id=cq_id)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data)

    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([64, 64])),
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
    # Generate a uniform range of values across the valid int32 range
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    uniform_values = torch.linspace(-2147483647, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor([0, 1, -1, 2147483647, -2147483647], dtype=torch.int32)
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)
    # Ensure zeros appear in every tile/page: zero out every k-th element
    k = 64
    flat = in_data.view(-1)
    flat[::k] = 0
    in_data = flat.view(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data)

    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(golden_tensor, output_tensor)


def is_int32_overflow(tensor, scalar):
    result = tensor.to(torch.int64) - scalar
    return (result < -(2**31) + 1) | (result > 2**31 - 1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([64, 64])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", [-54, -1, 0, 1, 13, -0])
@pytest.mark.parametrize("ttnn_op", [ttnn.ne, ttnn.eq, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le])
@pytest.mark.parametrize("use_legacy", [False])
# TODO: Test use_legacy = True for all cases after #23179 is completed
def test_unary_comp_ops(input_shapes, scalar, ttnn_op, use_legacy, device):
    # Generate a uniform range of values across the valid int32 range
    num_elements = int(torch.prod(torch.tensor(input_shapes)).item())
    uniform_values = torch.linspace(-2147483647, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor([0, -0, 1, -1, 2147483647, -2147483647, -100, -54, 13, 29], dtype=torch.int32)
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    if use_legacy == False and is_int32_overflow(in_data, scalar).any():
        pytest.xfail("Overflow occurs as in case of binary_ng, sub_tile is called")

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_op(input_tensor, scalar, use_legacy=use_legacy)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(in_data, scalar)

    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.016),
        (torch.bfloat16, ttnn.bfloat16, 0.012),
    ],
)
def test_unary_tanhshrink_ttnn(input_shapes, torch_dtype, ttnn_dtype, atol, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.tanhshrink(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.tanhshrink)
    golden_tensor = golden_function(in_data1)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_unary_tanhshrink_approx_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.tanhshrink(input_tensor1, fast_and_approximate_mode=True)
    golden_function = ttnn.get_golden_function(ttnn.tanhshrink)
    golden_tensor = golden_function(in_data1)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=0.25)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([2, 1024, 1024])),),
)
@pytest.mark.parametrize(
    "ttnn_function",
    [ttnn.silu, ttnn.asinh, ttnn.tanhshrink, ttnn.rad2deg, ttnn.deg2rad, ttnn.acosh, ttnn.hardsigmoid, ttnn.cbrt],
)
def test_unary_edge_case_ttnn(input_shapes, ttnn_function, device):
    in_data = create_full_range_tensor(input_shapes, torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data)

    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("ttnn_function", [ttnn.rad2deg, ttnn.deg2rad])
def test_unary_angle_conversion_ttnn(input_shapes, device, ttnn_dtype, ttnn_function):
    in_data1, input_tensor1 = data_gen_with_range_dtype(input_shapes, -100, 100, device, ttnn_dtype=ttnn_dtype)

    output_tensor = ttnn_function(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1)
    output = ttnn.to_torch(output_tensor)
    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass and assert_with_ulp(golden_tensor, output)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_trunc_ttnn(input_shapes, device):
    in_data = create_full_range_tensor(input_shapes, torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.trunc(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.trunc)
    golden_tensor = golden_function(in_data)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 128])),),
)
def test_unary_trunc_ttnn_opt(input_shapes, device):
    in_data = create_full_range_tensor(input_shapes, torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    cq_id = 0
    ttnn.trunc(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_function = ttnn.get_golden_function(ttnn.trunc)
    golden_tensor = golden_function(in_data)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.011),
        (torch.bfloat16, ttnn.bfloat16, 0.032),
        (torch.bfloat16, ttnn.bfloat8_b, 0.3),
    ],
)
@pytest.mark.parametrize("ttnn_function", [ttnn.silu, ttnn.swish])
def test_unary_silu_swish_ttnn(input_shapes, torch_dtype, ttnn_dtype, ttnn_function, device, atol):
    torch.manual_seed(0)
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn_function(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1, device=device)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)


@pytest.mark.parametrize("ttnn_function", [ttnn.silu, ttnn.swish])
def test_unary_silu_swish_threshold(ttnn_function, device):
    in_data1 = torch.tensor([[-1.0, 0.0, 0.5, 1.0, 1.5, 3.5, 5.0, 5.2, 5.5]], dtype=torch.bfloat16)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1, device=device)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=0.032)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, low, high",
    [
        (torch.float32, ttnn.float32, 1, 100),
        (torch.bfloat16, ttnn.bfloat16, -100, 1),
        (torch.float32, ttnn.float32, -100, 1),
        (torch.bfloat16, ttnn.bfloat8_b, -100, 1),
        (torch.bfloat16, ttnn.bfloat8_b, 1, 100),
    ],
)
@pytest.mark.parametrize("ttnn_function", [ttnn.acosh, ttnn.asinh])
def test_unary_inverse_hyperbolic_edge_case_ttnn(
    input_shapes, device, torch_dtype, ttnn_dtype, low, high, ttnn_function
):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(low, high)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)
    output_tensor = ttnn_function(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1, device=device)

    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_acosh_ttnn(input_shapes, device):
    in_data1 = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(1, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.acosh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.acosh)
    golden_tensor = golden_function(in_data1, device=device)
    output = ttnn.to_torch(output_tensor)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_unary_asinh_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.asinh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.asinh)
    golden_tensor = golden_function(in_data1, device=device)

    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [(-0.9, 1), (-100, 100)],
)
def test_unary_atanh_ttnn(input_shapes, torch_dtype, ttnn_dtype, low, high, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(low, high)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)
    output_tensor = ttnn.atanh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.atanh)
    golden_tensor = golden_function(in_data1)

    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "param",
    {0.65, 7.7, 36.49, 58.6, 97.2},
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.hardshrink,
        ttnn.softshrink,
    ],
)
def test_unary_shrink_functions_ttnn(input_shapes, param, torch_dtype, ttnn_dtype, ttnn_function, device):
    in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor, lambd=param)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data, lambd=param)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "param",
    {7.0, 36.49, 58.5, 97.2},
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.hardshrink,
        ttnn.softshrink,
    ],
)
def test_unary_shrink_functions_bf8b_ttnn(input_shapes, param, ttnn_function, device):
    in_data = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-100, 100)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    in_data = ttnn.to_torch(input_tensor, dtype=torch.bfloat16)

    output_tensor = ttnn_function(input_tensor, lambd=param)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data, lambd=param)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=0.02)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([2, 1024, 1024])),),
)
@pytest.mark.parametrize(
    "param",
    {0.45, 7.7, 197.2, 1e5},
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.hardshrink,
        ttnn.softshrink,
    ],
)
def test_unary_shrink_functions_edge_case_ttnn(input_shapes, param, ttnn_function, device):
    in_data = create_full_range_tensor(input_shapes, torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, lambd=param)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data, lambd=param)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_frac_ttnn(input_shapes, device):
    in_data = create_full_range_tensor(input_shapes, torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.frac(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.frac)
    golden_tensor = golden_function(in_data)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 128])),),
)
def test_unary_frac_ttnn_opt(input_shapes, device):
    in_data = create_full_range_tensor(input_shapes, torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    cq_id = 0
    ttnn.frac(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_function = ttnn.get_golden_function(ttnn.frac)
    golden_tensor = golden_function(in_data)

    assert_with_ulp(output_tensor, golden_tensor)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.0015),
        (torch.bfloat16, ttnn.bfloat16, 0.004),
        (torch.bfloat16, ttnn.bfloat8_b, 0.004),
    ],
)
def test_unary_softsign_ttnn(input_shapes, torch_dtype, ttnn_dtype, atol, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.softsign(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.softsign)
    golden_tensor = golden_function(in_data1, device=device)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.0004),
        (torch.bfloat16, ttnn.bfloat16, 0.004),
        (torch.bfloat16, ttnn.bfloat8_b, 0.015),
    ],
)
def test_unary_hardsigmoid_ttnn(input_shapes, torch_dtype, ttnn_dtype, atol, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.hardsigmoid(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardsigmoid)
    golden_tensor = golden_function(in_data1, device=device)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("low, high", [(-3, 3), (-100, 100)])  # computation range
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.0012),
        (torch.bfloat16, ttnn.bfloat16, 0.016),
    ],
)
def test_unary_hardswish_ttnn(input_shapes, low, high, torch_dtype, ttnn_dtype, atol, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(low, high)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.hardswish(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardswish)
    golden_tensor = golden_function(in_data1, device=device)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("low, high", [(-100, -3), (-2, 2), (3, 100)])
def test_unary_hardswish_bf8b_ttnn(input_shapes, low, high, device):
    in_data1 = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(low, high)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    in_data1 = ttnn.to_torch(input_tensor1, dtype=torch.bfloat16)

    output_tensor = ttnn.hardswish(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardswish)
    golden_tensor = golden_function(in_data1, device=device)

    assert_allclose(output_tensor, golden_tensor, atol=0.025)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (-2.0, 2.0),
        (-26.5, 33.6),
        (-0.5, 21.0),
    ],
)
def test_unary_hardtanh_ttnn(input_shapes, torch_dtype, ttnn_dtype, min_val, max_val, device):
    torch.manual_seed(0)
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.hardtanh(input_tensor1, min_val=min_val, max_val=max_val)
    golden_function = ttnn.get_golden_function(ttnn.hardtanh)
    golden_tensor = golden_function(in_data1, min_val=min_val, max_val=max_val)

    assert_equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.int32, ttnn.int32),
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_unary_signbit_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    if torch_dtype == torch.int32:
        in_data = torch.randint(-100, 100, input_shapes, dtype=torch_dtype)
    else:
        in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data = ttnn.to_torch(input_tensor, dtype=torch_dtype)

    output_tensor = ttnn.signbit(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.signbit)
    golden_tensor = golden_function(in_data)

    assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


def test_unary_signbit_int32_edge_case_ttnn(device):
    in_data = torch.tensor([-2147483648, 2147483647, +0, -0, 0], dtype=torch.int32)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.signbit(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.signbit)
    golden_tensor = golden_function(in_data)

    assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
def test_unary_signbit_float_edge_case_ttnn(torch_dtype, ttnn_dtype, device):
    in_data = torch.tensor(
        [-0.0, 0.0, +0.0, -float("inf"), +float("inf"), +float("nan"), -float("nan")], dtype=torch_dtype
    )
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.signbit(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.signbit)
    golden_tensor = golden_function(in_data)

    assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("threshold", [1.0, 10.0, 100.0, -5, -8.0, -100.0])
@pytest.mark.parametrize("value", [10.0, 100.0, -7.0, -85.5])
def test_unary_threshold_ttnn(input_shapes, threshold, value, device):
    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.threshold(input_tensor1, threshold, value)
    golden_function = ttnn.get_golden_function(ttnn.threshold)
    golden_tensor = golden_function(in_data1, threshold, value)

    assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (None, None),
        (-29.4, None),
        (None, 18.0),
        (-10.5, 34.5),
        (12.5, 82.5),
        (1, -1),
        (0, 0),
        (0, 1),
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
def test_unary_clamp_tss_float_ttnn(input_shapes, min_val, max_val, torch_dtype, ttnn_dtype, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-10, 10)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    min = min_val
    max = max_val
    if min is None and max is None:
        with pytest.raises(RuntimeError, match="Only one of 'min' or 'max' can be None. Please provide one value"):
            ttnn.clamp(input_tensor1, min, max)
    else:
        output_tensor = ttnn.clamp(input_tensor1, min, max)
        golden_function = ttnn.get_golden_function(ttnn.clamp)
        golden_tensor = golden_function(in_data1, min, max)
        assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))
        assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.001),
        (torch.bfloat16, ttnn.bfloat16, 0.008),
    ],
)
def test_unary_tanh_ttnn(input_shapes, torch_dtype, ttnn_dtype, atol, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-10, 10)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.tanh(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data1)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_unary_tanh_approx_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    in_data1 = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-10, 10)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data1 = ttnn.to_torch(input_tensor1, dtype=torch_dtype)

    output_tensor = ttnn.tanh(input_tensor1, fast_and_approximate_mode=True)
    golden_function = ttnn.get_golden_function(ttnn.tanh)
    golden_tensor = golden_function(in_data1)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=0.15)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_square_uint16_ttnn(input_shapes, device):
    in_data = torch.randint(
        0, 255, input_shapes, dtype=torch.int32
    )  # Beyond 255 leads to overflow of uint16 range, since it a square op.
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    cq_id = 0
    output_tensor = ttnn.square(input_tensor, queue_id=cq_id)
    golden_function = ttnn.get_golden_function(ttnn.square)
    golden_tensor = golden_function(in_data)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "min_val, max_val",
    [
        (None, None),
        (-29, None),
        (None, 18),
        (-10, 34),
        (12, 82),
        (1, -1),
        (0, 0),
        (0, 1),
    ],
)
def test_unary_clamp_tss_int32_ttnn(input_shapes, min_val, max_val, device):
    torch.manual_seed(0)
    in_data1 = torch.randint(-100, 100, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    min = min_val
    max = max_val
    if min is None and max is None:
        with pytest.raises(RuntimeError, match="Only one of 'min' or 'max' can be None. Please provide one value"):
            ttnn.clamp(input_tensor1, min, max)
    else:
        output_tensor = ttnn.clamp(input_tensor1, min, max)
        golden_function = ttnn.get_golden_function(ttnn.clamp)
        golden_tensor = golden_function(in_data1, min, max)
        assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_unary_cosh_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-9, 9)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data = ttnn.to_torch(input_tensor, dtype=torch_dtype)

    output_tensor = ttnn.cosh(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.cosh)
    golden_tensor = golden_function(in_data)

    if ttnn_dtype == ttnn.bfloat16:
        assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=1)
    else:
        assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_unary_sinh_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-9, 9)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data = ttnn.to_torch(input_tensor, dtype=torch_dtype)

    output_tensor = ttnn.sinh(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.sinh)
    golden_tensor = golden_function(in_data)

    if ttnn_dtype == ttnn.bfloat16:
        assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=5.0)

    else:
        assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0])
def test_unary_rpow_ttnn(input_shapes, exponent, device):
    in_data1 = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-30, 30)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor1, exponent)
    golden_function = ttnn.get_golden_function(ttnn.rpow)
    golden_tensor = golden_function(in_data1, exponent)

    assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=8)  # ULP<=1 for exponents less than 5
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.float32, ttnn.float32, 0.0094),
        (torch.bfloat16, ttnn.bfloat16, 0.04),
        (torch.bfloat16, ttnn.bfloat8_b, 0.05),
    ],
)
def test_unary_cbrt_ttnn(input_shapes, torch_dtype, ttnn_dtype, atol, device):
    in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data = ttnn.to_torch(input_tensor, dtype=torch_dtype)

    output_tensor = ttnn.cbrt(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.cbrt)
    golden_tensor = golden_function(in_data)

    if ttnn_dtype == ttnn.bfloat16:
        assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=2.0)
    else:
        assert_allclose(ttnn.to_torch(output_tensor), golden_tensor, rtol=1e-05, atol=atol)


def test_cbrt_arange(device):
    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # Mask NaN, special values where cbrt has ULP>1 (Covered in atol test below).
    # Also mask values in range -1 to 1.
    mask = torch.isnan(input_tensor) | torch.isinf(input_tensor) | ((input_tensor > -1.0) & (input_tensor < 1.0))
    input_tensor[mask] = 1.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.cbrt)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.cbrt(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 2, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.isinf, ttnn.isnan, ttnn.isposinf, ttnn.isneginf, ttnn.isfinite])
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_inf_nan_check(ttnn_op, torch_dtype, ttnn_dtype, device):
    in_data = torch.tensor(
        [float("-inf"), float("inf"), float("nan"), 5.0, -5.0, 0.0, -0.0, 1e38, 1e-45, 3.4e38, -3.4e38],
        dtype=torch_dtype,
    )
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_op(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(in_data)

    assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 1.0, 5.75, 10.0])
def test_unary_leaky_relu_ttnn(input_shapes, negative_slope, torch_dtype, ttnn_dtype, device):
    in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat8_b:
        in_data = ttnn.to_torch(input_tensor, dtype=torch_dtype)

    output_tensor = ttnn.leaky_relu(input_tensor, negative_slope=negative_slope)
    golden_function = ttnn.get_golden_function(ttnn.leaky_relu)
    golden_tensor = golden_function(in_data, negative_slope=negative_slope)

    if ttnn_dtype == ttnn.bfloat8_b:
        assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)
    else:
        assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([100])),
        (torch.Size([10, 10])),
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 1, 102400, 32])),
        (torch.Size([1, 1, 102400, 64])),
        (torch.Size([1, 1, 400, 512])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_unary_hardmish(input_shapes, torch_dtype, ttnn_dtype, device):
    in_data1 = create_full_range_tensor(input_shapes, torch_dtype)

    # limit the range to avoid overflow in hardmish
    in_data1 = in_data1[(in_data1 + 2.8).abs() < 3.3e38 / 5]

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.hardmish(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.hardmish)

    golden_tensor = golden_function(in_data1, device=device)
    tt_res = ttnn.to_torch(output_tensor)

    assert_with_pcc(tt_res, golden_tensor, pcc=0.9999)


def test_hardmish_bfloat16_ulp(device):
    # Generate all possible bit pattersn for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # Mask NaN, special values where hardmish has ULP>1 (Covered in atol test below).
    mask = (
        torch.isnan(input_tensor)
        | ((input_tensor >= -2.0847e-23) & (input_tensor <= 2.0939e-23))
        | (input_tensor == -0.0)
        | (input_tensor >= 6.8122e37)
        | (input_tensor == -torch.inf)
    )
    input_tensor[mask] = 0.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    golden_function = ttnn.get_golden_function(ttnn.hardmish)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.hardmish(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


def test_hardmish_bfloat16_allclose(device):
    # Generate all possible bit pattersn for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    golden_function = ttnn.get_golden_function(ttnn.hardmish)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.hardmish(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_allclose(golden, result, rtol=1e-05, atol=1e-35)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 1, 3, 64, 12])),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.rsqrt, ttnn.sqrt])
@pytest.mark.parametrize("fast_approx_mode", [True, False])
def test_unary_root_ops_ttnn(input_shapes, torch_dtype, ttnn_dtype, ttnn_op, fast_approx_mode, device):
    torch.manual_seed(0)
    if fast_approx_mode:
        in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(1, 100)
    else:
        in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_op(input_tensor, fast_and_approximate_mode=fast_approx_mode)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(in_data)

    if fast_approx_mode:
        assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=2)
    else:
        output_tensor = ttnn.to_torch(output_tensor, dtype=torch_dtype)
        if torch_dtype == torch.bfloat16:
            # Check if both tensors have non-finite values at the same indices
            golden_nonfinite = ~torch.isfinite(golden_tensor)
            output_nonfinite = ~torch.isfinite(output_tensor)

            # Verify non-finite values occur at the same indices
            assert torch.equal(
                golden_nonfinite, output_nonfinite
            ), f"Non-finite values don't match at the same indices."

            # For finite values, check all of them
            finite_mask = torch.isfinite(golden_tensor) & torch.isfinite(output_tensor)
            if finite_mask.any():
                assert_with_ulp(
                    golden_tensor[finite_mask], output_tensor[finite_mask], ulp_threshold=2, allow_nonfinite=False
                )
        else:
            assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=2, allow_nonfinite=True)


@pytest.mark.parametrize(
    "param",
    {-1.5, 1.7, 0.0},
)
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
def test_unary_rdiv_inf_nan_check(param, round_mode, device):
    dtype = torch.bfloat16
    if dtype == torch.bfloat16 and param == 0.0:
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    in_data = torch.zeros(torch.Size([1, 1, 32, 32]), dtype=dtype)
    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.rdiv(input_tensor, param, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden_tensor = golden_function(in_data, param, round_mode=round_mode)

    assert torch.equal(golden_tensor, ttnn.to_torch(output_tensor))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 1, 3, 64, 12])),
    ),
)
@pytest.mark.parametrize(
    "param",
    {-98.5, -43.7, -8.5, 0.45, 7.7, 58.4, 89.9},
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("round_mode", [None, "trunc", "floor"])
def test_unary_rdiv_ttnn(input_shapes, torch_dtype, ttnn_dtype, param, round_mode, device):
    torch.manual_seed(0)
    in_data = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.rdiv(input_tensor, param, round_mode=round_mode)
    golden_function = ttnn.get_golden_function(ttnn.rdiv)
    golden_tensor = golden_function(in_data, param, round_mode=round_mode)
    output_tensor = ttnn.to_torch(output_tensor)

    if (round_mode != None) and (torch_dtype == torch.bfloat16):
        assert_with_pcc(golden_tensor, output_tensor, pcc=0.999)
    else:
        assert_with_ulp(golden_tensor, output_tensor, ulp_threshold=3, allow_nonfinite=True)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "input_vals, torch_input_dtype, torch_output_dtype, ttnn_input_dtype, ttnn_output_dtype",
    [
        # uint16 -> bfloat16 conversions
        ([16457, 16429, 32641], torch.uint16, torch.bfloat16, ttnn.uint16, ttnn.bfloat16),
        ([0, 0, 0], torch.uint16, torch.bfloat16, ttnn.uint16, ttnn.bfloat16),
        ([65535, 65534, 65533], torch.uint16, torch.bfloat16, ttnn.uint16, ttnn.bfloat16),
        ([31744, 64512], torch.uint16, torch.bfloat16, ttnn.uint16, ttnn.bfloat16),
        # bfloat16 -> uint16 conversions
        ([3.140625, 2.703125, 0.0], torch.bfloat16, torch.uint16, ttnn.bfloat16, ttnn.uint16),
        ([1.0, -1.0, 0.0], torch.bfloat16, torch.uint16, ttnn.bfloat16, ttnn.uint16),
        # int32 -> uint32 conversions
        ([-1, 0, 2147483647], torch.int32, torch.uint32, ttnn.int32, ttnn.uint32),
        # uint32 -> float32 conversions
        ([16457, 16429, 32641], torch.uint32, torch.float32, ttnn.uint32, ttnn.float32),
        ([1078523331, 1078523332], torch.uint32, torch.float32, ttnn.uint32, ttnn.float32),
    ],
)
def test_unary_bitcast_ttnn(
    input_shapes, input_vals, torch_input_dtype, torch_output_dtype, ttnn_input_dtype, ttnn_output_dtype, device
):
    """Test bitcast operation - reinterprets bit pattern without conversion"""
    # Create PyTorch reference
    x_torch = torch.tensor(input_vals, dtype=torch_input_dtype)
    y_torch = x_torch.view(torch_output_dtype)

    # Pad input to match tile size
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    padded_vals = input_vals + [0] * (num_elements - len(input_vals))

    # Create PyTorch tensor and convert to TTNN tensor
    padded_torch_tensor = torch.tensor(padded_vals, dtype=torch_input_dtype).reshape(input_shapes)
    input_tensor = ttnn.from_torch(
        padded_torch_tensor,
        dtype=ttnn_input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Perform bitcast
    output_tensor = ttnn.bitcast(input_tensor, ttnn_output_dtype)
    # Convert to ROW_MAJOR layout before converting to torch (bitcast preserves input layout)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    # Convert to torch tensor
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_output_dtype)

    # Extract the relevant values from the output tensor (first len(input_vals) elements)
    output_vals = output_tensor.flatten()[: len(input_vals)].tolist()
    expected_vals = y_torch.tolist()

    # Compare values
    # Note: NaN values may convert to inf due to hardware packer limitation
    # For non-NaN, non-inf values, we expect exact match
    for i, (expected, actual) in enumerate(zip(expected_vals, output_vals)):
        if torch.isnan(torch.tensor(expected)):
            # NaN values may convert to inf in bfloat16 due to packer hardware limitation
            assert torch.isinf(torch.tensor(actual)) or torch.isnan(
                torch.tensor(actual)
            ), f"Value {i}: Expected NaN, got {actual}"
        elif torch.isinf(torch.tensor(expected)):
            # Inf values should match
            assert torch.isinf(torch.tensor(actual)), f"Value {i}: Expected Inf, got {actual}"
        else:
            # Normal values should match exactly
            # Note: There may be precision loss due to hardware limitations
            if torch_output_dtype == torch.float32:
                # Allow tolerance for precision issues
                assert (
                    abs(expected - actual) < 0.002 or expected == actual
                ), f"Value {i}: Expected {expected}, got {actual}, diff: {abs(expected - actual)}"
            else:
                assert expected == actual, f"Value {i}: Expected {expected}, got {actual}"


@pytest.mark.parametrize(
    "input_shape",
    (
        torch.Size([3, 128, 32]),
        torch.Size([1, 1, 3, 64, 12]),
    ),
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.bfloat16, ttnn.bfloat16, 0.016),
        (torch.float32, ttnn.float32, 0.015),
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-100, 100),
        (-1, 2),
        (0, 2),
    ],
)
@pytest.mark.parametrize("scalar", [0.25, 0.38, 0.5, 0.85])
def test_unary_logit(input_shape, scalar, torch_dtype, ttnn_dtype, high, low, device, atol):
    torch.manual_seed(0)
    in_data = torch.empty(input_shape, dtype=torch_dtype).uniform_(low, high)
    input_tensor_a = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.logit(input_tensor_a, eps=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.logit)
    golden_tensor = golden_function(in_data, eps=scalar)

    assert_allclose(output_tensor, golden_tensor, rtol=1e-05, atol=atol)


@pytest.mark.parametrize("input_shape", (torch.Size([3, 128, 32]),))
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, atol",
    [
        (torch.bfloat16, ttnn.bfloat16, 0.04),
        (torch.float32, ttnn.float32, 0.016),
    ],
)
@pytest.mark.parametrize("eps", [0.0, 1.0, None])
def test_unary_logit_edge_cases(input_shape, torch_dtype, ttnn_dtype, device, eps, atol):
    torch.manual_seed(0)
    in_data = torch.empty(input_shape, dtype=torch_dtype).uniform_(-1, 1.1)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.logit(input_tensor, eps=eps)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.logit)
    golden_tensor = golden_function(in_data, eps=eps)
    if eps is None:
        golden_nonfinite = ~torch.isfinite(golden_tensor)
        output_nonfinite = ~torch.isfinite(output_tensor)

        # Verify non-finite values occur at the same indices
        assert torch.equal(golden_nonfinite, output_nonfinite), f"Non-finite values don't match at the same indices."

        # For finite values, check all of them
        finite_mask = torch.isfinite(golden_tensor) & torch.isfinite(output_tensor)
        if finite_mask.any():
            assert torch.allclose(
                output_tensor[finite_mask], golden_tensor[finite_mask], equal_nan=True, rtol=1e-05, atol=atol
            )
    else:
        assert torch.allclose(output_tensor, golden_tensor, equal_nan=True, rtol=1e-05, atol=atol)
