# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.common.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


def create_full_range_tensor(input_shape, dtype, value_ranges):
    num_elements = torch.prod(torch.tensor(input_shape)).item()
    num_ranges = len(value_ranges)
    elements_per_range = num_elements // num_ranges
    leftover = num_elements % num_ranges

    segments = []
    for i, (low, high) in enumerate(value_ranges):
        n = elements_per_range + (1 if i < leftover else 0)
        segments.append(torch.linspace(low, high, steps=n, dtype=dtype))
    return torch.cat(segments).reshape(input_shape)


shape = torch.Size([1, 2, 32, 128])

DIV_RANGES_A = [
    (-100, 100),
    (-300, 300),
    (-750, 500),
    (-1000, 1000),
    (-1e4, 1e4),
    (-1e5, 1e5),
    (-1e7, 1e7),
    (-1e10, 1e10),
    (-1e15, 1e15),
    (1e8, 1e10),
    (1e12, 1e15),
    (-1e10, -1e8),
    (-1e15, -1e12),
    (-1e-5, 1e-5),
    (-1e-10, 1e-10),
]
DIV_RANGES_B = [
    (-50, 200),
    (-400, 600),
    (-2000, 3000),
    (-2e4, 2e4),
    (-3e5, 3e5),
    (-5e6, 5e6),
    (-2e8, 2e8),
    (-8e9, 8e9),
    (-1e14, 1e14),
    (2e8, 5e9),
    (8e11, 8e14),
    (-5e8, -2e7),
    (-8e14, -8e11),
    (-2e-4, 2e-4),
    (-2e-8, 2e-8),
]

MOD_RANGES_A = [
    (-100, 100),
    (-300, 300),
    (-1000, 1000),
    (-1e4, 1e4),
    (-5e4, 5e4),
    (0, 100),
    (-100, 0),
    (-1, 1),
    (-1e-2, 1e-2),
    (-1e-4, 1e-4),
    (50, 5000),
    (-5000, -50),
    (-2e4, 2e4),
    (1, 1e3),
    (-1e3, -1),
]
MOD_RANGES_B = [
    (0.5, 50),
    (1, 200),
    (2, 500),
    (5, 2000),
    (10, 5000),
    (0.5, 10),
    (-50, -0.5),
    (0.25, 4),
    (-4, -0.25),
    (0.01, 2),
    (20, 800),
    (-800, -20),
    (3, 3000),
    (0.5, 100),
    (-100, -0.5),
]


def test_remainder_fp32(device):
    torch_input_tensor_a = create_full_range_tensor(shape, torch.float32, MOD_RANGES_A)
    torch_input_tensor_b = create_full_range_tensor(shape, torch.float32, MOD_RANGES_B)

    golden_fn = ttnn.get_golden_function(ttnn.remainder)
    torch_output = golden_fn(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.remainder(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert torch.allclose(output, torch_output, atol=1e-2, rtol=1e-3, equal_nan=True)


def test_fmod_fp32(device):
    torch_input_tensor_a = create_full_range_tensor(shape, torch.float32, MOD_RANGES_A)
    torch_input_tensor_b = create_full_range_tensor(shape, torch.float32, MOD_RANGES_B)

    golden_fn = ttnn.get_golden_function(ttnn.fmod)
    torch_output = golden_fn(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.fmod(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert torch.allclose(output, torch_output, atol=1e-2, rtol=1e-3, equal_nan=True)


def test_div_no_nan_fp32(device):
    torch.manual_seed(0)
    torch_input_tensor_a = create_full_range_tensor(shape, torch.float32, DIV_RANGES_A)
    torch_input_tensor_b = create_full_range_tensor(shape, torch.float32, DIV_RANGES_B)
    # No-nan behavior
    b_flat = torch_input_tensor_b.flatten()
    b_flat[::101] = 0.0
    torch_input_tensor_b = b_flat.reshape(shape)

    golden_fn = ttnn.get_golden_function(ttnn.div_no_nan)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_output = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    output = ttnn.div_no_nan(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)
    assert_with_ulp(output, torch_output, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.remainder,
    ],
)
def test_remainder_forge(device, ttnn_function):
    torch.manual_seed(213919)
    input1 = torch.randn(2, 32, 32)
    input2 = torch.randn(2, 32, 32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output = golden_fn(input1, input2, device=device)

    input1 = ttnn.from_torch(input1, dtype=ttnn.float32)
    input2 = ttnn.from_torch(input2, dtype=ttnn.float32)

    input1 = ttnn.to_device(input1, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input2 = ttnn.to_device(input2, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    input1 = ttnn.to_layout(input1, ttnn.TILE_LAYOUT)
    input2 = ttnn.to_layout(input2, ttnn.TILE_LAYOUT)

    output = ttnn.remainder(input1, input2)

    output = ttnn.to_torch(output)

    status = ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.999
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [1, 1, 32, 32], [1, 2, 32, 64, 64]],
)
def test_binary_fmod_bf16(
    device,
    input_shapes,
):
    torch_input_tensor_a = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor_b = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-80, 120)
    torch_output_tensor = torch.fmod(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.fmod(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, 1)


# This test was added for #17361
# If input is a multiple of the scalar, the result should be 0, but both Torch and TT output either 0 or the scalar value itself depending on the operands.
# This inconsistency is persistent due to some fp precision loss in both Torch and TT.
# Eg: torch.remainder of (3, 1.5) = 0.0 and of (3, 0.003) = 0.003
# Eg: ttnn.remainder of (4, 0.004) = 0.004 and of (3, 0.003) = 0.0
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([6, 5, 320, 320])),
        (torch.Size([2, 1, 384, 320])),
        (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
        (torch.Size([1024])),
        (torch.Size([])),
    ),
)
@pytest.mark.parametrize("scalar", [-0.002, -0.001, -0.0006, -0.0003, 0.0, 0.0005, 0.0007, 0.001, 0.002])
def test_remainder_scalar(input_shapes, scalar, device):
    torch.manual_seed(0)
    if len(input_shapes) == 0:
        torch_input_tensor = torch.tensor(5.0, dtype=torch.bfloat16)
    else:
        torch_input_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
        )(input_shapes)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    output_tensor = ttnn.remainder(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    # Handle special case where TT returns -inf but PyTorch returns nan for fmod with zero divisor
    if scalar == 0.0:
        output_tensor = torch.where(
            torch.isinf(output_tensor), torch.tensor(float("nan"), dtype=output_tensor.dtype), output_tensor
        )
        assert torch.allclose(output_tensor, torch_output_tensor, equal_nan=True)
    else:
        assert torch.allclose(output_tensor, torch_output_tensor, atol=0.001, rtol=0)


# This test was added for #17362
# If input is a multiple of the scalar, the result should be 0, but both Torch and TT output either 0 or the scalar value itself depending on the operands.
# This inconsistency is persistent due to some fp precision loss in both Torch and TT.
# Eg: torch.remainder of (3, 1.5) = 0.0 and of (3, 0.003) = 0.003
# Eg: ttnn.remainder of (4, 0.004) = 0.004 and of (3, 0.003) = 0.0
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([2, 5, 32, 320])),
        (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
        (torch.Size([1024])),
        (torch.Size([])),
    ),
)
@pytest.mark.parametrize("scalar", [-0.0029, -0.002, -0.0005, 0.0, 0.0007, 0.001, 0.0025])
def test_fmod_scalar(input_shapes, scalar, device):
    torch.manual_seed(0)
    if len(input_shapes) == 0:
        torch_input_tensor = torch.tensor(5.0, dtype=torch.bfloat16)
    else:
        torch_input_tensor = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.bfloat16), ttnn.bfloat16
        )(input_shapes)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    torch_output_tensor = golden_function(torch_input_tensor, scalar, device=device)

    output_tensor = ttnn.fmod(input_tensor_a, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    # Handle special case where TT returns -inf but PyTorch returns nan for fmod with zero divisor
    if scalar == 0.0:
        output_tensor = torch.where(
            torch.isinf(output_tensor), torch.tensor(float("nan"), dtype=output_tensor.dtype), output_tensor
        )
        assert torch.allclose(output_tensor, torch_output_tensor, equal_nan=True)
    else:
        assert torch.allclose(output_tensor, torch_output_tensor, atol=0.001, rtol=0)


@pytest.mark.parametrize("val_a, val_b", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("approx", [True, False])
def test_div_by_zero(device, val_a, val_b, dtype, approx):
    torch_dtype = getattr(torch, dtype)
    tt_dtype = getattr(ttnn, dtype)

    if val_a == 0.0 and val_b == 0.0 and dtype == "bfloat16":
        pytest.skip("Skipping test for 0/0 on bfloat16")

    x_torch = torch.tensor([[val_a]], dtype=torch_dtype)
    y_torch = torch.tensor([[val_b]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.divide)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    z_tt_div = ttnn.divide(x_tt, y_tt, fast_and_approximate_mode=approx)
    tt_out = ttnn.to_torch(z_tt_div)

    # Note: torch.equal return false for if both tensors are nan
    # This is why we use assert_with_ulp to test for equality

    if approx and dtype == "bfloat16":
        pytest.skip("Skipping test for fast approximate mode")

    assert_with_ulp(z_torch, tt_out, 0, allow_nonfinite=True)


@pytest.mark.parametrize("val_a, val_b", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("approx", [True, False])
def test_divide_inplace_by_zero(device, val_a, val_b, dtype, approx):
    torch_dtype = getattr(torch, dtype)
    tt_dtype = getattr(ttnn, dtype)

    if val_a == 0.0 and val_b == 0.0 and dtype == "bfloat16":
        pytest.skip("Skipping test for 0/0 on bfloat16")

    x_torch = torch.tensor([[val_a]], dtype=torch_dtype)
    y_torch = torch.tensor([[val_b]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.divide)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt_inplace = ttnn.from_torch(x_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt_inplace = ttnn.from_torch(y_torch, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.divide_(x_tt_inplace, y_tt_inplace, fast_and_approximate_mode=approx)
    tt_out_inplace = ttnn.to_torch(x_tt_inplace)

    if approx and dtype == "bfloat16":
        pytest.skip("Skipping test for fast approximate mode")

    assert_with_ulp(z_torch, tt_out_inplace, 0, allow_nonfinite=True)


@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_remainder_nan(testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    if testing_dtype == "bfloat16":
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.remainder(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))


@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_fmod_nan(testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    if testing_dtype == "bfloat16":
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.fmod(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))


def test_optional_output_tensor_remainder(device):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.tensor([[5.0, 7.0, -5.0, -7.0, 3.5, 10.0, 1.5, -1.5, 9.0, 15.0]], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.tensor([[2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.5, 0.5, -2.0, -4.0]], dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    golden_function = ttnn.get_golden_function(ttnn.remainder)
    torch_golden = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    torch_opt_output_tensor = torch.zeros_like(torch_golden)
    optional_output_tensor = ttnn.from_torch(torch_opt_output_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn.remainder(
        input_tensor_a,
        input_tensor_b,
        output_tensor=optional_output_tensor,
    )
    optional_output_tensor = ttnn.to_torch(optional_output_tensor)

    assert_with_ulp(optional_output_tensor, torch_golden, ulp_threshold=0)
