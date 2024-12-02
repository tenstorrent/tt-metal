# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import math

from functools import partial
from models.utility_functions import torch_random
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


@skip_for_grayskull("Float32 unsupported on Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_fp32(input_shapes, device):
    torch_input_tensor = torch.rand(input_shapes, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.00001, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.00001"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_bf16(input_shapes, device):
    torch_input_tensor = torch.rand(input_shapes, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.01, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.01"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-3, rtol=1e-2, equal_nan=False)
    assert status


@skip_for_grayskull("Float32 unsupported on Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_pos_range_fp32(input_shapes, device):
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=0, high=1e3, dtype=torch.float32),
        ttnn.float32,
    )(input_shapes)
    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.0001, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.0001"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-4, rtol=1e-4, equal_nan=False)
    assert status


@skip_for_grayskull("Float32 unsupported on Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_neg_range_fp32(input_shapes, device):
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-1e3, high=0, dtype=torch.float32),
        ttnn.float32,
    )(input_shapes)
    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.0001, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.0001"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-4, rtol=1e-4, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_pos_range_bf16(input_shapes, device):
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=0, high=4, dtype=torch.float32),
        ttnn.float32,
    )(input_shapes)
    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.01, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.01"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-2, rtol=1e-3, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 3, 32, 32])), (torch.Size([12, 32, 256, 256])), (torch.Size([8, 8, 200, 400]))),
)
def test_cos_neg_range_bf16(input_shapes, device):
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-4, high=0, dtype=torch.float32),
        ttnn.float32,
    )(input_shapes)
    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.01, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.01"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-2, rtol=1e-3, equal_nan=False)
    assert status


@skip_for_grayskull("Float32 unsupported for Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 3, 32, 32])), (torch.Size([12, 32, 256, 256])), (torch.Size([8, 8, 200, 400]))),
)
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_cos_standard_angles(input_shapes, input_dtype, device):
    standard_angles = torch.arange(-2 * math.pi, 2 * math.pi + 1e-6, math.pi / 2)
    standard_angles = standard_angles.to(torch.float32)
    num_angles = standard_angles.numel()
    random_indices = torch.randint(0, num_angles, size=input_shapes, dtype=torch.long)
    torch_input_tensor = standard_angles[random_indices]

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.01, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.01"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-2, rtol=1e-3, equal_nan=False)
    assert status


# bf16 returns inf instead of nan
@skip_for_grayskull("Float32 unsupported for Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 3, 32, 32])), (torch.Size([12, 32, 256, 256])), (torch.Size([8, 8, 200, 400]))),
)
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_nan_values(input_shapes, input_dtype, device):
    torch_input_tensor = torch.full(input_shapes, float("nan"), dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    # assert torch.all(torch.isnan(output_tensor)), "Output tensor is not all NaN"


# fp32 and bf16 returns -inf instead of nan
@skip_for_grayskull("Float32 unsupported for Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 3, 32, 32])), (torch.Size([12, 32, 256, 256])), (torch.Size([8, 8, 200, 400]))),
)
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
def test_inf_values(input_shapes, input_dtype, device):
    torch_input_tensor = torch.full(input_shapes, float("inf"), dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    # assert torch.all(torch.isnan(output_tensor)), "Output tensor is not all NaN"


@skip_for_grayskull("Float32 unsupported on Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_cos_zeros(input_shapes, input_dtype, device):
    torch_input_tensor = torch.zeros(input_shapes, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.00001, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.00001"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Float32 unsupported on Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_pi_2_fp32(input_shapes, device):
    torch_input_tensor = torch.full(input_shapes, math.pi / 2, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.00001, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.00001"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-7, rtol=1e-5, equal_nan=False)
    assert status


# Returns 0.0005 instead of 0.0
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 3, 32, 32])),
        (torch.Size([12, 32, 256, 256])),
        (torch.Size([8, 8, 200, 400])),
    ),
)
def test_cos_pi_2_bf16(input_shapes, device):
    torch_input_tensor = torch.full(input_shapes, math.pi / 2, dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.cos(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.to(torch.float32)

    violating_values = output_tensor[torch.abs(output_tensor) > 1]
    assert (
        violating_values.numel() == 0
    ), f"Output tensor values are out of amplitude limits [-1, 1], found values: {violating_values}"

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)

    max_abs_diff = torch.max(torch.abs(torch_output_tensor - output_tensor))
    assert max_abs_diff <= 0.0005, f"Maximum absolute difference {max_abs_diff.item()} is greater than 0.0005"

    status = torch.allclose(torch_output_tensor, output_tensor, atol=1e-3, rtol=1e-5, equal_nan=False)
    assert status
