# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 20, 31])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([2, 4, 320, 1024])),
    ),
)
@pytest.mark.parametrize("fill_value", [1, 0, 5.5, -2.235])
def test_fill(device, input_shapes, fill_value):
    torch_input_tensor = torch.randn((input_shapes), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    output_tensor = ttnn.to_torch(output)
    equal_passed, equal_message = assert_equal(torch_output_tensor, output_tensor)
    assert equal_passed


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 20, 31])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize("fill_value", [5.88, -9.76])
def test_fill_fp32(device, input_shapes, fill_value):
    torch_input_tensor = torch.randn((input_shapes), dtype=torch.float32)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    output_tensor = ttnn.to_torch(output)
    equal_passed, equal_message = assert_equal(torch_output_tensor, output_tensor)
    assert equal_passed


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1])),
        # (torch.Size([1, 16, 1, 1])),
    ),
)
@pytest.mark.parametrize("fill_value", [0.07980])
def test_typecast(device, input_shapes, fill_value):
    ttnn.set_printoptions(profile="full")
    torch_input_tensor = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_zero = torch.zeros(([1, 16, 1, 1]), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_zero, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    output = ttnn.repeat(output, [1, 16, 1, 1])
    # output = ttnn.add(input_tensor2, output, use_legacy=True) # to cast to [1, 16, 1, 1]
    print("output before typcast", output)
    output1 = ttnn.typecast(output, ttnn.bfloat8_b)
    print("output after", output1)
    # output_tensor = ttnn.to_torch(output)
    # equal_passed, equal_message = assert_equal(torch_output_tensor, output_tensor)
    # assert equal_passed
