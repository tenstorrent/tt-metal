# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("fill_value", [2147483647, -2147483648, 15.5])
def test_fill_int32(device, fill_value):
    torch_input_tensor = torch.zeros((1, 2), dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    output_tensor = ttnn.to_torch(output)

    equal_passed, equal_message = assert_equal(torch_output_tensor, output_tensor)
    assert equal_passed


@pytest.mark.parametrize("fill_value", [2147483647, 25.5, 4294967293, 1000000000, 4294967295])
def test_fill_uint32(device, fill_value):
    torch_input_tensor = torch.ones((1, 2), dtype=torch.uint32)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    torch_output = ttnn.from_torch(torch_output_tensor, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.eq(output, torch_output)
    assert torch.all(ttnn.to_torch(result))


@pytest.mark.parametrize("fill_value", [2, 43.5, 65535])
def test_fill_uint16(device, fill_value):
    torch_input_tensor = torch.ones((1, 2), dtype=torch.uint16)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.fill(input_tensor, fill_value)
    torch_output = ttnn.from_torch(torch_output_tensor, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.eq(output, torch_output)
    assert torch.all(ttnn.to_torch(result))


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (2, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "fill_value",
    [
        0.1,
        0.3,
        1 / 3,
        -0.1,
        1.2,
        3.14159,
        1.00390625,  # Tie rounds down to even.
        1.01171875,  # Tie rounds up to even.
        -1.00390625,
        -1.01171875,
        1.99609375,  # Rounding carries into the exponent.
        515,  # Integer scalar variants must also round when the output is BF16.
        -515,
    ],
)
def test_fill_rounding(device, shape, dtype, layout, memory_config, fill_value):
    torch_input = torch.zeros(shape, dtype=dtype)
    input_tensor = ttnn.from_torch(torch_input, layout=layout, device=device, memory_config=memory_config)
    output = ttnn.fill(input_tensor, fill_value)
    torch.testing.assert_close(ttnn.to_torch(output), torch.full_like(torch_input, fill_value), rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fill_value", [0.1, 1.01171875])
def test_fill_rounding_preallocated_output(device, input_dtype, output_dtype, fill_value):
    shape = (1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch.zeros(shape, dtype=input_dtype), layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.from_torch(torch.zeros(shape, dtype=output_dtype), layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(2):
        output = ttnn.fill(input_tensor, fill_value, output_tensor=output_tensor)
        expected = torch.full(shape, fill_value, dtype=output_dtype)
        torch.testing.assert_close(ttnn.to_torch(output), expected, rtol=0, atol=0)
        torch.testing.assert_close(ttnn.to_torch(output_tensor), expected, rtol=0, atol=0)
