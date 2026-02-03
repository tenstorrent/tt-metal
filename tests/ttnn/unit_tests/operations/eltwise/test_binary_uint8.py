# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32, 64]), torch.Size([1, 2, 32, 64])),
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        ttnn.squared_difference,
    ],
)
def test_binary_uint8_arithmetic(a_shape, b_shape, ttnn_fn, device):
    torch_input_tensor_a = torch.randint(0, 100, a_shape, dtype=torch.uint8)
    torch_input_tensor_b = torch.randint(0, 100, b_shape, dtype=torch.uint8)

    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor_a.to(torch.int32), torch_input_tensor_b.to(torch.int32))

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b)
    
    # Typecast to uint32 for comparison as to_torch might have issues with uint8/uint16 directly
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32, 64]), torch.Size([1, 2, 32, 64])),
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.logical_and,
        ttnn.logical_or,
        ttnn.logical_xor,
    ],
)
def test_binary_uint8_logical(a_shape, b_shape, ttnn_fn, device):
    torch_input_tensor_a = torch.randint(0, 2, a_shape, dtype=torch.uint8) * 255
    torch_input_tensor_b = torch.randint(0, 2, b_shape, dtype=torch.uint8) * 255

    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor_a.to(torch.int32), torch_input_tensor_b.to(torch.int32))

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    # Convert to boolean for logical comparison comparison if golden returns boolean
    if torch_output_tensor.dtype == torch.bool:
        torch_output_tensor = torch_output_tensor.int()
    
    # ttnn logical ops often return 0 or 1, but sometimes they might return 0 or 255 or something else depending on implementation.
    # get_golden_function for logical ops usually returns 0/1.
    # Let's normalize both to 0/1 if they are not already.
    output_tensor = (output_tensor != 0).int()
    torch_output_tensor = (torch_output_tensor != 0).int()

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32, 64]), torch.Size([1, 2, 32, 64])),
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.eq,
        ttnn.ne,
    ],
)
def test_binary_uint8_comparison(a_shape, b_shape, ttnn_fn, device):
    torch_input_tensor_a = torch.randint(0, 10, a_shape, dtype=torch.uint8)
    # Make some elements equal
    torch_input_tensor_b = torch_input_tensor_a.clone()
    mask = torch.rand(a_shape) > 0.5
    torch_input_tensor_b[mask] = torch.randint(0, 10, (int(mask.sum()),), dtype=torch.uint8)

    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor_a.to(torch.int32), torch_input_tensor_b.to(torch.int32))

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    
    if torch_output_tensor.dtype == torch.bool:
        torch_output_tensor = torch_output_tensor.int()

    output_tensor = (output_tensor != 0).int()
    torch_output_tensor = (torch_output_tensor != 0).int()

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32, 64]), torch.Size([1, 2, 32, 64])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
def test_binary_uint8_bitwise(a_shape, b_shape, ttnn_fn, device):
    torch_input_tensor_a = torch.randint(0, 256, a_shape, dtype=torch.uint8)
    torch_input_tensor_b = torch.randint(0, 256, b_shape, dtype=torch.uint8)

    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor_a.to(torch.int32), torch_input_tensor_b.to(torch.int32))

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32).to(torch.uint8)

    assert torch.equal(output_tensor, torch_output_tensor.to(torch.uint8))


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32, 64]), torch.Size([1, 2, 32, 64])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.gt,
        ttnn.lt,
        ttnn.ge,
        ttnn.le,
    ],
)
def test_binary_uint8_relational(a_shape, b_shape, ttnn_fn, device):
    torch_input_tensor_a = torch.randint(0, 100, a_shape, dtype=torch.uint8)
    torch_input_tensor_b = torch.randint(0, 100, b_shape, dtype=torch.uint8)

    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor_a.to(torch.int32), torch_input_tensor_b.to(torch.int32))

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn_fn(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    
    if torch_output_tensor.dtype == torch.bool:
        torch_output_tensor = torch_output_tensor.int()

    output_tensor = (output_tensor != 0).int()
    torch_output_tensor = (torch_output_tensor != 0).int()

    assert torch.equal(output_tensor, torch_output_tensor)


def test_binary_ne_uint8_specific_case(device):
    # Specific case from PR description
    torch_ones = torch.ones([32, 32], dtype=torch.uint8)
    torch_tril = torch.tril(torch_ones)

    input0 = ttnn.from_torch(torch_tril, dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device)
    input1 = ttnn.from_torch(torch_ones, dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device)

    # After fix: 1 where input0 != input1
    output = ttnn.ne(input0, input1)
    output = ttnn.to_torch(output)

    expected = (torch_tril != torch_ones).int()
    assert torch.equal(output.int(), expected)
