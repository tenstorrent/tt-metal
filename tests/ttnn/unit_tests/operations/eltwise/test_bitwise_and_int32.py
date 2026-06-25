# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -100, 100),
    ],
)
def test_bitwise_and_int32(input_shapes, low_a, high_a, low_b, high_b, device):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.randint(low_a, high_a, input_shapes, dtype=torch.int32)
    torch_input_tensor_b = torch.randint(low_b, high_b, input_shapes, dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.bitwise_and)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.bitwise_and(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


def test_bitwise_and_int32_edge_cases(device):
    torch_input_tensor_a = torch.tensor(
        [0, -1, 1, 2147483647, -2147483648, 1073741823, -1073741824, 255, -256, 12345, -98765, 0],
        dtype=torch.int32,
    )
    torch_input_tensor_b = torch.tensor(
        [0, -1, -1, 2147483647, -2147483648, -1, 0xFF, 0x0F, -1, 0xFFFF, 0xF0F0, 2147483647],
        dtype=torch.int32,
    )

    golden_function = ttnn.get_golden_function(ttnn.bitwise_and)
    torch_output_tensor = golden_function(torch_input_tensor_a, 5)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.bitwise_and(input_tensor_a, 5)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([4, 3, 512, 512])),
    ],
)
@pytest.mark.parametrize("scalar", [5, 0xFF, 0])
def test_bitwise_and_int32_scalar(input_shapes, scalar, device):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.randint(-2147483647, 2147483647, input_shapes, dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.bitwise_and)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.bitwise_and(input_tensor_a, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype, hi, mask",
    [
        (ttnn.uint16, 0xFFFF, 0xFFFF),
        (ttnn.uint32, 0x7FFFFFFF, 0xFFFFFFFF),
    ],
)
@pytest.mark.parametrize("scalar", [0x0F0F, 0xFF, 0])
def test_bitwise_and_scalar_unsigned(input_shapes, ttnn_dtype, hi, mask, scalar, device):
    # Exercises the tensor-scalar SFPU fast path for unsigned dtypes (uint16 / uint32).
    torch.manual_seed(0)
    torch_input_tensor_a = torch.randint(0, hi, input_shapes, dtype=torch.int32)

    torch_output_tensor = (torch_input_tensor_a & scalar) & mask

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.to_torch(ttnn.bitwise_and(input_tensor_a, scalar)).to(torch.int64) & mask

    assert torch.equal(output_tensor, torch_output_tensor.to(torch.int64))
