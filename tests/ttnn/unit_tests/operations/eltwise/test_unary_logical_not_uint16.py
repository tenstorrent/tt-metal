# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 3, 320, 384]),
        torch.Size([2, 64, 128]),
        torch.Size([100]),
        torch.Size([64, 64]),
        torch.Size([3, 128, 32]),
        torch.Size([1, 1, 32, 320, 12]),
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [
        (0, 100),  # Small range
        (100, 1000),  # Medium range
        (10000, 30000),  # Large range
        (0, 65535),  # Full uint16 range
    ],
)
def test_unary_logical_not_uint16_basic(input_shapes, low, high, device):
    """Test basic uint16 logical not functionality"""
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data = torch.randint(low, high, (num_elements,), dtype=torch.int32)

    # Add some zeros to test logical not properly
    zero_indices = torch.randperm(num_elements)[: num_elements // 5]  # 20% zeros
    in_data[zero_indices] = 0

    in_data = in_data.reshape(input_shapes)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)

    # Convert back to torch for comparison
    # Note: Since ttnn.to_torch converts uint16 to int16 and then to int32,
    # we need to handle the conversion carefully
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    golden_tensor = golden_function(in_data, device=device)

    assert torch.equal(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([64, 64]),
        torch.Size([1, 3, 320, 384]),
    ],
)
def test_unary_logical_not_uint16_edge_cases(input_shapes, device):
    """Test uint16 logical not with edge cases: 0, 1, 65535"""
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)

    # Create tensor with edge cases
    edge_values = [0, 1, 65535]  # Zero, one, max uint16
    in_data = torch.zeros(num_elements, dtype=torch.int32)

    # Fill with edge values cyclically
    for i in range(num_elements):
        in_data[i] = edge_values[i % len(edge_values)]

    # Add some random values
    random_indices = torch.randperm(num_elements)[: num_elements // 3]
    in_data[random_indices] = torch.randint(2, 65534, (len(random_indices),), dtype=torch.int32)

    in_data = in_data.reshape(input_shapes)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)

    # Handle uint16 conversion issue
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    golden_tensor = golden_function(in_data, device=device)

    assert torch.equal(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([2, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
def test_unary_logical_not_uint16_memory_configs(input_shapes, memory_config, device):
    """Test uint16 logical not with different memory configurations"""
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data = torch.randint(0, 65535, (num_elements,), dtype=torch.int32)

    # Add zeros for better logical not testing
    zero_prob = 0.3
    zero_mask = torch.rand(num_elements) < zero_prob
    in_data[zero_mask] = 0

    in_data = in_data.reshape(input_shapes)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )

    output_tensor = ttnn.logical_not(input_tensor, memory_config=memory_config)

    # Handle uint16 conversion issue
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    golden_tensor = golden_function(in_data, device=device)

    assert torch.equal(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([64, 64]),
    ],
)
def test_unary_logical_not_uint16_zeros_only(input_shapes, device):
    """Test uint16 logical not with tensor containing only zeros"""
    in_data = torch.zeros(input_shapes, dtype=torch.int32)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)

    # Handle uint16 conversion issue
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    # All zeros should become all ones when logical not is applied
    expected_tensor = torch.ones_like(in_data, dtype=torch.int32)

    assert torch.equal(output_tensor, expected_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([64, 64]),
    ],
)
def test_unary_logical_not_uint16_non_zeros_only(input_shapes, device):
    """Test uint16 logical not with tensor containing only non-zero values"""
    in_data = torch.randint(1, 65535, input_shapes, dtype=torch.int32)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)

    # Handle uint16 conversion issue
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    # All non-zeros should become all zeros when logical not is applied
    expected_tensor = torch.zeros_like(in_data, dtype=torch.int32)

    assert torch.equal(output_tensor, expected_tensor)


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 1),  # Zero should become one
        (1, 0),  # One should become zero
        (100, 0),  # Any non-zero should become zero
        (65535, 0),  # Max uint16 should become zero
    ],
)
def test_unary_logical_not_uint16_single_values(value, expected, device):
    """Test uint16 logical not with specific single values"""
    input_shape = torch.Size([1, 1, 32, 32])
    in_data = torch.full(input_shape, value, dtype=torch.int32)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)

    # Handle uint16 conversion issue
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    expected_tensor = torch.full(input_shape, expected, dtype=torch.int32)

    assert torch.equal(output_tensor, expected_tensor)
