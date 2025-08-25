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
        torch.Size([1, 2, 64, 128]),
        torch.Size([]),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a",
    [
        (0, 100),
        (100, 1000),
        (1000, 10000),
        (30000, 50000),
        (0, 65535),  # full uint16 range
    ],
)
def test_unary_logical_not_uint16(input_shapes, low_a, high_a, device):
    if len(input_shapes) == 0:
        torch_input_tensor = torch.randint(low=0, high=2**16, size=(), dtype=torch.int32)
    else:
        num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
        torch_input_tensor = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)
        torch_input_tensor[::5] = 0  # every 5th element is zero
        corner_case = torch.tensor([65535, 32767, 1])  # max uint16, mid-range, and 1
        torch_input_tensor[-3:] = corner_case
        torch_input_tensor = torch_input_tensor[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


def test_unary_logical_not_uint16_edge_cases(device):
    """Test edge cases for uint16 logical not"""
    torch_input_tensor = torch.tensor([0, 1, 65535, 32767, 16384], dtype=torch.int32)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    output_tensor = ttnn.logical_not(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [128, 160],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 6)), ttnn.CoreRange((3, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [2240, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 6)), ttnn.CoreRange((3, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [32, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 0))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "a_shape",
    [
        torch.Size([5, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
def test_unary_logical_not_uint16_sharded(a_shape, sharded_config, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor = torch.linspace(0, 65535, num_elements, dtype=torch.int32)
    torch_input_tensor[::5] = 0  # every 5th element is zero
    torch_input_tensor = torch_input_tensor[:num_elements].reshape(a_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    output_tensor = ttnn.logical_not(input_tensor, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 2, 64, 128]),
    ],
)
def test_unary_logical_not_uint16_comprehensive(input_shapes, device):
    """Comprehensive test with various uint16 patterns"""
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)

    # Create test data with specific patterns
    torch_input_tensor = torch.zeros(num_elements, dtype=torch.int32)

    # Pattern 1: zeros and ones
    torch_input_tensor[: num_elements // 4] = 0
    torch_input_tensor[num_elements // 4 : num_elements // 2] = 1

    # Pattern 2: small values
    torch_input_tensor[num_elements // 2 : 3 * num_elements // 4] = torch.randint(2, 100, (num_elements // 4,))

    # Pattern 3: large values and edge cases
    torch_input_tensor[3 * num_elements // 4 :] = torch.randint(30000, 65536, (num_elements - 3 * num_elements // 4,))

    # Ensure some edge cases
    if num_elements >= 4:
        torch_input_tensor[-4] = 65535  # max uint16
        torch_input_tensor[-3] = 32767  # half max
        torch_input_tensor[-2] = 1  # minimal non-zero
        torch_input_tensor[-1] = 0  # zero

    torch_input_tensor = torch_input_tensor.reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.logical_not(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)
