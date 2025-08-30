# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 3, 320, 384]),
        torch.Size([1, 2, 64, 128]),
        torch.Size([]),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a",
    [
        (0, 100),
        (1000, 10000),
        (0, 65535),  # full uint16 range
    ],
)
def test_unary_logical_not_uint16(input_shapes, low_a, high_a, device):
    """Test uint16 logical not with various shapes and value ranges"""
    if len(input_shapes) == 0:
        torch_input_tensor = torch.randint(low=0, high=2**16, size=(), dtype=torch.int32)
    else:
        num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
        torch_input_tensor = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)
        torch_input_tensor[::5] = 0  # every 5th element is zero
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
    """Test uint16 logical not with specific edge cases"""
    # Test critical uint16 edge values
    edge_values = [
        0,  # Zero - should become 1
        1,  # One - should become 0
        65535,  # Max uint16 - should become 0
        32767,  # Mid-range - should become 0
        32768,  # Mid-range + 1 - should become 0
        2,  # Small non-zero - should become 0
        16384,  # Quarter max - should become 0
        49151,  # Three-quarter max - should become 0
    ]

    torch_input_tensor = torch.tensor(edge_values, dtype=torch.int32)

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
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((2, 2), (2, 3)), ttnn.CoreRange((0, 0), (0, 1))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [320, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (4, 6))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
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

    torch_input_tensor[::5] = 0  # enforce that every 5th element is zero
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
