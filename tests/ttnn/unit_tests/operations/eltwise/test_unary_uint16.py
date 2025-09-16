# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


def _run_unary_uint16_test(torch_input_tensor, ttnn_fn, device, memory_config=None):
    """Helper function to run uint16 unary operation tests with consistent setup"""
    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )

    if memory_config:
        output_tensor = ttnn_fn(input_tensor, memory_config=memory_config)
    else:
        output_tensor = ttnn_fn(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor), (
        f"Mismatch in {ttnn_fn.__name__} operation. "
        f"Expected shape: {torch_output_tensor.shape}, Got shape: {output_tensor.shape}"
    )


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 2048, 2048]),
        torch.Size([1, 1, 4096, 4096]),
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
@pytest.mark.parametrize("ttnn_fn", [ttnn.logical_not])
def test_unary_uint16_operations(input_shapes, low_a, high_a, ttnn_fn, device):
    """Test uint16 unary operations with various shapes and value ranges"""
    if len(input_shapes) == 0:
        torch_input_tensor = torch.randint(low=0, high=2**16, size=(), dtype=torch.int32)
    else:
        num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
        torch_input_tensor = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)

        # Create more diverse test patterns
        torch_input_tensor[::5] = 0  # every 5th element is zero
        torch_input_tensor[1::7] = 1  # some elements are one
        if num_elements > 10:
            torch_input_tensor[-3:] = torch.tensor([65535, 32767, 1])  # add edge cases

        torch_input_tensor = torch_input_tensor[:num_elements].reshape(input_shapes)

    _run_unary_uint16_test(torch_input_tensor, ttnn_fn, device)


@pytest.mark.parametrize("ttnn_fn", [ttnn.logical_not])
@pytest.mark.parametrize(
    "edge_pattern",
    [
        "critical_values",
        "powers_of_two",
        "boundary_values",
        "sequential_pattern",
    ],
)
def test_unary_uint16_edge_cases(ttnn_fn, edge_pattern, device):
    """Test uint16 unary operations with specific edge case patterns"""

    if edge_pattern == "critical_values":
        # Test critical uint16 edge values
        edge_values = [
            0,  # Zero - should become 1 for logical_not
            1,  # One - should become 0 for logical_not
            65535,  # Max uint16 - should become 0 for logical_not
            32767,  # Mid-range - should become 0 for logical_not
            32768,  # Mid-range + 1 - should become 0 for logical_not
            2,  # Small non-zero - should become 0 for logical_not
            16384,  # Quarter max - should become 0 for logical_not
            49151,  # Three-quarter max - should become 0 for logical_not
        ]
    elif edge_pattern == "powers_of_two":
        # Test powers of 2 and nearby values
        edge_values = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65535]
    elif edge_pattern == "boundary_values":
        # Test boundary values around different ranges
        edge_values = [0, 1, 255, 256, 257, 32767, 32768, 32769, 65534, 65535]
    else:  # sequential_pattern
        # Test sequential values around zero and max
        edge_values = list(range(0, 10)) + list(range(65526, 65536))

    torch_input_tensor = torch.tensor(edge_values, dtype=torch.int32)
    _run_unary_uint16_test(torch_input_tensor, ttnn_fn, device)


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
@pytest.mark.parametrize("ttnn_fn", [ttnn.logical_not])
def test_unary_uint16_sharded(a_shape, sharded_config, ttnn_fn, device):
    """Test uint16 unary operations with sharded memory configurations"""
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor = torch.linspace(0, 65535, num_elements, dtype=torch.int32)

    # Create more diverse test patterns for sharded testing
    torch_input_tensor[::5] = 0  # enforce that every 5th element is zero
    torch_input_tensor[1::8] = 1  # some elements are one
    torch_input_tensor[2::11] = 65535  # some max values

    torch_input_tensor = torch_input_tensor[:num_elements].reshape(a_shape)

    _run_unary_uint16_test(torch_input_tensor, ttnn_fn, device, memory_config=sharded_config)


@pytest.mark.parametrize("ttnn_fn", [ttnn.logical_not])
@pytest.mark.parametrize(
    "test_pattern",
    [
        "all_zeros",
        "all_ones",
        "all_max",
        "alternating",
        "random_sparse",
    ],
)
def test_unary_uint16_stress_patterns(ttnn_fn, test_pattern, device):
    """Test uint16 unary operations with stress/edge patterns"""
    shape = torch.Size([1, 1, 64, 64])
    num_elements = int(torch.prod(torch.tensor(shape)).item())

    if test_pattern == "all_zeros":
        torch_input_tensor = torch.zeros(shape, dtype=torch.int32)
    elif test_pattern == "all_ones":
        torch_input_tensor = torch.ones(shape, dtype=torch.int32)
    elif test_pattern == "all_max":
        torch_input_tensor = torch.full(shape, 65535, dtype=torch.int32)
    elif test_pattern == "alternating":
        torch_input_tensor = torch.zeros(shape, dtype=torch.int32)
        torch_input_tensor.view(-1)[::2] = 65535  # Alternating 0 and max
    else:  # random_sparse
        torch_input_tensor = torch.zeros(shape, dtype=torch.int32)
        # Set 10% of elements to random non-zero values
        num_nonzero = num_elements // 10
        indices = torch.randperm(num_elements)[:num_nonzero]
        torch_input_tensor.view(-1)[indices] = torch.randint(1, 65536, (num_nonzero,), dtype=torch.int32)

    _run_unary_uint16_test(torch_input_tensor, ttnn_fn, device)
