# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "ttnn_op, value_ranges",
    [
        (
            ttnn.add,
            [
                (0, 100, 0, 300),
                (1000, 10000, 500, 1000),
                (30000, 40000, 10000, 15000),
                (50000, 55000, 1000, 2000),
                (0, 70000, 80000, 200000),
            ],
        ),
        (
            ttnn.sub,
            [
                (100, 500, 0, 100),
                (1000, 10000, 500, 1000),
                (30000, 40000, 10000, 15000),
                (50000, 55000, 1000, 2000),
                (80000, 200000, 0, 70000),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize("range_idx", [0, 1, 2, 3, 4])
def test_binary_uint32_bcast(ttnn_op, value_ranges, a_shape, b_shape, range_idx, device):
    """Test uint32 addition and subtraction with broadcast"""
    low_a, high_a, low_b, high_b = value_ranges[range_idx]

    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
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
    "ttnn_op, low_a, high_a, low_b, high_b",
    [
        (ttnn.add, 0, 100, 100, 200),  # Addition: ranges don't overlap to avoid overflow
        (ttnn.sub, 50000, 100000, 0, 40000),  # Subtraction: ensure a > b for valid results
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    ((torch.Size([5, 7, 64, 128]), torch.Size([5, 7, 64, 128])),),
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
def test_binary_uint32_sharded(ttnn_op, low_a, high_a, low_b, high_b, a_shape, b_shape, sharded_config, device):
    """Test uint32 addition and subtraction with sharded memory configurations"""
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (0, 100, 200, 300),  # Test underflow cases (a < b)
        (500, 1000, 1000, 10000),
        (10000, 15000, 30000, 40000),
        (1000, 2000, 50000, 55000),
        (0, 70000, 80000, 200000),
    ],
)
def test_binary_sub_uint32_underflow(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
    """Test uint32 subtraction with underflow cases (wrap-around behavior)"""
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize("use_legacy", [True, False])
def test_binary_sub_uint32_edge_cases(use_legacy, device):
    """Test uint32 subtraction with edge cases including underflow"""
    torch_input_tensor_a = torch.tensor(
        [4294967295, 2147483647, 1000000000, 500, 0, 100, 1, 0]  # uint32 max, int32 max, large, medium, min
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_input_tensor_b = torch.tensor(
        [1, 1000000000, 2147483647, 1000, 1, 50, 2, 1]  # Various values to test different scenarios
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Expected output values for uint32 subtraction with these specific inputs
    expected_output_values = torch.tensor(
        [4294967294, 1147483647, 3147483649, 4294966796, 4294967295, 50, 4294967295, 4294967295], dtype=torch.uint32
    )
    expected_tensor = ttnn.from_torch(
        expected_output_values,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, use_legacy=use_legacy)

    # Compare using ttnn.eq and check if all results are ones (all elements match)
    comparison_result = ttnn.eq(output_tensor, expected_tensor)
    comparison_torch = ttnn.to_torch(comparison_result, dtype=torch.bool)

    # Verify all comparisons are True (all elements match)
    assert torch.all(comparison_torch), f"Mismatch found in uint32 subtraction results"


# For inputs within int32 range [0, 2147483647]
def test_binary_add_uint32_lower_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 1147482, 2147483647, 1])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 1, 1147483, 0, 2147483646])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


# For inputs outside int32 range [2147483648, 4294967295]
def test_binary_add_uint32_upper_edge_cases(device):
    torch_input_tensor_a = torch.tensor([3000000000, 2750000000, 2147483648, 4294967292, 1, 4294967295, 0])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([1294967290, 1400000000, 2147483647, 2, 4294967291, 0, 4294967295])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    # Since ttnn.to_torch does not support int64, we cannot compare with the torch output. We can manually check the outputs:
    # Torch output: tensor([4294967290, 4150000000, 4294967295, 4294967294, 4294967292, 4294967295, 4294967295])
    # TT output: ttnn.Tensor([4294967290, 4150000000, 4294967295, 4294967294, 4294967292, 4294967295, 4294967295],
    #               shape=Shape([3]), dtype=DataType::UINT32, layout=Layout::TILE)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
@pytest.mark.parametrize("use_legacy", [True, False])
def test_bitwise_uint32(device, ttnn_function, use_legacy):
    x_torch = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0],
            [0, 4294967295, 2147483648, 4294967294, 4294967295, 1234567890],
        ],
        dtype=torch.uint32,
    )
    y_torch = torch.tensor(
        [
            [9, 3, 0, 1, 7, 0],
            [4294967295, 0, 2147483647, 2, 1, 4294967295],
        ],
        dtype=torch.uint32,
    )
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt, use_legacy=use_legacy)
    tt_out = ttnn.to_torch(z_tt_out, dtype=torch.uint32)

    assert torch.equal(z_torch, tt_out)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
@pytest.mark.parametrize("use_legacy", [True, False])
def test_bitwise_uint32_full_range(device, ttnn_function, use_legacy):
    x_values = torch.linspace(0, 4294967295, 1024, dtype=torch.float64)
    x_torch = x_values.to(dtype=torch.uint32)

    y_values = torch.linspace(4294967295, 0, 1024, dtype=torch.float64)
    y_torch = y_values.to(dtype=torch.uint32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt, use_legacy=use_legacy)
    tt_out = ttnn.to_torch(z_tt_out, dtype=torch.uint32)

    assert torch.equal(z_torch, tt_out)
