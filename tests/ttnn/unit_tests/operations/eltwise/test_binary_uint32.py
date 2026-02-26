# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

pytestmark = pytest.mark.use_module_device


def create_full_range_tensor(input_shape, dtype, value_ranges):
    num_elements = torch.prod(torch.tensor(input_shape)).item()

    num_ranges = len(value_ranges)
    elements_per_range = num_elements // num_ranges
    remainder = num_elements % num_ranges

    segments = []
    for i, (low, high) in enumerate(value_ranges):
        range_elements = elements_per_range + (1 if i < remainder else 0)

        segment = torch.linspace(low, high, steps=range_elements, dtype=dtype)
        segments.append(segment)

    in_data = torch.cat(segments)
    in_data = in_data.reshape(input_shape)
    return in_data


# Common test range for comparison and logical operations
COMMON_RANGE_A = [
    (0, 1000),
    (1e3, 1e6),
    (1e6, 1e8),
    (1e9, 2147483647),
]
COMMON_RANGE_B = [
    (0, 1500),
    (1e4, 1e7),
    (1e7, 1e9),
    (2e9, 2147483647),
]

BINARY_OP_TEST_CASES = [
    # Arithmetic ops
    (ttnn.add, [(0, 100), (1e3, 1e4), (1e6, 1e8), (1e9, 2e9)], [(0, 500), (1e4, 1e6), (1e7, 1e9), (1e7, 1e8)]),
    (ttnn.sub, [(100, 500), (1e5, 1e7), (1e7, 2e9), (2e9, 2147483647)], [(0, 100), (1e3, 1e5), (1e7, 1e9), (1e9, 2e9)]),
    (
        ttnn.mul,
        [(0, 100), (500, 1e3), (1e4, 1e5), (1e6, 1e7), (2e9, 2147483647)],
        [(0, 500), (1e3, 1e4), (1e2, 1e4), (1, 1e2), (1, 1)],
    ),
    (
        ttnn.squared_difference,
        [(0, 100), (500, 1000), (1500, 5000), (41000, 80000)],
        [(0, 500), (1000, 1500), (4500, 10000), (10000, 50000)],
    ),
    (
        ttnn.rsub,
        [
            (0, 100),
            (50000, 100000),
            (0, 10000),
        ],
        [(200, 400), (0, 4000), (20000, 2147483647)],
    ),
    # Comparison and logical ops
    *[
        (op, COMMON_RANGE_A, COMMON_RANGE_B)
        for op in (ttnn.eq, ttnn.ne, ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor)
    ],
]


@pytest.mark.parametrize("ttnn_op, value_ranges_a, value_ranges_b", BINARY_OP_TEST_CASES)
@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 128])),),
)
def test_binary_uint32_full_range(ttnn_op, value_ranges_a, value_ranges_b, input_shapes, device):
    torch_input_tensor_a = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.int32, value_ranges=value_ranges_a
    )
    torch_input_tensor_b = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.int32, value_ranges=value_ranges_b
    )

    if ttnn_op in {ttnn.logical_or, ttnn.logical_xor, ttnn.logical_and}:
        torch_input_tensor_a[..., ::5] = 0  # every 5th element across the last dim
        torch_input_tensor_b[..., ::10] = 0  # every 10th element across the last dim

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


@pytest.mark.parametrize(
    "ttnn_op, low_a, high_a, low_b, high_b",
    [
        (ttnn.add, 0, 1e4, 1e4, 1e8),
        (ttnn.sub, 50000, 100000, 0, 40000),
        (ttnn.mul, 0, 46340, 0, 46340),
        (ttnn.eq, 0, 1e8, 1e5, 1e9),
        (ttnn.ne, 0, 1e8, 1e5, 1e9),
        (ttnn.logical_and, 0, 1e8, 1e5, 1e9),
        (ttnn.squared_difference, 0, 32767, 32767, 65535),
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        pytest.param([(1, 1, 1), (8, 16, 32)], id="broadcast_lhs_1"),  # scalar bcast
        pytest.param([(1, 16, 1), (8, 1, 32)], id="broadcast_both_5"),  # mixed bcast
    ],
)
def test_binary_uint32_bcast(ttnn_op, input_shapes, low_a, high_a, low_b, high_b, device):
    a_shape, b_shape = input_shapes

    num_elements_a = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(low_a, high_a, num_elements_a, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements_a].reshape(a_shape)

    num_elements_b = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(low_b, high_b, num_elements_b, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements_b].reshape(b_shape)

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
        (ttnn.add, 0, 100, 100, 200),
        (ttnn.sub, 50000, 100000, 0, 40000),  # Subtraction: ensure a > b for valid results
        (ttnn.mul, 0, 23170, 23170, 46340),
        (ttnn.eq, 0, 1610612735, 536870911, 2147483647),
        (ttnn.ne, 0, 1610612735, 536870911, 2147483647),
        (ttnn.logical_and, 0, 1073741824, 1073741824, 2147483647),
        (ttnn.logical_or, 0, 1073741824, 1073741824, 2147483647),
        (ttnn.logical_xor, 0, 1073741824, 1073741824, 2147483647),
        (ttnn.squared_difference, 32767, 65535, 0, 32767),
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

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, use_legacy=use_legacy)

    # Since ttnn.to_torch does not support int64, we convert torch_output_tensor to uint32 and compare the results using ttnn.eq
    torch_output_tensor = ttnn.from_torch(
        torch_output_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    comparison_result = ttnn.eq(output_tensor, torch_output_tensor)
    comparison_torch = ttnn.to_torch(comparison_result)

    # Verify all comparisons are True (all elements match)
    assert torch.all(comparison_torch), f"Mismatch found in uint32 subtraction results"
    # Torch output: tensor([4294967294, 1147483647, 3147483649, 4294966796, 4294967295, 50, 4294967295, 4294967295])
    # TT output: ttnn.Tensor([4294967294, 1147483647, 3147483649, 4294966796, 4294967295, 50, 4294967295, 4294967295],
    #               shape=Shape([8]), dtype=DataType::UINT32, layout=Layout::TILE)


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

    # Since ttnn.to_torch does not support int64, we convert torch_output_tensor to uint32 and compare the results using ttnn.eq
    torch_output_tensor = ttnn.from_torch(
        torch_output_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    comparison_result = ttnn.eq(output_tensor, torch_output_tensor)
    comparison_torch = ttnn.to_torch(comparison_result)

    # Verify all comparisons are True (all elements match)
    assert torch.all(comparison_torch), f"Mismatch found in uint32 addition results"
    # Torch output: tensor([4294967290, 4150000000, 4294967295, 4294967294, 4294967292, 4294967295, 4294967295])
    # TT output: ttnn.Tensor([4294967290, 4150000000, 4294967295, 4294967294, 4294967292, 4294967295, 4294967295],
    #               shape=Shape([7]), dtype=DataType::UINT32, layout=Layout::TILE)


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


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.logical_and,
        ttnn.logical_or,
        ttnn.logical_xor,
    ],
)
def test_binary_comp_logical_ops_uint32_edge_cases(ttnn_op, device):
    torch_input_tensor_a = torch.tensor(
        [0, 1, 0, 2147483647, 2147483647, 2147483647, 1073741823, 1073741823, 4294967295, 4294967294, 4294967295]
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor(
        [0, 0, 1, 2147483647, 2147483646, 0, 1000, 1073741823, 4294967295, 4294967295, 0]
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device).to(torch.int64)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor).to(torch.int64)

    assert torch.equal(output_tensor, torch_output_tensor)


# For inputs within int32 range [0, 2147483647]
def test_binary_mul_uint32_lower_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 2147483647, 1, 1073741823, 715827882, 46340])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 1, 1, 2147483646, 2, 3, 46340])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


# For inputs outside int32 range [2147483648, 4294967295]
def test_binary_mul_uint32_upper_edge_cases(device):
    torch_input_tensor_a = torch.tensor(
        [4294967295, 1, 4294967295, 4294967294, 2147483647, 1431655765, 16777215, 65535]
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([1, 4294967295, 0, 1, 2, 3, 256, 65535])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b)

    # Since ttnn.to_torch does not support int64, we convert torch_output_tensor to uint32 and compare the results using ttnn.eq
    torch_output_tensor = ttnn.from_torch(
        torch_output_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    comparison_result = ttnn.eq(output_tensor, torch_output_tensor)
    comparison_torch = ttnn.to_torch(comparison_result)

    # Verify all comparisons are True (all elements match)
    assert torch.all(comparison_torch), f"Mismatch found in uint32 multiplication results"
    # Torch output: tensor([4294967295, 4294967295, 0, 4294967294, 4294967294, 4294967295, 4294967040, 4294836225])
    # TT output: ttnn.Tensor([4294967295, 4294967295, 0, 4294967294, 4294967294, 4294967295, 4294967040, 4294836225],
    #               shape=Shape([8]), dtype=DataType::UINT32, layout=Layout::TILE)


def test_binary_squared_difference_uint32_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 5, 10, 65535, 0, 4294967295, 4294901760, 4294967295])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_input_tensor_b = torch.tensor([0, 0, 1, 10, 2, 65535, 65535, 4294901760, 4294967295, 4294901761])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.squared_difference)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn.squared_difference(input_tensor_a, input_tensor_b)

    # Since ttnn.to_torch does not support uint32 to int64 conversion, we convert torch_output_tensor to uint32 and compare the results using ttnn.eq
    torch_output_tensor = ttnn.from_torch(
        torch_output_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    comparison_result = ttnn.eq(output_tensor, torch_output_tensor)
    comparison_torch = ttnn.to_torch(comparison_result)

    # Verify all comparisons are True (all elements match)
    assert torch.all(comparison_torch), f"Mismatch found in uint32 squared_difference results"
    # Torch output: tensor([    0,     1,     1,    25,    64,     0, 4294836225, 4294836225, 4294836225, 4294705156])
    # TT output: ttnn.Tensor([    0,     1,     1,    25,    64,     0, 4294836225, 4294836225, 4294836225, 4294705156],
    #               shape=Shape([10]), dtype=DataType::UINT32, layout=Layout::TILE)


def test_binary_rsub_uint32_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 0, 2147483647, 2147483646, 2147483640, 4294967292, 6])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_input_tensor_b = torch.tensor([0, 1, 2147483647, 2147483647, 4294967295, 4294967295, 4294967295])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.rsub(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.rsub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)
    # Since ttnn.to_torch does not support uint32 to int64 conversion, we convert torch_output_tensor to uint32 and compare the results using ttnn.eq
    torch_output_tensor = ttnn.from_torch(
        torch_output_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    comparison_result = ttnn.eq(output_tensor, torch_output_tensor)
    comparison_torch = ttnn.to_torch(comparison_result)

    # Verify all comparisons are True (all elements match)
    assert torch.all(comparison_torch), f"Mismatch found in uint32 rsub results"
    # Torch output: tensor([      0,     1,     0,     1, 2147483655,     3, 4294967289])
    # TT output: ttnn.Tensor([      0,     1,     0,     1, 2147483655,     3, 4294967289],
    #               shape=Shape([7]), dtype=DataType::UINT32, layout=Layout::TILE)
