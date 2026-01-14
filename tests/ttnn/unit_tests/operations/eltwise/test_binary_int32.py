# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_ulp

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


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -100, 100),
        (-300, 300, -250, 250),
        (-500, 500, -750, 750),
        (-1000, 1000, -500, 1000),
        (-1e4, 1e4, -5e3, 5e3),
        (2e9, 2077000000, 2e9, 2147483647),  # large positive input
        (-2147483647, -2e9, -2077000000, -2e9),  # large negative input
        (-2147483647, 2147483647, -2147483647, 2147483647),  # full range
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
        ttnn.add,
        ttnn.sub,
        ttnn.squared_difference,
        ttnn.rsub,
    ],
)
@pytest.mark.parametrize("use_legacy", [True, False])
def test_binary_int32(input_shapes, low_a, high_a, low_b, high_b, ttnn_op, use_legacy, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)

    if ttnn_op in {ttnn.logical_or, ttnn.logical_xor, ttnn.logical_and}:
        torch_input_tensor_a[::5] = 0  # every 5th element is zero
        torch_input_tensor_b[::10] = 0  # every 10th element is zero

    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
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
    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, use_legacy=use_legacy)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


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
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -100, 100),
        (-300, 300, -250, 250),
        (-500, 500, -750, 750),
        (-1000, 1000, -500, 1000),
        (-1e4, 1e4, -5e3, 5e3),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
        ttnn.squared_difference,
        ttnn.rsub,
    ],
)
def test_binary_int32_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, ttnn_op, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

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
    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor)

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
@pytest.mark.parametrize(
    "ttnn_fn", ("logical_or", "logical_xor", "logical_and", "add", "sub", "mul", "squared_difference", "rsub")
)
def test_binary_int32_sharded(a_shape, b_shape, sharded_config, ttnn_fn, device):
    ttnn_op = getattr(ttnn, ttnn_fn)
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(-100, 100, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(-200, 200, num_elements, dtype=torch.int32)
    torch_input_tensor_b[::10] = 0
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, memory_config=sharded_config, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
    ],
)
def test_binary_logical_int32_edge_cases(logical_op, device):
    torch_input_tensor_a = torch.tensor(
        [0, 1, 0, 1, -1, 2147483647, -2147483647, 2147483647, 0, 1073872896, -1073872896]
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor(
        [0, 0, -1, 1, -1, 2147483647, -2147483647, 0, -2147483647, 1073872896, -1073872896]
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = logical_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_left_shift,
        ttnn.logical_left_shift,
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.int32,
        ttnn.uint32,
    ],
)
def test_binary_left_shift(device, ttnn_function, ttnn_dtype):
    # Test with regular values and extreme values for both int32 and uint32
    if ttnn_dtype == ttnn.int32:
        x_torch = torch.tensor(
            [[99, 3, 100, 1, 72, 0, -100, 22, 12, 1000, 2147483647, -2147483648, -1, 1]], dtype=torch.int32
        )  # Include int32 extremes

    else:  # ttnn.uint32
        # For uint32, test with values that represent the full uint32 range
        # Note: 4294967295 (uint32 max) is represented as -1 in int32 two's complement
        x_torch = torch.tensor(
            [[99, 3, 100, 1, 72, 0, 5, 22, 12, 1000, 0, -1, 2147483647, -2147483648]], dtype=torch.int32
        )  # uint32 extremes as int32

    y_torch = torch.tensor([[1, 2, 31, 4, 5, 0, -20, 1, -3, -25, 0, 1, 31, 30]], dtype=torch.int32)

    if ttnn_dtype == ttnn.uint32:  # Stimulate uint32 input
        x_uint32 = x_torch.to(torch.int64) & 0xFFFFFFFF
        y_uint32 = y_torch.to(torch.int64) & 0xFFFFFFFF
        x_torch = x_uint32.to(torch.int32)
        y_torch = y_uint32.to(torch.int32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    assert torch.equal(tt_out, z_torch)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.bitwise_right_shift,
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.int32,
        ttnn.uint32,
    ],
)
def test_bitwise_right_shift(device, ttnn_function, ttnn_dtype):
    x_torch = torch.tensor(
        [
            [
                19,
                101,
                21,
                47,
                0,
                -4,
                -99,
                -278,
                1000,
                99999,
                -99999,
                -7544,
                1,
                -1,
                2**31 - 1,
                -(2**31),
                123456789,
                -123456789,
            ]
        ],
        dtype=torch.int32,
    )

    y_torch = torch.tensor([[5, 31, 4, 5, 0, 1, 4, 1, 32, 66, 1, 14, 0, 1, 31, 31, 1, 5]], dtype=torch.int32)
    if ttnn_dtype == ttnn.uint32:  # Stimulate uint32 input
        x_uint32 = x_torch.to(torch.int64) & 0xFFFFFFFF
        y_uint32 = y_torch.to(torch.int64) & 0xFFFFFFFF
        x_torch = x_uint32.to(torch.int32)
        y_torch = y_uint32.to(torch.int32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_out)

    assert torch.equal(tt_out, z_torch)


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.logical_right_shift,
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.int32,
        ttnn.uint32,
    ],
)
@pytest.mark.parametrize("use_legacy", [True, False])
def test_logical_right_shift(device, ttnn_function, ttnn_dtype, use_legacy):
    x_torch = torch.tensor(
        [
            [
                19,
                101,
                21,
                47,
                0,
                -4,
                -99,
                -278,
                1000,
                99999,
                -99999,
                -7544,
                1,
                -1,
                2**31 - 1,
                -(2**31),
                123456789,
                -123456789,
            ]
        ],
        dtype=torch.int32,
    )

    y_torch = torch.tensor([[5, 31, 4, 5, 0, 1, 4, 1, 32, 66, 1, 14, 0, 1, 31, 31, 1, 5]], dtype=torch.int32)
    if ttnn_dtype == ttnn.uint32:  # Stimulate uint32 input
        x_uint32 = x_torch.to(torch.int64) & 0xFFFFFFFF
        y_uint32 = y_torch.to(torch.int64) & 0xFFFFFFFF
        x_torch = x_uint32.to(torch.int32)
        y_torch = y_uint32.to(torch.int32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn_function(x_tt, y_tt, use_legacy=use_legacy)
    tt_out = ttnn.to_torch(z_tt_out)

    assert torch.equal(tt_out, z_torch)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-500, 500, -750, 750),
        (-1e3, 1e3, -1e5, 1e5),
        (-450, 450, -1e6, 1e6),
        (0, 46340, 0, 46340),
        (0, -46340, 0, 46340),
        # large inputs
        (-3, 3, 536870911, 715827882),
        (-2, 2, -715827882, -1073741823),
        (-2, 2, 715827882, 1073741823),
        (-1, 1, 1073741823, 2147483647),
        (-1, 1, -2147483648, -1073741823),
    ],
)
def test_binary_mul_int32(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    if high_a in (3, 2, 1):
        values_a = torch.arange(low_a, high_a + 1, dtype=torch.int32)
        torch_input_tensor_a = values_a[torch.randint(0, len(values_a), (num_elements,))]
    else:
        torch_input_tensor_a = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)

    if high_b in (3, 2, 1):
        values_b = torch.arange(low_b, high_b + 1, dtype=torch.int32)
        indices_b = torch.randint(0, len(values_b), (num_elements,))
        torch_input_tensor_b = values_b[indices_b]
    else:
        torch_input_tensor_b = torch.linspace(low_b, high_b, num_elements, dtype=torch.int32)

    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

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
    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize("use_legacy", [True, False])
def test_binary_mul_int32_edge_cases(use_legacy, device):
    torch_input_tensor_a = torch.tensor(
        [
            0,
            -0,
            -1,
            1,
            2147483647,  # upper int32 limit
            -2147483648,  # lower int32 limit
            1073741823,
            -536870911,
            51130563,
            131071,
            -1000,
            -10000,
            # test for overflowing outputs to ensure int32 arithmetic behaviour
            1073741824,
            -1073741824,
            -99999999,
            3457894,
        ],
        dtype=torch.int32,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor(
        [0, -1, -2147483647, 1e8, 1, 1, -2, 3, -40, 16384, -1e3, 1e4, 2, -9867, 5e4, 63835], dtype=torch.int32
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # If the result of two inputs exceeds the int32 range, it wraps around due to overflow.
    # For example, 2147483647 * 2 outputs -2 instead of 4294967294 because
    # 4294967294 (decimal) = 0xFFFFFFFE (hex)
    # Interpreted as int32 (2's complement), 0xFFFFFFFE represents -2.

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b, use_legacy=use_legacy)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "shapes",
    [
        # scalar bcast
        pytest.param([(1, 1, 1), (8, 16, 32)], id="broadcast_lhs_1"),
        pytest.param([(8, 16, 32), (1, 1, 1)], id="broadcast_rhs_1"),
        # no subtile bcast
        pytest.param([(1, 16, 32), (8, 16, 32)], id="broadcast_lhs_2"),
        pytest.param([(8, 16, 32), (1, 16, 32)], id="broadcast_rhs_2"),
        # row bcast
        pytest.param([(8, 1, 32), (8, 16, 32)], id="broadcast_lhs_3"),
        pytest.param([(8, 16, 32), (8, 1, 32)], id="broadcast_rhs_3"),
        pytest.param([(1, 1, 32), (8, 16, 32)], id="broadcast_lhs_4"),
        pytest.param([(8, 16, 32), (1, 1, 32)], id="broadcast_rhs_4"),
        # col bcast
        pytest.param([(8, 16, 1), (8, 16, 32)], id="broadcast_lhs_5"),
        pytest.param([(8, 16, 32), (8, 16, 1)], id="broadcast_rhs_5"),
        pytest.param([(1, 16, 1), (8, 16, 32)], id="broadcast_lhs_6"),
        pytest.param([(8, 16, 32), (1, 16, 1)], id="broadcast_rhs_6"),
        # row-col mixed bcast
        pytest.param([(1, 1, 32), (8, 16, 1)], id="broadcast_both_1"),
        pytest.param([(8, 16, 1), (1, 1, 32)], id="broadcast_both_2"),
        pytest.param([(8, 1, 32), (8, 16, 1)], id="broadcast_both_3"),
        pytest.param([(8, 16, 1), (8, 1, 32)], id="broadcast_both_4"),
        pytest.param([(1, 16, 1), (8, 1, 32)], id="broadcast_both_5"),
        pytest.param([(8, 1, 32), (1, 16, 1)], id="broadcast_both_6"),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.lt,
        ttnn.gt,
        ttnn.ge,
        ttnn.le,
        ttnn.div,
    ],
)
def test_binary_implicit_broadcast(device, shapes, ttnn_op):
    torch.manual_seed(0)

    min_int = torch.iinfo(torch.int32).min
    max_int = torch.iinfo(torch.int32).max
    torch_input_tensor_a = torch.randint(low=min_int, high=max_int, size=shapes[0], dtype=torch.int32)
    torch_input_tensor_b = torch.randint(low=min_int, high=max_int, size=shapes[1], dtype=torch.int32)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    if ttnn_op == ttnn.div:
        assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-6, equal_nan=False)
    else:
        assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.lt,
        ttnn.gt,
        ttnn.ge,
        ttnn.le,
    ],
)
def test_comp_ops_edge_cases(ttnn_op, device):
    torch_input_tensor_a = torch.tensor(
        [0, 1, 0, 0, 1254, 43, 2147483647, -2147483648, 2147483647, 0, -123456789, -56738943, 2147483647, -2147483648]
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor(
        [0, 0, -1, 2, 324, 53342, 2147483647, -2147483648, 0, -2147483648, -3, -5, -2, 2]
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 128])),),
)
def test_binary_div_int32_full_range(input_shapes, device):
    value_ranges_a = [
        (-300, 300),
        (-500, 500),
        (-1000, 1000),
        (-1e4, 1e4),
        (-1e5, 1e5),
        (-1e7, 1e7),
        (2e9, 2077000000),  # large positive input
        (-2147483647, -2e9),  # large negative input
        (-2147483647, 2147483647),  # full range
        (-2147483647, 2147483647),  # large numerator
        (-10, 10),  # small numerator
    ]

    value_ranges_b = [
        (-250, 250),
        (-750, 750),
        (-500, 1000),
        (-5e3, 5e3),
        (-5e4, 5e4),
        (-1e6, 1e6),
        (2e9, 2147483647),  # large positive input
        (-2077000000, -2e9),  # large negative input
        (-2147483647, 2147483647),  # full range
        (-10, 10),  # small denominator
        (-2147483647, 2147483647),  # large denominator
    ]

    torch_input_tensor_a = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.int32, value_ranges=value_ranges_a
    )
    torch_input_tensor_b = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.int32, value_ranges=value_ranges_b
    )

    torch_input_tensor_b[
        torch_input_tensor_b == 0
    ] = 1  # avoid division by zero since nan and inf are not representable in int32

    golden_function = ttnn.get_golden_function(ttnn.div)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

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

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-6, equal_nan=False)
    assert_with_ulp(output_tensor, torch_output_tensor, ulp_threshold=2.0)


def test_div_int32_optional_output(device):
    torch_input_tensor_a = torch.arange(-(2**23), 2**23, 1024, dtype=torch.int32)
    torch_input_tensor_b = torch.arange(-(2**23) - 1, 2**23 - 1, 1024, dtype=torch.int32)
    torch_input_tensor_b[torch_input_tensor_b == 0] = 1
    zeros_tensor = torch.zeros_like(torch_input_tensor_a, dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.div)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    preallocated_tensor = ttnn.from_torch(zeros_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.div(input_tensor_a, input_tensor_b, output_tensor=preallocated_tensor)
    output_tensor = ttnn.to_torch(preallocated_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-6, equal_nan=False)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-300, 300, -250, 250),
        (-500, 500, -750, 750),
        (-1000, 1000, -500, 1000),
        (-1e4, 1e4, -5e7, 5e7),
        (2e9, 2077000000, 2e9, 2147483647),  # large positive input
        (-2147483647, -2e9, -2077000000, -2e9),  # large negative input
        (-2147483647, 2147483647, -2147483647, 2147483647),  # full range
        (-10, 10, -2147483647, 2147483647),  # small numerator, large denominator
        (-2147483647, 2147483647, -10, 10),  # large numerator, small denominator
        # a=-2147483648 and b=-1 is not supported
        (-2147483648, 2147483647, -2147483648, -2),
        (-2147483648, 2147483647, 1, 2147483647),
        (2021531526, 2147483647, 9, 123),
    ],
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_int32_rounding_modes(input_shapes, low_a, high_a, low_b, high_b, rounding_mode, device):
    # Skip some cases for rounding_mode==None that aren't supported due to:
    # https://github.com/tenstorrent/tt-metal/issues/33334
    if rounding_mode is None and low_a == -2147483648:
        pytest.skip("a == -2147483648 is not supported for rounding_mode=None")

    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes)

    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(input_shapes)

    torch_input_tensor_b[
        torch_input_tensor_b == 0
    ] = 1  # avoid division by zero since nan and inf are not representable in int32

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

    golden_function = ttnn.get_golden_function(ttnn.div)
    torch_output_tensor = golden_function(
        torch_input_tensor_a, torch_input_tensor_b, rounding_mode=rounding_mode, device=device
    )

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode)
    output_tensor = ttnn.to_torch(output_tensor)

    if rounding_mode is not None:
        assert_equal(torch_output_tensor, output_tensor)
    else:
        assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-6, equal_nan=False)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_edge_cases(rounding_mode, device):
    pairs = [
        (16777215, 1),
        (16777216, 2),
        (16777217, -7),
        (-16777215, 3),
        (16777216, -3),
        (-16777216, -4),
        (-16777217, -5),
        (2147483647, 1),
        (-2147483647, 1),
        (2147483647, -1e7),
        (-2147483647, 1e4),
        (2147483647, -2147483647),
        (-2147483647, 2147483647),
        (2147483647, 2147483647),
        (-2147483647, -2147483647),
        (2147483647, 1073741823),
        (1073741823, -2147483647),
        (1073741824, -2147483647),
    ]

    numerators, denominators = zip(*pairs)
    torch_input_tensor_a = torch.tensor(numerators, dtype=torch.int32)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor(denominators, dtype=torch.int32)
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.div)
    torch_output_tensor = golden_function(
        torch_input_tensor_a, torch_input_tensor_b, rounding_mode=rounding_mode, device=device
    )

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b, rounding_mode=rounding_mode)
    output_tensor = ttnn.to_torch(output_tensor)

    if rounding_mode is None:
        assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-6, equal_nan=False)
    else:
        assert torch.equal(torch_output_tensor, output_tensor)


def test_div_inf_nan_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, -1, 0, 0, 1, -1, -1, 1, 2147483647, 0], dtype=torch.int32)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 0, 1, -1, 1, -1, 1, -1, 0, -2147483647], dtype=torch.int32)
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.div)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.div(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 2, 32, 128])),),
)
def test_binary_divide_int32_full_range(input_shapes, device):
    value_ranges_a = [
        (-300, 300),
        (-750, 500),
        (-1000, 1000),
        (-1e4, 1e4),
        (-1e5, 1e5),
        (-1e7, 1e7),
        (-16777216, 16777216),  # full fp32 int range
        (1e8, 16777216),  # large positive input
        (-16777216, -1e8),  # large negative input
        (-16777216, 16777216),  # large numerator
        (-10, 10),  # small numerator
    ]

    value_ranges_b = [
        (-250, 250),
        (-750, 750),
        (-500, 1000),
        (-5e3, 5e3),
        (-5e4, 5e4),
        (-1e6, 1e6),
        (-16777216, 16777216),  # full fp32 int range
        (1.5e7, 16777216),  # large positive input
        (-16777216, -1e7),  # large negative input
        (-10, 10),  # large numerator
        (-16777216, 16777216),  # small numerator
    ]

    torch_input_tensor_a = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.int32, value_ranges=value_ranges_a
    )
    torch_input_tensor_b = create_full_range_tensor(
        input_shape=input_shapes, dtype=torch.int32, value_ranges=value_ranges_b
    )

    torch_input_tensor_b[
        torch_input_tensor_b == 0
    ] = 1  # avoid division by zero since nan and inf are not representable in int32

    golden_function = ttnn.get_golden_function(ttnn.divide)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

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

    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b)

    ARCH_NAME = ttnn.get_arch_name()
    if "blackhole" in ARCH_NAME:
        assert_with_ulp(output_tensor, torch_output_tensor, ulp_threshold=2.0)
    elif "wormhole" in ARCH_NAME:
        assert_with_ulp(output_tensor, torch_output_tensor, ulp_threshold=1.0)


def test_divide_edge_cases(device):
    pairs = [
        (3, 2),
        (2, 2),
        (10, 3),
        (20, 2),
        (16777215, 1),
        (16777216, 2),
        (-16777215, 3),
        (16777216, -3),
        (-16777216, -4),
        (16777216, 16777215),
        (-16777229, 19),
        (-16777229, -8388615),
        (16777230, 8388615),
    ]

    numerators, denominators = zip(*pairs)
    torch_input_tensor_a = torch.tensor(numerators, dtype=torch.int32)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor(denominators, dtype=torch.int32)
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.divide)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b)

    assert_with_ulp(output_tensor, torch_output_tensor, ulp_threshold=1.0)


def test_divide_inf_nan_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, -1, 0, 0, 1, -1, -1, 1, 2147483647, 0], dtype=torch.int32)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 0, 1, -1, 1, -1, 1, -1, 0, -2147483647], dtype=torch.int32)
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.divide)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(torch_output_tensor, output_tensor, atol=1e-10, rtol=1e-5, equal_nan=True)


def test_binary_scalar_div_int32(device):
    torch_dtype = torch.int32
    ttnn_dtype = ttnn.int32

    x_torch = torch.tensor([[1000, -1000, 1000, -1999]], dtype=torch_dtype)
    y_torch = 500
    z_torch = torch.divide(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = 500
    z_tt = ttnn.divide(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt)

    z_tt_floor = ttnn.divide(x_tt, y_tt, rounding_mode="floor")
    tt_out_floor = ttnn.to_torch(z_tt_floor)
    z_torch_floor = torch.divide(x_torch, y_torch, rounding_mode="floor")

    z_tt_trunc = ttnn.divide(x_tt, y_tt, rounding_mode="trunc")
    tt_out_trunc = ttnn.to_torch(z_tt_trunc)
    z_torch_trunc = torch.divide(x_torch, y_torch, rounding_mode="trunc")

    assert torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=True)
    assert torch.equal(z_torch_floor, tt_out_floor)
    assert torch.equal(z_torch_trunc, tt_out_trunc)
