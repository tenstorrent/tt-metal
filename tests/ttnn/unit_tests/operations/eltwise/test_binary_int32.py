# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


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
        ttnn.add,
        ttnn.sub,
    ],
)
def test_binary_int32(input_shapes, low_a, high_a, low_b, high_b, ttnn_op, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)

    if ttnn_op in {ttnn.logical_or, ttnn.logical_xor}:
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
    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
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
        ttnn.add,
        ttnn.sub,
        ttnn.mul,
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
@pytest.mark.parametrize("ttnn_fn", ("logical_or", "logical_xor", "add", "sub", "mul"))
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
    ],
)
def test_binary_logical_int32_edge_cases(logical_op, device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 1, -1, 2147483647, -2147483647, 2147483647, 0])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, -1, 1, -1, 2147483647, -2147483647, 0, -2147483647])
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
        # large outputs
        (0, 46340, 0, 46340),
        (0, -46340, 0, 46340),
        # large inputs
        (-3, 3, 536870911, 715827882),
        (-2, 2, -715827882, -1073741823),
        (-2, 2, 715827882, 1073741823),
        (-1, 1, 1073741823, 2147483647),
        (-1, 1, -2147483647, -1073741823),
    ],
)
def test_binary_mul_int32(input_shapes, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    if high_a in (3, 2, 1):
        values_a = torch.arange(low_a, high_a + 1, dtype=torch.int32)
        torch_input_tensor_a = values_a[torch.randint(0, len(values_a), (num_elements,))]
    else:
        torch_input_tensor_a = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)

    if high_b in (32, 4, 2, 1):
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
        [0, -0, -1, 1, 2147483647, -2147483647, 1073741823, -536870911, 51130563, 131071, -1000, -10000]
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, -1, -2147483647, 100000000, 1, 1, -2, 3, -40, 16384, -1000, 10000])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b, use_legacy=use_legacy)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)
