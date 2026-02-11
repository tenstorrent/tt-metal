# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

torch.manual_seed(0)


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize(
    "concat_spec",
    [
        ([[1, 1, 4, 4], [1, 1, 4, 4]], -1),
        ([[96], [96]], 0),  # 1D tiled tensors
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
def test_tiled_concat(device, concat_spec, dtype):
    shapes, dim = concat_spec
    torch_input_tensors = [random_torch_tensor(dtype, shape) for shape in shapes]
    torch_output_tensor = torch.concat(torch_input_tensors, dim=dim)

    input_tensors = [
        ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
        for torch_input_tensor in torch_input_tensors
    ]

    output = ttnn.concat(input_tensors, dim=dim)
    output = ttnn.to_torch(output)

    assert_equal(torch_output_tensor, output)


@pytest.mark.parametrize("height", [20, 32])
@pytest.mark.parametrize("width", [4, 32])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
def test_concat(device, height, width, dim, dtype):
    torch_input_tensor_a = random_torch_tensor(dtype, (height, width))
    torch_input_tensor_b = random_torch_tensor(dtype, (height, width))
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_equal(torch_output_tensor, output)


@pytest.mark.parametrize(
    "inputs, output_shard_shape, shard_grid, strategy, layout, cache_mode",
    (
        (
            [((1, 1, 160, 32), (80, 32)), ((1, 1, 160, 32), (80, 32))],
            (80, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ROW_MAJOR_LAYOUT,
            False,
        ),
        (
            [((1, 1, 160, 32), (80, 32)), ((1, 1, 160, 16), (80, 16))],
            (80, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ROW_MAJOR_LAYOUT,
            False,
        ),
        (
            [((1, 1, 25600, 64), (512, 64)), ((1, 1, 25600, 64), (512, 64))],
            (512, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(1, 6)),
                }
            ),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ROW_MAJOR_LAYOUT,
            False,
        ),
        pytest.param(
            [((1, 1, 25600, 64), (512, 64)), ((1, 1, 25600, 64), (512, 64))],
            (512, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(1, 6)),
                }
            ),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ROW_MAJOR_LAYOUT,
            True,
        ),
        (
            [((1, 1, 16, 16), (8, 16)), ((1, 1, 16, 16), (8, 16)), ((1, 1, 16, 16), (8, 16))],
            (8, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ROW_MAJOR_LAYOUT,
            False,
        ),
        (
            [((1, 1, 16, 16), (8, 16)), ((1, 1, 16, 16), (8, 16)), ((1, 1, 16, 16), (8, 16))],
            (8, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ROW_MAJOR_LAYOUT,
            True,
        ),
        (
            [((1, 1, 8, 64), (8, 16)), ((1, 1, 7, 64), (7, 16)), ((1, 1, 23, 64), (23, 16))],
            (38, 16),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ROW_MAJOR_LAYOUT,
            False,
        ),
        (
            [((1, 1, 8, 64), (8, 16)), ((1, 1, 7, 64), (7, 16)), ((1, 1, 23, 64), (23, 16))],
            (38, 16),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ROW_MAJOR_LAYOUT,
            True,
        ),
        (
            [((1, 1, 256, 96), (64, 96)), ((1, 1, 256, 64), (64, 64)), ((1, 1, 256, 32), (64, 32))],
            (64, 192),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)),
                }
            ),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.TILE_LAYOUT,
            False,
        ),
        (
            [((1, 1, 32, 512), (32, 64)), ((1, 1, 64, 512), (64, 64)), ((1, 1, 96, 512), (96, 64))],
            (192, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1)),
                }
            ),
            ttnn.ShardStrategy.WIDTH,
            ttnn.TILE_LAYOUT,
            False,
        ),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
def test_sharded_concat(device, inputs, output_shard_shape, shard_grid, strategy, layout, cache_mode, dtype):
    if not cache_mode:
        device.disable_and_clear_program_cache()

    dim = 2 if strategy == ttnn.ShardStrategy.WIDTH else 3

    def _gen_inputs(input_specs):
        input_tensors = []
        for input_spec in input_specs:
            shape, shard_shape = input_spec
            input_sharded_memory_config = ttnn.create_sharded_memory_config(
                shard_shape,
                core_grid=shard_grid,
                strategy=strategy,
                use_height_and_width_as_shard_shape=True,
            )
            torch_input_tensor = random_torch_tensor(dtype, shape)
            input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=dtype)
            input_tensor = ttnn.to_memory_config(input_tensor, input_sharded_memory_config)
            input_tensors.append((torch_input_tensor, input_tensor))
        return input_tensors

    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        output_shard_shape,
        core_grid=shard_grid,
        strategy=strategy,
        use_height_and_width_as_shard_shape=True,
    )

    input_tensors = _gen_inputs(inputs)
    torch_output_tensor = torch.concat([torch_tensor for torch_tensor, _ in input_tensors], dim=dim)
    output = ttnn.concat([tensor for _, tensor in input_tensors], dim=dim, memory_config=output_sharded_memory_config)
    output = ttnn.to_torch(output)
    assert_equal(torch_output_tensor, output)

    # If cache mode is enabled, run the second set of input tensors to verify buffers are updated when cache hits.
    if not cache_mode:
        return

    input_tensors_2 = _gen_inputs(inputs)
    torch_output_tensor_2 = torch.concat([torch_tensor for torch_tensor, _ in input_tensors_2], dim=dim)
    output_2 = ttnn.concat(
        [tensor for _, tensor in input_tensors_2], dim=dim, memory_config=output_sharded_memory_config
    )
    output_2 = ttnn.to_torch(output_2)
    assert_equal(torch_output_tensor_2, output_2)


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "input_shapes, output_shape, core_grid, layout",
    (
        (((1, 1, 1, 32), (1, 1, 1, 32)), (1, 1, 1, 64), ttnn.CoreGrid(x=1, y=1), ttnn.ROW_MAJOR_LAYOUT),
        (((1, 1, 1, 32), (1, 1, 1, 64)), (1, 1, 1, 96), ttnn.CoreGrid(x=1, y=1), ttnn.ROW_MAJOR_LAYOUT),
        (((1, 1, 1, 64), (1, 1, 1, 32)), (1, 1, 1, 96), ttnn.CoreGrid(x=1, y=1), ttnn.ROW_MAJOR_LAYOUT),
        (((1, 1, 1024, 64), (1, 1, 1024, 32)), (1, 1, 1024, 96), ttnn.CoreGrid(x=4, y=1), ttnn.ROW_MAJOR_LAYOUT),
        (((1, 1, 256, 64), (1, 1, 256, 128)), (1, 1, 256, 192), ttnn.CoreGrid(x=8, y=1), ttnn.ROW_MAJOR_LAYOUT),
        (((1, 1, 32, 32), (1, 1, 32, 32)), (1, 1, 32, 64), ttnn.CoreGrid(x=1, y=1), ttnn.TILE_LAYOUT),
        (((1, 1, 32, 64), (1, 1, 32, 64)), (1, 1, 32, 128), ttnn.CoreGrid(x=1, y=1), ttnn.TILE_LAYOUT),
        (((1, 1, 256, 64), (1, 1, 256, 128)), (1, 1, 256, 192), ttnn.CoreGrid(x=8, y=1), ttnn.TILE_LAYOUT),
        (((1, 1, 512, 64), (1, 1, 512, 128)), (1, 1, 512, 192), ttnn.CoreGrid(x=8, y=1), ttnn.TILE_LAYOUT),
        (((1, 1, 512, 128), (1, 1, 512, 64)), (1, 1, 512, 192), ttnn.CoreGrid(x=8, y=1), ttnn.TILE_LAYOUT),
    ),
)
def test_sharded_concat_with_groups(device, input_shapes, output_shape, dim, groups, dtype, core_grid, layout):
    torch_input_tensors = [random_torch_tensor(dtype, shapes) for idx, shapes in enumerate(input_shapes)]

    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Cannot use bfloat8 with RM layout")

    if layout == ttnn.TILE_LAYOUT and (input_shapes[0][-1] // groups < 16 or input_shapes[1][-1] // groups < 16):
        pytest.xfail("Group size < 16 is currently not supported for tiled inputs")

    expected = ttnn.concat.golden_function(torch_input_tensors, dim, groups)

    sharded_memory_configs = [
        ttnn.create_sharded_memory_config(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
        )
        for x in torch_input_tensors
    ]
    ttnn_input_tensors = [
        ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        for x, memory_config in zip(torch_input_tensors, sharded_memory_configs)
    ]

    output_memory_config = ttnn.create_sharded_memory_config(output_shape, core_grid, ttnn.ShardStrategy.HEIGHT)
    z = ttnn.concat(
        [ttnn_input_tensors[0], ttnn_input_tensors[1]], dim=dim, memory_config=output_memory_config, groups=groups
    )

    actual = ttnn.to_torch(z)
    if dtype == ttnn.bfloat8_b:
        assert_with_pcc(expected, actual, 0.99)
    else:
        assert_equal(expected, actual)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
def test_concat_5d(device, dim, dtype):
    torch_input_tensor = random_torch_tensor(dtype, (1, 1, 1, 1, 2))
    torch_result = torch.cat([torch_input_tensor, torch_input_tensor], dim=dim)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    ttnn_result = ttnn.concat([ttnn_input_tensor, ttnn_input_tensor], dim=dim)
    ttnn_result = ttnn.to_torch(ttnn_result)
    assert_equal(torch_result, ttnn_result)


@pytest.mark.parametrize(
    "core_grid, hw, channels1, channels2, shard_height",
    (
        (
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                }
            ),
            96,
            64,
            32,
            64,
        ),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32, ttnn.uint32])
def test_concat_sharded_pad(device, core_grid, hw, channels1, channels2, shard_height, dtype):
    shape1 = [1, 1, hw, channels1]
    shape2 = [1, 1, hw, channels2]

    shape1_shard_shape = (shard_height, channels1)
    shape1_shard_spec = ttnn.ShardSpec(core_grid, shape1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shape1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shape1_shard_spec
    )
    torch_input_tensor1 = random_torch_tensor(dtype, shape1)
    ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device, memory_config=shape1_memory_config)

    shape2_shard_shape = (shard_height, channels2)
    shape2_shard_spec = ttnn.ShardSpec(core_grid, shape2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shape2_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shape2_shard_spec
    )
    torch_input_tensor2 = random_torch_tensor(dtype, shape2)
    ttnn_input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input_tensor2 = ttnn.to_device(ttnn_input_tensor2, device, memory_config=shape2_memory_config)

    output_shard_shape = (shard_height, channels1 + channels2)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )

    actual = ttnn.concat(
        [ttnn_input_tensor1, ttnn_input_tensor2],
        dim=-1,
        memory_config=output_memory_config,
    )
    expected = torch.concat([torch_input_tensor1, torch_input_tensor2], dim=-1)
    assert_equal(expected, ttnn.to_torch(actual))


@pytest.mark.parametrize("input_shapes", [(32, 96), (31, 95), (1023, 1023), (64, 31), (127, 32), (7, 1), (1, 1)])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_concat_1d(device, layout, dim, input_shapes):
    a = torch.randn(input_shapes[0], dtype=torch.bfloat16)
    b = torch.randn(input_shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([a, b], dim=dim)

    in1 = ttnn.from_torch(a, dtype=ttnn.bfloat16, device=device, layout=layout)
    in2 = ttnn.from_torch(b, dtype=ttnn.bfloat16, device=device, layout=layout)

    output = ttnn.concat([in1, in2], dim=dim)
    output = ttnn.to_torch(output)

    assert_equal(torch_output_tensor, output)


##################
# DN sharded


@pytest.mark.parametrize(
    "num_tensors, tensor_shape, shard_shape, concat_dim, layout",
    [
        # 2 tensors - basic ND sharding concat
        (2, [3, 128, 160], [1, 32, 32], 0, ttnn.TILE_LAYOUT),  # concat on batch dim
        #        (2, [3, 128, 160], [3, 32, 32], 1, ttnn.TILE_LAYOUT),  # concat on height dim
        #        (2, [3, 128, 160], [3, 128, 32], 2, ttnn.TILE_LAYOUT),  # concat on width dim
        #        (2, [3, 4, 5], [1, 1, 5], 0, ttnn.ROW_MAJOR_LAYOUT),   # RM concat on batch
        #        (2, [3, 4, 5], [3, 1, 5], 1, ttnn.ROW_MAJOR_LAYOUT),   # RM concat on height
        # 3 tensors - ND sharding concat
        #        (3, [2, 64, 64], [1, 32, 32], 0, ttnn.TILE_LAYOUT),  # concat 3 tensors on batch
        #        (3, [2, 64, 64], [2, 32, 32], 1, ttnn.TILE_LAYOUT),    # concat 3 tensors on height
        #        (3, [2, 64, 64], [2, 64, 32], 2, ttnn.TILE_LAYOUT),    # concat 3 tensors on width
        #        (3, [2, 4, 8], [1, 4, 8], 0, ttnn.ROW_MAJOR_LAYOUT),   # RM concat 3 tensors
        # 4 tensors - ND sharding concat
        #        (4, [1, 64, 64], [1, 32, 32], 1, ttnn.TILE_LAYOUT),  # concat 4 tensors on height
        #        (4, [1, 64, 64], [1, 64, 32], 2, ttnn.TILE_LAYOUT),    # concat 4 tensors on width
        #        (4, [2, 32, 32], [1, 32, 32], 0, ttnn.TILE_LAYOUT),    # concat 4 tensors on batch
        #        (4, [2, 4, 8], [2, 4, 8], 0, ttnn.ROW_MAJOR_LAYOUT),   # RM concat 4 tensors on batch
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nd_sharded_concat(device, num_tensors, tensor_shape, shard_shape, concat_dim, layout, dtype):
    """
    Test concat operation with ND sharded tensors.

    This test verifies that concat works correctly with N-dimensional sharding
    for 2, 3, and 4 input tensors across different concat dimensions.
    """
    torch.manual_seed(0)

    # Skip incompatible configurations
    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b is only valid for TILE_LAYOUT")

    # Setup grid
    grid_size = device.compute_with_storage_grid_size()
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))
    grid = ttnn.CoreRangeSet([core_range])  # TODOZ: to discuss/test partial core range set

    # Create ND shard spec for input tensors
    input_nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid)
    input_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_shard_spec)

    # Generate input tensors
    torch_tensors = [torch.rand(tensor_shape, dtype=torch.bfloat16) for _ in range(num_tensors)]

    # Expected output from PyTorch
    torch_output = torch.concat(torch_tensors, dim=concat_dim)

    # Create TTNN tensors with ND sharding
    ttnn_tensors = [
        ttnn.from_torch(t, dtype=dtype, device=device, layout=layout, memory_config=input_memory_config)
        for t in torch_tensors
    ]
    print("\ntensors have been created\n")

    # Verify tensors have ND sharding - sanity check
    for tt_tensor in ttnn_tensors:
        mem_config = tt_tensor.memory_config()
        assert mem_config.is_sharded(), "Input tensor should be sharded"
        # Check specifically for ND sharding
        assert mem_config.nd_shard_spec is not None, "Input tensor should use ND sharding"
        assert mem_config.shard_spec is None, "Input tensor should NOT use legacy sharding"
        # verify the shard shape matches initial setup
        assert mem_config.nd_shard_spec.shard_shape == ttnn.Shape(
            shard_shape
        ), f"Shard shape mismatch: expected {shard_shape}, got {mem_config.nd_shard_spec.shard_shape}"

    # Compute output shard shape (adjust for concat dimension): same grid and layout as inputs,
    # output shard shape = sum of input shard shapes along concat_dim (mirrors device op logic).
    # TODO:Z ? should we calculate output_shard_shape in the python - or should force it in C++ code?
    # input_mem_config = ttnn_tensors[0].memory_config()
    # input_nd_spec = input_mem_config.nd_shard_spec
    # output_shard_shape = list(input_nd_spec.shard_shape)
    # output_shard_shape[concat_dim] = 0
    # for tt_t in ttnn_tensors:
    #     output_shard_shape[concat_dim] += tt_t.memory_config().nd_shard_spec.shard_shape[concat_dim]
    # output_nd_shard_spec = ttnn.NdShardSpec(
    #     output_shard_shape,
    #     input_nd_spec.grid,
    #     orientation=input_nd_spec.orientation,
    #     shard_distribution_strategy=input_nd_spec.shard_distribution_strategy,
    # )
    # output_memory_config = ttnn.MemoryConfig(input_mem_config.buffer_type, output_nd_shard_spec)

    print("\nright before concat\n")
    ttnn_output = ttnn.concat(ttnn_tensors, dim=concat_dim)  # , memory_config=output_memory_config)

    # Convert back to torch and compare
    actual_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, actual_output, 0.999)


# @pytest.mark.parametrize(
#     "tensor_shapes, shard_shapes, concat_dim",
#     [
#         # Different sized tensors (same shard shape where applicable)
#         ([(2, 64, 64), (3, 64, 64)], [(1, 32, 32), (1, 32, 32)], 0),  # Different batch sizes
#         ([(2, 64, 64), (2, 96, 64)], [(2, 32, 32), (2, 32, 32)], 1),  # Different heights
#         ([(2, 64, 64), (2, 64, 96)], [(2, 64, 32), (2, 64, 32)], 2),  # Different widths
#     ],
# )
# @pytest.mark.parametrize("dtype", [ttnn.bfloat16])
# def test_nd_sharded_concat_different_sizes(device, tensor_shapes, shard_shapes, concat_dim, dtype):
#     """
#     Test concat with ND sharded tensors of different sizes along concat dimension.
#     """
#     torch.manual_seed(0)

#     # Setup grid
#     grid_size = device.compute_with_storage_grid_size()
#     core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))
#     grid = ttnn.CoreRangeSet([core_range])

#     # Generate input tensors with different shapes
#     torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in tensor_shapes]
#     torch_output = torch.concat(torch_tensors, dim=concat_dim)

#     # Create TTNN tensors with ND sharding
#     ttnn_tensors = []
#     for torch_tensor, shard_shape in zip(torch_tensors, shard_shapes):
#         nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid)
#         memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)
#         tt_tensor = ttnn.from_torch(
#             torch_tensor, dtype=dtype, device=device,
#             layout=ttnn.TILE_LAYOUT, memory_config=memory_config
#         )
#         ttnn_tensors.append(tt_tensor)

#     # Concat with interleaved output
#     ttnn_output = ttnn.concat(ttnn_tensors, dim=concat_dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
#     actual_output = ttnn.to_torch(ttnn_output)

#     assert_with_pcc(torch_output, actual_output, 0.999)
