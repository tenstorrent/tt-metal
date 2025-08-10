# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "concat_spec",
    (([[1, 1, 12, 50], [1, 1, 12, 50]], -1),),
)
def test_tiled_concat(device, concat_spec):
    shapes, dim = concat_spec
    torch_input_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in shapes]
    torch_output_tensor = torch.concat(torch_input_tensors, dim=dim)

    input_tensors = [
        ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        for torch_input_tensor in torch_input_tensors
    ]

    output = ttnn.concat(input_tensors, dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize("height", [20, 32])
@pytest.mark.parametrize("width", [4, 32])
@pytest.mark.parametrize("dim", [0, 1])
def test_concat(device, height, width, dim):
    torch_input_tensor_a = torch.rand((height, width), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


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
def test_sharded_concat(device, inputs, output_shard_shape, shard_grid, strategy, layout, cache_mode):
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
            torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
            input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
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
    assert_with_pcc(torch_output_tensor, output)

    # If cache mode is enabled, run the second set of input tensors to verify buffers are updated when cache hits.
    if not cache_mode:
        return

    input_tensors_2 = _gen_inputs(inputs)
    torch_output_tensor_2 = torch.concat([torch_tensor for torch_tensor, _ in input_tensors_2], dim=dim)
    output_2 = ttnn.concat(
        [tensor for _, tensor in input_tensors_2], dim=dim, memory_config=output_sharded_memory_config
    )
    output_2 = ttnn.to_torch(output_2)
    assert_with_pcc(torch_output_tensor_2, output_2)


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
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
    torch_input_tensors = [torch.rand(shapes, dtype=torch.bfloat16) for idx, shapes in enumerate(input_shapes)]

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
    assert_with_pcc(expected, actual, 1.0 if dtype == ttnn.bfloat16 else 0.99995)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_concat_5d(device, dim):
    torch_input_tensor = torch.rand(1, 1, 1, 1, 2, dtype=torch.bfloat16)
    torch_result = torch.cat([torch_input_tensor, torch_input_tensor], dim=dim)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_result = ttnn.concat([ttnn_input_tensor, ttnn_input_tensor], dim=dim)
    ttnn_result = ttnn.to_torch(ttnn_result)
    assert_with_pcc(torch_result, ttnn_result, 0.9999)


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
def test_concat_sharded_pad(device, core_grid, hw, channels1, channels2, shard_height):
    shape1 = [1, 1, hw, channels1]
    shape2 = [1, 1, hw, channels2]

    shape1_shard_shape = (shard_height, channels1)
    shape1_shard_spec = ttnn.ShardSpec(core_grid, shape1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shape1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shape1_shard_spec
    )
    torch_input_tensor1 = torch.randn(shape1, dtype=torch.bfloat16)
    ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device, memory_config=shape1_memory_config)

    shape2_shard_shape = (shard_height, channels2)
    shape2_shard_spec = ttnn.ShardSpec(core_grid, shape2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shape2_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shape2_shard_spec
    )
    torch_input_tensor2 = torch.randn(shape2, dtype=torch.bfloat16)
    ttnn_input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
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
    assert_with_pcc(expected, ttnn.to_torch(actual), 0.9999)
