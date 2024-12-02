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
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_tiled_concat(device, concat_spec, async_mode):
    shapes, dim = concat_spec
    device.enable_async(async_mode)
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
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_concat(device, height, width, dim, async_mode):
    device.enable_async(async_mode)
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
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_sharded_concat(device, inputs, output_shard_shape, shard_grid, strategy, layout, cache_mode, async_mode):
    device.enable_async(async_mode)
    if cache_mode:
        device.enable_program_cache()

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


def grouped_concat(activations, residuals, groups):
    """
    Concatenate activations and residuals with flexible interleaving based on groups.

    Args:
        activations (torch.Tensor): Activation tensor with shape [N, H, W, C].
        residuals (torch.Tensor): Residual tensor with shape [N, H, W, C].
        groups (int): Number of groups to split channels into.

    Returns:
        torch.Tensor: Concatenated tensor with interleaved groups.
    """
    assert (
        activations.shape[:-1] == residuals.shape[:-1]
    ), "Activations and residuals must have the same shape in all dims but -1"

    N, H, W, activation_channels = activations.shape
    assert activation_channels % groups == 0, "Channel count must be divisible by the number of groups"

    N, H, W, residual_channels = residuals.shape
    assert residual_channels % groups == 0, "Channel count must be divisible by the number of groups"

    act_groups = activations.view(N, H, W, groups, activation_channels // groups)
    res_groups = residuals.view(N, H, W, groups, residual_channels // groups)

    # Interleave activations and residuals along the channel axis
    interleaved = torch.cat([act_groups, res_groups], dim=-1)  # Shape: [N, H, W, groups, 2 * group_size]

    # Reshape to combine groups and channels correctly
    interleaved = interleaved.permute(0, 1, 2, 3, 4).reshape(N, H, W, residual_channels + activation_channels)

    return interleaved


@pytest.mark.parametrize("dim", [3])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize(
    "input_shapes, output_shape, core_grid",
    (
        (
            ((1, 1, 1, 32), (1, 1, 1, 32)),
            (1, 1, 1, 64),
            ttnn.CoreGrid(x=1, y=1),
        ),
        (
            ((1, 1, 1, 32), (1, 1, 1, 64)),
            (1, 1, 1, 96),
            ttnn.CoreGrid(x=1, y=1),
        ),
        (
            ((1, 1, 1, 64), (1, 1, 1, 32)),
            (1, 1, 1, 96),
            ttnn.CoreGrid(x=1, y=1),
        ),
        (
            ((1, 1, 1024, 64), (1, 1, 1024, 32)),
            (1, 1, 1024, 96),
            ttnn.CoreGrid(x=4, y=1),
        ),
        (
            ((1, 1, 256, 64), (1, 1, 256, 128)),
            (1, 1, 256, 192),
            ttnn.CoreGrid(x=8, y=1),
        ),
    ),
)
def test_sharded_concat_with_groups(device, input_shapes, output_shape, dim, groups, core_grid):
    torch_input_tensors = [torch.full(shapes, idx + 1, dtype=torch.bfloat16) for idx, shapes in enumerate(input_shapes)]

    expected = grouped_concat(torch_input_tensors[0], torch_input_tensors[1], groups)

    sharded_memory_configs = [
        ttnn.create_sharded_memory_config(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
        )
        for x in torch_input_tensors
    ]
    ttnn_input_tensors = [
        ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
        )
        for x, memory_config in zip(torch_input_tensors, sharded_memory_configs)
    ]

    output_memory_config = ttnn.create_sharded_memory_config(output_shape, core_grid, ttnn.ShardStrategy.HEIGHT)
    z = ttnn.concat(
        [ttnn_input_tensors[0], ttnn_input_tensors[1]], dim=dim, memory_config=output_memory_config, groups=groups
    )

    actual = ttnn.to_torch(z)
    assert_with_pcc(expected, actual, 1.0)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_concat_5d(device, dim):
    torch_input_tensor = torch.rand(1, 1, 1, 1, 2, dtype=torch.bfloat16)
    torch_result = torch.cat([torch_input_tensor, torch_input_tensor], dim=dim)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_result = ttnn.concat([ttnn_input_tensor, ttnn_input_tensor], dim=dim)
    ttnn_result = ttnn.to_torch(ttnn_result)
    assert_with_pcc(torch_result, ttnn_result, 0.9999)
