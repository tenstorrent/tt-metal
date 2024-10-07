# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


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

    if ttnn.has_tile_padding(input_tensor_a, dim=dim) or ttnn.has_tile_padding(input_tensor_b, dim=dim):
        pytest.skip("Cannot concat tensors with tile padding")

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
            marks=pytest.mark.xfail(reason="two tensors concat kernel doesn't work with program cache (#13466)"),
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
