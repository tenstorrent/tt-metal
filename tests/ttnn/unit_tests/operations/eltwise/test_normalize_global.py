# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose

# H and W of 4x4, 17x50, 7x19, 30x40 and 56x56 are not multiples of 32. On those the op used to count the tile padding
# as data and return the padded shape.
SHAPES = [
    [1, 1, 32, 32],
    [2, 3, 64, 64],
    [1, 4, 128, 96],
    [1, 1, 4, 4],
    [1, 1, 17, 50],
    [2, 3, 7, 19],
    [1, 2, 30, 40],
    [1, 8, 56, 56],
]
TOLERANCES = {ttnn.bfloat16: 2e-2, ttnn.bfloat8_b: 6e-2, ttnn.float32: 1e-5}


def torch_normalize_global(x):
    xd = x.double()
    return (xd - xd.mean()) / xd.std(correction=0)


def _torch_dtype(dtype):
    return torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
def test_normalize_global(device, shape, dtype, layout):
    torch.manual_seed(0)
    x = (torch.randn(shape) * 3 + 1).to(_torch_dtype(dtype))

    output = ttnn.normalize_global(ttnn.from_torch(x, dtype=dtype, layout=layout, device=device))

    assert list(output.shape) == shape
    tolerance = TOLERANCES[dtype]
    assert_allclose(
        expected_result=torch_normalize_global(x).float(),
        actual_result=ttnn.to_torch(output).float(),
        rtol=tolerance,
        atol=tolerance,
    )


# Half the elements are 2 and half are 0, so the mean and the std are both 1 and every output is +1 or -1. With the
# padding counted, 4x4 gave +11.27 and -0.09.
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "shape", [[1, 1, 4, 4], [1, 1, 17, 50], [2, 3, 7, 19]], ids=lambda shape: "x".join(map(str, shape))
)
def test_normalize_global_two_values(device, shape, dtype):
    x = torch.zeros(shape, dtype=_torch_dtype(dtype))
    x.view(-1)[::2] = 2.0

    output = ttnn.normalize_global(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))

    assert list(output.shape) == shape
    torch.testing.assert_close(ttnn.to_torch(output).float(), x.float() - 1, rtol=0, atol=TOLERANCES[dtype])


# bfloat8_b runs as bfloat16 and comes back as bfloat8_b. The golden takes the values the device holds.
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
def test_normalize_global_bfloat8_b(device, shape):
    torch.manual_seed(0)
    input_tensor = ttnn.from_torch(
        torch.randn(shape) * 3 + 1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )

    output = ttnn.normalize_global(input_tensor)

    assert list(output.shape) == shape
    assert output.dtype == ttnn.bfloat8_b
    tolerance = TOLERANCES[ttnn.bfloat8_b]
    assert_allclose(
        expected_result=torch_normalize_global(ttnn.to_torch(input_tensor)).float(),
        actual_result=ttnn.to_torch(output).float(),
        rtol=tolerance,
        atol=tolerance,
    )


# A mean far from 0 next to a small std. The mean is now taken in FP32: rounded to bfloat16, a mean of 100.3 next to a
# std of 0.5 put outputs about 1.0 off, and bfloat8_b came back as inf on shapes with tile padding.
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32], ids=["bf16", "bf8", "fp32"])
@pytest.mark.parametrize(
    "shape",
    [[1, 1, 32, 32], [1, 1, 4, 4], [1, 2, 32, 32], [2, 3, 64, 64], [1, 2, 30, 40], [2, 2, 7, 7]],
    ids=lambda shape: "x".join(map(str, shape)),
)
def test_normalize_global_large_mean(device, shape, dtype):
    torch.manual_seed(0)
    input_tensor = ttnn.from_torch(
        torch.randn(shape) * 0.5 + 100.3, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    output = ttnn.normalize_global(input_tensor)

    assert list(output.shape) == shape
    # A FLOAT32 ULP at 100 is 7.6e-6, already 1.5e-5 of the std.
    tolerance = 1e-4 if dtype == ttnn.float32 else TOLERANCES[dtype]
    assert_allclose(
        expected_result=torch_normalize_global(ttnn.to_torch(input_tensor)).float(),
        actual_result=ttnn.to_torch(output).float(),
        rtol=tolerance,
        atol=tolerance,
    )


def _memory_config(shape, placement):
    if placement == "height_sharded":
        return ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.HEIGHT)
    if placement == "block_sharded":
        return ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK)
    return {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[placement]


# Without memory_config the output stays where the input is; with one it goes there. Height-sharded inputs used to fail.
@pytest.mark.parametrize(
    "shape, input_placement",
    [
        ([1, 2, 30, 40], "l1"),
        pytest.param([1, 8, 64, 64], "height_sharded", marks=pytest.mark.requires_grid_size((8, 1))),
        pytest.param([1, 1, 256, 256], "block_sharded", marks=pytest.mark.requires_grid_size((2, 2))),
    ],
    ids=["l1", "height_sharded", "block_sharded"],
)
@pytest.mark.parametrize("output_placement", [None, "dram", "l1"], ids=["none", "dram", "l1"])
def test_normalize_global_memory_config(device, shape, input_placement, output_placement):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_memory_config(shape, input_placement),
    )
    kwargs = {} if output_placement is None else {"memory_config": _memory_config(shape, output_placement)}

    output = ttnn.normalize_global(input_tensor, **kwargs)

    expected_memory_config = input_tensor.memory_config() if output_placement is None else kwargs["memory_config"]
    assert output.memory_config().memory_layout == expected_memory_config.memory_layout
    assert output.memory_config().buffer_type == expected_memory_config.buffer_type
    assert_allclose(
        expected_result=torch_normalize_global(x).float(),
        actual_result=ttnn.to_torch(output).float(),
        rtol=1e-5,
        atol=1e-5,
    )


# The op frees the copies it makes once used. A TILE input is only viewed, never copied, so freeing must not reach it:
# every input reads back unchanged, and a second call gives the same output.
@pytest.mark.parametrize(
    "shape, dtype, layout, placement",
    [
        ([2, 3, 64, 64], ttnn.bfloat16, ttnn.TILE_LAYOUT, "dram"),
        ([1, 2, 30, 40], ttnn.float32, ttnn.TILE_LAYOUT, "l1"),
        pytest.param(
            [1, 8, 64, 64],
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            "height_sharded",
            marks=pytest.mark.requires_grid_size((8, 1)),
        ),
        ([1, 2, 30, 40], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "dram"),
        ([1, 2, 30, 40], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "l1"),
    ],
    ids=["bf16_tile_dram", "fp32_tile_l1", "fp32_tile_height_sharded", "bf8_tile_dram", "bf16_row_major_l1"],
)
def test_normalize_global_keeps_input(device, shape, dtype, layout, placement):
    torch.manual_seed(0)
    input_tensor = ttnn.from_torch(
        torch.randn(shape) * 3 + 1,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=_memory_config(shape, placement),
    )
    before = ttnn.to_torch(input_tensor)

    first = ttnn.to_torch(ttnn.normalize_global(input_tensor))

    assert input_tensor.is_allocated()
    assert torch.equal(ttnn.to_torch(input_tensor), before)
    assert torch.equal(ttnn.to_torch(ttnn.normalize_global(input_tensor)), first)


# A ROW_MAJOR shard that is not whole tiles cannot be tilized in place, so that input is tilized in interleaved memory
# of its buffer type, which is then where the output goes by default. Such inputs used to fail.
@pytest.mark.requires_grid_size((4, 2))
@pytest.mark.parametrize(
    "shape, strategy, grid, whole_tiles",
    [
        ([1, 2, 30, 40], ttnn.ShardStrategy.HEIGHT, (1, 2), False),
        ([1, 3, 20, 64], ttnn.ShardStrategy.HEIGHT, (1, 2), False),
        ([1, 1, 60, 64], ttnn.ShardStrategy.BLOCK, (2, 2), False),
        ([1, 2, 32, 64], ttnn.ShardStrategy.HEIGHT, (1, 2), True),
        ([1, 1, 32, 128], ttnn.ShardStrategy.WIDTH, (1, 4), True),
    ],
    ids=["height_30_rows", "height_30_rows_3_planes", "block_30_rows", "height_32_rows", "width_32_columns"],
)
@pytest.mark.parametrize("output_placement", [None, "dram"], ids=["none", "dram"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_normalize_global_row_major_sharded(device, shape, strategy, grid, whole_tiles, output_placement, dtype):
    torch.manual_seed(0)
    x = (torch.randn(shape) * 3 + 1).to(_torch_dtype(dtype))
    memory_config = ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=grid[0], x=grid[1]), strategy)
    input_tensor = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    kwargs = {} if output_placement is None else {"memory_config": ttnn.DRAM_MEMORY_CONFIG}

    output = ttnn.normalize_global(input_tensor, **kwargs)

    assert list(output.shape) == shape
    if output_placement is None:
        expected_layout = memory_config.memory_layout if whole_tiles else ttnn.TensorMemoryLayout.INTERLEAVED
        assert output.memory_config().memory_layout == expected_layout
        assert output.memory_config().buffer_type == ttnn.BufferType.L1
    else:
        assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        assert output.memory_config().buffer_type == ttnn.BufferType.DRAM
    tolerance = TOLERANCES[dtype]
    assert_allclose(
        expected_result=torch_normalize_global(x).float(),
        actual_result=ttnn.to_torch(output).float(),
        rtol=tolerance,
        atol=tolerance,
    )


def _nd_sharded_memory_config(shard_shape):
    return ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            shard_shape=ttnn.Shape(shard_shape),
            grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))}),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


# The output has the ND shard spec that was asked for, or the input's without memory_config. It used to come back with
# one derived for [1, N * C, H, W], or run out of L1 when the shard spans N.
@pytest.mark.requires_grid_size((4, 2))
@pytest.mark.parametrize(
    "shape, input_shard_shape, output_shard_shape",
    [
        ([2, 3, 64, 64], None, [1, 3, 32, 64]),
        ([2, 3, 30, 40], None, [1, 3, 32, 64]),
        ([8, 1, 256, 256], None, [8, 1, 32, 32]),
        ([2, 3, 64, 64], [1, 3, 32, 64], [2, 1, 64, 64]),
        ([2, 3, 64, 64], [1, 3, 32, 64], None),
    ],
    ids=["aligned", "unaligned", "shard_spans_n", "nd_sharded_input", "nd_sharded_input_no_memory_config"],
)
def test_normalize_global_nd_sharded_output(device, shape, input_shard_shape, output_shard_shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    input_memory_config = (
        ttnn.DRAM_MEMORY_CONFIG if input_shard_shape is None else _nd_sharded_memory_config(input_shard_shape)
    )
    input_tensor = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    kwargs = {} if output_shard_shape is None else {"memory_config": _nd_sharded_memory_config(output_shard_shape)}

    output = ttnn.normalize_global(input_tensor, **kwargs)

    assert list(output.shape) == shape
    expected_memory_config = kwargs.get("memory_config", input_memory_config)
    assert output.memory_config().nd_shard_spec == expected_memory_config.nd_shard_spec
    assert_allclose(
        expected_result=torch_normalize_global(x).float(),
        actual_result=ttnn.to_torch(output).float(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_normalize_global_requires_rank_4(device, expect_error):
    input_tensor = ttnn.from_torch(
        torch.randn([2, 64, 64]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "normalize_global requires a rank 4 tensor, got rank 3"):
        ttnn.normalize_global(input_tensor)
