# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp
from models.common.utility_functions import is_wormhole_b0


def is_simulator():
    return os.environ.get("TT_METAL_SIMULATOR") is not None


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [128, 160],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6)), ttnn.CoreRange((1, 0), (1, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [1792, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3)), ttnn.CoreRange((1, 0), (1, 3))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [256, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([4, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "input_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "out_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op, torch_dtype, ttnn_dtype",
    [
        (ttnn.relu, torch.bfloat16, ttnn.bfloat16),
        (ttnn.square, torch.float32, ttnn.float32),
        (ttnn.abs, torch.int32, ttnn.int32),
    ],
)
def test_unary_sharded_interleaved(input_shape, input_config, out_config, ttnn_op, torch_dtype, ttnn_dtype, device):
    """Test unary_ng ops with all combinations of sharded/interleaved input and output."""
    torch.manual_seed(0)
    if torch_dtype.is_floating_point:
        torch_input = torch.empty(input_shape, dtype=torch_dtype).uniform_(-100, 100)
    else:
        torch_input = torch.randint(-100, 100, input_shape, dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_config,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input, memory_config=out_config))
    assert torch.equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 64, 128]),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.sinh])
def test_unary_ng_row_major(input_shape, ttnn_op, device):
    """Test unary_ng ops with ROW_MAJOR layout on interleaved tensors."""
    torch.manual_seed(0)
    torch_input = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-10, 10)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(torch_input, device=device)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input))
    assert_with_ulp(ttnn_output, golden_tensor, ulp_threshold=5.0)


@pytest.mark.parametrize(
    "shape, sub_core_grid",
    [
        (
            torch.Size([1, 2, 32, 960]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
        (
            torch.Size([1, 7, 32, 96]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 6)),
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.cbrt])
def test_unary_ng_sub_core_grids(shape, sub_core_grid, ttnn_op, device):
    """Test unary_ng ops with sub_core_grids on interleaved tensors."""
    torch.manual_seed(0)
    torch_input = torch.empty(shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = ttnn.to_torch(ttnn_op(ttnn_input, sub_core_grids=sub_core_grid))
    assert_with_ulp(ttnn_output, golden_tensor, ulp_threshold=1.0)


@pytest.mark.parametrize("ttnn_op", [ttnn.square])
def test_unary_ng_uneven_sharding_fallback(ttnn_op, device):
    """Test that uneven sharding falls back to interleaved path gracefully.

    When the tensor dimensions don't divide evenly into the shard shape, is_uneven()
    returns true and unary_ng falls back to the TensorAccessor (interleaved) path
    instead of using the native sharded path.
    """
    torch.manual_seed(42)
    input_shape = torch.Size([1, 1, 160, 96])
    torch_input = torch.randint(0, 255, input_shape, dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden_tensor = golden_function(torch_input, device=device)

    uneven_shard_config = ttnn.create_sharded_memory_config(
        [64, 96],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 2))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=uneven_shard_config,
    )

    ttnn_output = ttnn_op(ttnn_input, memory_config=uneven_shard_config)
    ttnn_output = ttnn.typecast(ttnn_output, dtype=ttnn.uint32)
    ttnn_output = ttnn.to_torch(ttnn_output, dtype=torch.int32)
    assert torch.equal(ttnn_output, golden_tensor)


@pytest.mark.parametrize(
    "input_shape, shard_shape, core_grid, strategy",
    [
        (
            [2, 16, 16, 640],
            [16, 640],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 3))}),
            ttnn.ShardStrategy.HEIGHT,
        ),
        (
            [1, 1, 64, 512],
            [64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0))}),
            ttnn.ShardStrategy.WIDTH,
        ),
        (
            [1, 1, 256, 512],
            [64, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 3))}),
            ttnn.ShardStrategy.BLOCK,
        ),
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        pytest.param(
            torch.float32,
            ttnn.float32,
            marks=pytest.mark.skipif(
                is_simulator() and is_wormhole_b0(), reason="tt-sim + Wormhole float32 failure #39185"
            ),
        ),
    ],
)
def test_unary_ng_row_major_sharded(input_shape, shard_shape, core_grid, strategy, device, torch_dtype, ttnn_dtype):
    """Test unary_ng abs with ROW_MAJOR layout and sharded memory config."""
    torch.manual_seed(42)
    torch_input = torch.empty(input_shape, dtype=torch_dtype).uniform_(-100, 100)

    golden_function = ttnn.get_golden_function(ttnn.abs)
    torch_output = golden_function(torch_input, device=device)

    shard_mem_config = ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_input = ttnn.to_memory_config(ttnn_input, memory_config=shard_mem_config)

    ttnn_output = ttnn.to_torch(ttnn.abs(ttnn_input, memory_config=shard_mem_config))
    assert torch.equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "strategy, shard_shape_rm, shard_shape_cm, core_grid",
    [
        (
            ttnn.ShardStrategy.HEIGHT,
            [256 // 4, 256],
            [256, 256 // 4],
            ttnn.CoreGrid(y=4, x=1),
        ),
        (
            ttnn.ShardStrategy.WIDTH,
            [256, 256 // 4],
            [256 // 4, 256],
            ttnn.CoreGrid(y=1, x=4),
        ),
        (
            ttnn.ShardStrategy.BLOCK,
            [256 // 2, 256 // 4],
            [256 // 2, 256 // 4],
            ttnn.CoreGrid(y=2, x=4),
        ),
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_unary_ng_shard_orientation(strategy, shard_shape_rm, shard_shape_cm, core_grid, shard_orientation, device):
    """Test unary_ng with both ROW_MAJOR and COL_MAJOR shard orientations."""
    torch.manual_seed(0)
    input_shape = [1, 1, 256, 256]
    torch_input = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_function = ttnn.get_golden_function(ttnn.abs)
    torch_output = golden_function(torch_input, device=device)

    shard_shape = shard_shape_rm if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR else shard_shape_cm

    shard_mem_config = ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=shard_mem_config,
    )

    ttnn_output = ttnn.to_torch(ttnn.abs(ttnn_input, memory_config=shard_mem_config))
    assert torch.equal(ttnn_output, torch_output)


@pytest.mark.parametrize(
    "input_shape, input_shard_shape, input_core_grid, input_strategy, output_memory_config",
    [
        (
            [1, 1, 256, 128],
            [64, 128],
            ttnn.CoreGrid(y=4, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ),
        (
            [1, 1, 128, 256],
            [128, 64],
            ttnn.CoreGrid(y=1, x=4),
            ttnn.ShardStrategy.WIDTH,
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ),
        (
            [1, 1, 256, 256],
            [128, 64],
            ttnn.CoreGrid(y=2, x=4),
            ttnn.ShardStrategy.BLOCK,
            ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ),
    ],
)
def test_unary_ng_generic_sharded_memory_config(
    input_shape, input_shard_shape, input_core_grid, input_strategy, output_memory_config, device
):
    """Test unary_ng with generic sharded memory configs (no explicit shard spec).

    When using L1_HEIGHT_SHARDED_MEMORY_CONFIG etc., the shard spec is inferred
    from the input tensor. The output strategy must match the input strategy.
    """
    torch.manual_seed(0)
    torch_input = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_function = ttnn.get_golden_function(ttnn.neg)
    torch_output = golden_function(torch_input, device=device)

    input_shard_config = ttnn.create_sharded_memory_config(
        shape=input_shard_shape,
        core_grid=input_core_grid,
        strategy=input_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_shard_config,
    )

    ttnn_output = ttnn.to_torch(ttnn.neg(ttnn_input, memory_config=output_memory_config))
    assert torch.equal(ttnn_output, torch_output)


def test_unary_ng_reshard(device):
    """Test unary_ng where input is sharded on one grid and output on a different grid."""
    torch.manual_seed(0)
    input_shape = [1, 1, 64, 512]
    torch_input = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-100, 100)

    golden_function = ttnn.get_golden_function(ttnn.relu)
    torch_output = golden_function(torch_input, device=device)

    input_shard_config = ttnn.create_sharded_memory_config(
        shape=[64, 64],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (7, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    output_shard_config = ttnn.create_sharded_memory_config(
        shape=[64, 128],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_shard_config,
    )

    ttnn_output = ttnn.to_torch(ttnn.relu(ttnn_input, memory_config=output_shard_config))
    assert torch.equal(ttnn_output, torch_output)
