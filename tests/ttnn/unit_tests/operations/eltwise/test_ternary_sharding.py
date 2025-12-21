# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


@pytest.mark.parametrize(
    "shape, shard_strategy, shard_shape_row_major, shard_shape_col_major, core_grid",
    [
        # BLOCK sharding
        (
            (1, 1, 1024, 1024),
            ttnn.ShardStrategy.BLOCK,
            (1024 // 2, 1024 // 4),  # (512, 256)
            (1024 // 2, 1024 // 4),  # (512, 256) - same for COL_MAJOR
            ttnn.CoreGrid(y=2, x=4),
        ),
        # HEIGHT sharding
        (
            (1, 1, 256, 256),
            ttnn.ShardStrategy.HEIGHT,
            (256 // 4, 256),  # (64, 256)
            (256, 256 // 4),  # (256, 64) for COL_MAJOR
            ttnn.CoreGrid(y=4, x=1),
        ),
        # WIDTH sharding
        (
            (1, 1, 256, 256),
            ttnn.ShardStrategy.WIDTH,
            (256, 256 // 4),  # (256, 64)
            (256 // 4, 256),  # (64, 256) for COL_MAJOR
            ttnn.CoreGrid(y=1, x=4),
        ),
    ],
)
@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_no_bcast_with_sharding(
    device_module,
    shape,
    shard_strategy,
    shard_shape_row_major,
    shard_shape_col_major,
    core_grid,
    predicate_sharded,
    true_sharded,
    false_sharded,
    out_sharded,
    shard_orientation,
):
    device = device_module
    torch.manual_seed(0)
    torch_predicate = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.rand(shape, dtype=torch.bfloat16)
    torch_false = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = shard_shape_row_major
    else:
        shard_shape = shard_shape_col_major

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, sharded_mem_config)

    if out_sharded:
        out_mem_config = sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_width_bcast_with_height_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    true_shape = (5, 7, 2 * 32, 1)  # width broadcast [5, 7, 64, 1]
    false_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    output_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (10 * 32, 4 * 32)  # [320, 128]
        true_shard_shape = (10 * 32, 32)  # [320, 32]
    else:
        shard_shape = (4 * 32, 10 * 32)  # [128, 320] for COL_MAJOR
        true_shard_shape = (32, 10 * 32)  # [32, 320] for COL_MAJOR

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, height_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_width_bcast_with_width_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    true_shape = (1, 1, 2 * 32, 1)  # width broadcast [1, 1, 64, 1]
    false_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    output_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (4 * 32, 10 * 32)  # [128, 320]
        true_shard_shape = (2 * 32, 32)  # [64, 32]
    else:
        shard_shape = (10 * 32, 4 * 32)  # [320, 128] for COL_MAJOR
        true_shard_shape = (32, 2 * 32)  # [32, 64] for COL_MAJOR

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=4),  # 4 cores: (0,0) to (0,3)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),  # 1 core: (0,0) to (0,0)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, width_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_height_bcast_with_height_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 4 * 32)  # height broadcast [1, 7, 1, 128]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (4 * 32, 4 * 32)  # [128, 128]
        true_shard_shape = (32, 4 * 32)  # [32, 128]
    else:
        shard_shape = (4 * 32, 4 * 32)  # [128, 128] for COL_MAJOR
        true_shard_shape = (4 * 32, 32)  # [128, 32] for COL_MAJOR

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, height_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_height_bcast_with_width_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    true_shape = (1, 1, 1, 7 * 32)  # height broadcast [1, 1, 1, 224]
    false_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (4 * 32, 32)  # [128, 32]
        true_shard_shape = (32, 32)  # [32, 32]
    else:
        shard_shape = (32, 4 * 32)  # [32, 128] for COL_MAJOR
        true_shard_shape = (32, 32)  # [32, 32] for COL_MAJOR

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, width_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_width_bcast_with_block_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 32 * 2, 1)  # width broadcast [1, 7, 64, 1]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 32)  # [128, 32]
        true_shard_shape = (32 * 2, 32)  # [64, 32]
        block_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
        true_core_range = ttnn.CoreRange((0, 0), (0, 6))  # 1 row, 7 columns
    else:
        shard_shape = (32, 2 * 32 * 2)  # [32, 128] for COL_MAJOR
        true_shard_shape = (32, 32 * 2)  # [32, 64] for COL_MAJOR
        block_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns for COL_MAJOR
        true_core_range = ttnn.CoreRange((0, 0), (6, 0))  # 7 rows, 1 column for COL_MAJOR

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({true_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_height_bcast_with_block_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 4 * 32)  # height broadcast [1, 7, 1, 128]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 32)  # [128, 32]
        true_shard_shape = (32, 32)  # [32, 32]
        block_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
        true_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
    else:
        shard_shape = (32, 2 * 32 * 2)  # [32, 128] for COL_MAJOR
        true_shard_shape = (32, 32)  # [32, 32] for COL_MAJOR
        block_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns
        true_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({true_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_scalar_bcast_with_height_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    false_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 4 * 32)  # [128, 128]
        true_shard_shape = (1 * 32, 32)  # [32, 32] for scalar
    else:
        shard_shape = (4 * 32, 2 * 32 * 2)  # [128, 128] for COL_MAJOR
        true_shard_shape = (32, 1 * 32)  # [32, 32] for COL_MAJOR

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, height_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_scalar_bcast_with_width_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    true_shape = (1, 1, 1, 1)  # scalar broadcast [1, 1, 1, 1]
    false_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 1 * 2 * 32, 32)  # [128, 32]
        true_shard_shape = (1 * 32, 32)  # [32, 32] for scalar
    else:
        shard_shape = (32, 2 * 1 * 2 * 32)  # [32, 128] for COL_MAJOR
        true_shard_shape = (32, 1 * 32)  # [32, 32] for COL_MAJOR

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),  # 1 core: (0,0) to (0,0)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, width_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_scalar_bcast_with_block_sharding(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 32)  # [128, 32]
        true_shard_shape = (32, 32)  # [32, 32] for scalar
        block_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
        true_core_range = ttnn.CoreRange((0, 0), (0, 6))  # 1 row, 7 columns
    else:
        shard_shape = (32, 2 * 32 * 2)  # [32, 128] for COL_MAJOR
        true_shard_shape = (32, 32)  # [32, 32] for COL_MAJOR
        block_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns for COL_MAJOR
        true_core_range = ttnn.CoreRange((0, 0), (6, 0))  # 7 rows, 1 column for COL_MAJOR

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({true_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_width_bcast_with_height_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    true_shape = (5, 7, 2 * 32, 1)  # width broadcast [5, 7, 64, 1]
    false_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    output_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (11 * 32, 4 * 32)  # [352, 128] - uneven height sharding
    true_shard_shape = (11 * 32, 32)  # [352, 32] - uneven height sharding

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, height_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_width_bcast_with_width_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    true_shape = (1, 1, 2 * 32, 1)  # width broadcast [1, 1, 64, 1]
    false_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    output_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (32 * 2 * 2, 11 * 32)  # [128, 352] - uneven width sharding
    true_shard_shape = (32 * 2 * 1, 32)  # [64, 32]

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),  # 1 core
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, width_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_width_bcast_with_block_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 3 * 32)  # [2, 7, 64, 96]
    true_shape = (1, 7, 32 * 2, 1)  # width broadcast [1, 7, 64, 1]
    false_shape = (2, 7, 32 * 2, 3 * 32)  # [2, 7, 64, 96]
    output_shape = (2, 7, 32 * 2, 3 * 32)  # [2, 7, 64, 96]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 2 * 32)  # [192, 64] - uneven block sharding
    true_shard_shape = (32 * 2, 32)  # [64, 32]

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))}),  # 2 rows x 5 cols = 10 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_height_bcast_with_height_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 4 * 32)  # height broadcast [1, 7, 1, 128]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 4 * 32)  # [192, 128] - uneven height sharding
    true_shard_shape = (1 * 32, 4 * 32)  # [32, 128] - uneven height sharding

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),  # 5 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, height_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_height_bcast_with_width_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    # Height broadcast: true broadcasts along height dimension
    predicate_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    true_shape = (1, 1, 1, 7 * 32)  # height broadcast [1, 1, 1, 224]
    false_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    # Uneven sharding: 224 total width, shard width = 64
    # 224 / 64 = 3.5 (uneven division)
    shard_shape = (2 * 1 * 64, 64)  # [128, 64] - uneven width sharding
    true_shard_shape = (1 * 32, 64)  # [32, 64]

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, width_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_height_bcast_with_block_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)

    predicate_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    true_shape = (1, 7, 1, 5 * 32)  # height broadcast [1, 7, 1, 160]
    false_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    output_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 2 * 32)  # [192, 64]
    true_shard_shape = (32 * 2, 2 * 32)  # [64, 64]

    block_core_range = ttnn.CoreRange((0, 0), (2, 4))  # 3 rows x 5 cols = 15 cores

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Use same core grid for broadcast tensor (ternary kernel limitation for height broadcast)
    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),  # Same core range as main tensor
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_scalar_bcast_with_height_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    false_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 4 * 32)  # [192, 128] - uneven height sharding
    true_shard_shape = (2 * 32, 32)  # [64, 32]

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),  # 5 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, height_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_scalar_bcast_with_width_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    true_shape = (1, 1, 1, 1)  # scalar broadcast [1, 1, 1, 1]
    false_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (2 * 1 * 64, 2 * 32)  # [128, 64] - uneven width sharding
    true_shard_shape = (1 * 32, 32)  # [32, 32]

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),  # 1 core
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, width_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_where_ttt_scalar_bcast_with_block_sharding_uneven(
    device_module, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    true_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    false_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    output_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 2 * 32)  # [192, 64] - uneven block sharding
    true_shard_shape = (32, 32)  # [32, 32]

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 4))}),  # 15 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)

    predicate_tensor = ttnn.from_torch(
        torch_predicate, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_true, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_false, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_where_ttt_sharded_shardspec_mixed_buffer_type(dtype_pt, dtype_tt, device_module):
    device = device_module
    torch.manual_seed(0)
    dram_grid_size = device.dram_grid_size()
    input_shape = (1, 1, dram_grid_size.x * dram_grid_size.y * 32, 32)

    predicate_pt = torch.randint(0, 2, input_shape, dtype=torch.bfloat16)
    true_pt = torch.rand(input_shape, dtype=torch.bfloat16)
    false_pt = torch.rand(input_shape, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 3))})
    N, C, H, W = input_shape
    n_cores = 12
    import math

    # L1 sharded config for predicate
    shard_spec = ttnn.ShardSpec(
        shard_grid, [math.ceil((N * C * H) / n_cores / 32) * 32, W], ttnn.ShardOrientation.ROW_MAJOR
    )
    predicate_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # DRAM sharded config for true tensor
    from models.common.utility_functions import divup

    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(N * C * H, (dram_grid_size.x * dram_grid_size.y)),
            W,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    true_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

    # False tensor uses DRAM interleaved
    false_config = ttnn.DRAM_MEMORY_CONFIG

    predicate_tt = ttnn.from_torch(
        predicate_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=predicate_config,
    )
    true_tt = ttnn.from_torch(
        true_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=true_config,
    )
    false_tt = ttnn.from_torch(
        false_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=false_config,
    )

    out_pt = torch.where(predicate_pt.bool(), true_pt, false_pt)
    out_tt = ttnn.where(predicate_tt, true_tt, false_tt)
    assert torch_equal_nan(ttnn.to_torch(out_tt), out_pt)


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize(
    "config_permutation",
    [
        "hwb",  # predicate=HEIGHT, true=WIDTH, false=BLOCK
        "bwh",  # predicate=BLOCK, true=WIDTH, false=HEIGHT
        "whb",  # predicate=WIDTH, true=HEIGHT, false=BLOCK
        "bhw",  # predicate=BLOCK, true=HEIGHT, false=WIDTH
        "wbh",  # predicate=WIDTH, true=BLOCK, false=HEIGHT
        "hbw",  # predicate=HEIGHT, true=BLOCK, false=WIDTH
    ],
)
@pytest.mark.parametrize(
    "out_strategy",
    ["dram", "height", "width", "block"],
)
@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
def test_where_ttt_identical_mixed_strategy(
    device_module, dtype_pt, dtype_tt, config_permutation, out_strategy, predicate_sharded, true_sharded, false_sharded
):
    device = device_module
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    # Create base sharded memory configs with different strategies
    height_sharded_config = ttnn.create_sharded_memory_config(
        shape=(2 * 32 * 2, 4 * 32),  # [128, 128]
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    width_sharded_config = ttnn.create_sharded_memory_config(
        shape=(2 * 7 * 32 * 2, 32),  # [896, 32]
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    block_sharded_config = ttnn.create_sharded_memory_config(
        shape=(2 * 32 * 2, 32),  # [128, 32]
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),  # 4 rows, 7 columns = 28 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Map permutation string to strategies: h=HEIGHT, w=WIDTH, b=BLOCK
    strategy_map = {"h": height_sharded_config, "w": width_sharded_config, "b": block_sharded_config}

    predicate_strategy = config_permutation[0]
    true_strategy = config_permutation[1]
    false_strategy = config_permutation[2]

    predicate_sharded_config = strategy_map[predicate_strategy]
    true_sharded_config = strategy_map[true_strategy]
    false_sharded_config = strategy_map[false_strategy]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    predicate_tensor = ttnn.from_torch(
        torch_predicate,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, predicate_sharded_config)

    true_tensor = ttnn.from_torch(
        torch_true,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, true_sharded_config)

    false_tensor = ttnn.from_torch(
        torch_false,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, false_sharded_config)

    # Select output memory config based on out_strategy
    if out_strategy == "dram":
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG
    elif out_strategy == "height":
        out_mem_config = height_sharded_config
    elif out_strategy == "width":
        out_mem_config = width_sharded_config
    elif out_strategy == "block":
        out_mem_config = block_sharded_config

    torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)
    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape

    # Test without explicit output memory config
    output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_equal_nan(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
def test_where_ttt_height_bcast_mixed_strategy_mixed_L1(device_module, dtype_pt, dtype_tt):
    device = device_module
    # Clear program cache to ensure correct handling of broadcast shapes
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    predicate_shape = torch.Size([2, 7, 32 * 2, 4 * 32])  # [2, 7, 64, 128]
    true_shape = torch.Size([1, 7, 1, 4 * 32])  # [1, 7, 1, 128] - height broadcast
    false_shape = torch.Size([1, 7, 1, 4 * 32])  # [1, 7, 1, 128] - height broadcast

    predicate_sharded_config = ttnn.create_sharded_memory_config(
        [2 * 32 * 2, 4 * 32],  # [128, 128]
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_config = ttnn.create_sharded_memory_config(
        [7 * 32, 32],  # [224, 32]
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    false_sharded_config = ttnn.create_sharded_memory_config(
        [7 * 32, 32],  # [224, 32]
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    import itertools

    input_combinations = itertools.product(
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, predicate_sharded_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, true_sharded_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, false_sharded_config],
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, predicate_sharded_config],
    )

    for pred_config, true_config, false_config, out_config in input_combinations:
        torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
        torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
        torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

        predicate_tensor = ttnn.from_torch(
            torch_predicate,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=pred_config,
        )

        true_tensor = ttnn.from_torch(
            torch_true,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=true_config,
        )

        false_tensor = ttnn.from_torch(
            torch_false,
            dtype=dtype_tt,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=false_config,
        )

        torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)
        output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor, memory_config=out_config)
        output_tensor = ttnn.to_torch(output_tensor)
        assert torch_equal_nan(output_tensor, torch_output_tensor)
        assert output_tensor.shape == predicate_shape

        # Test without explicit output memory config
        torch_output_tensor = torch.where(torch_predicate.bool(), torch_true, torch_false)
        output_tensor = ttnn.where(predicate_tensor, true_tensor, false_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        assert torch_equal_nan(output_tensor, torch_output_tensor)
        assert output_tensor.shape == predicate_shape
