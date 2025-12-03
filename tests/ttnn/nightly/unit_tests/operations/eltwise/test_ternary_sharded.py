# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


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
@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_no_bcast_with_sharding(
    device,
    shape,
    shard_strategy,
    shard_shape_row_major,
    shard_shape_col_major,
    core_grid,
    input1_sharded,
    input2_sharded,
    input3_sharded,
    out_sharded,
    shard_orientation,
    ttnn_op,
):
    torch.manual_seed(0)
    torch_input1 = torch.randn(shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(shape, dtype=torch.bfloat16)

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

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, sharded_mem_config)

    if out_sharded:
        out_mem_config = sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert assert_with_pcc(output_tensor, torch_output_tensor)


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_width_bcast_with_height_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    input2_shape = (5, 7, 2 * 32, 1)  # width broadcast [5, 7, 64, 1]
    input3_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    output_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (10 * 32, 4 * 32)  # [320, 128]
        input2_shard_shape = (10 * 32, 32)  # [320, 32]
    else:
        shard_shape = (4 * 32, 10 * 32)  # [128, 320] for COL_MAJOR
        input2_shard_shape = (32, 10 * 32)  # [32, 320] for COL_MAJOR

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, height_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_width_bcast_with_width_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    input2_shape = (1, 1, 2 * 32, 1)  # width broadcast [1, 1, 64, 1]
    input3_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    output_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (4 * 32, 10 * 32)  # [128, 320]
        input2_shard_shape = (2 * 32, 32)  # [64, 32]
    else:
        shard_shape = (10 * 32, 4 * 32)  # [320, 128] for COL_MAJOR
        input2_shard_shape = (32, 2 * 32)  # [32, 64] for COL_MAJOR

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=4),  # 4 cores: (0,0) to (0,3)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),  # 1 core: (0,0) to (0,0)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, width_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_height_bcast_with_height_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 1, 4 * 32)  # height broadcast [1, 7, 1, 128]
    input3_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (4 * 32, 4 * 32)  # [128, 128]
        input2_shard_shape = (32, 4 * 32)  # [32, 128]
    else:
        shard_shape = (4 * 32, 4 * 32)  # [128, 128] for COL_MAJOR
        input2_shard_shape = (4 * 32, 32)  # [128, 32] for COL_MAJOR

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, height_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_height_bcast_with_width_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    input2_shape = (1, 1, 1, 7 * 32)  # height broadcast [1, 1, 1, 224]
    input3_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (4 * 32, 32)  # [128, 32]
        input2_shard_shape = (32, 32)  # [32, 32]
    else:
        shard_shape = (32, 4 * 32)  # [32, 128] for COL_MAJOR
        input2_shard_shape = (32, 32)  # [32, 32] for COL_MAJOR

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, width_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_width_bcast_with_block_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 32 * 2, 1)  # width broadcast [1, 7, 64, 1]
    input3_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 32)  # [128, 32]
        input2_shard_shape = (32 * 2, 32)  # [64, 32]
        block_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
        input2_core_range = ttnn.CoreRange((0, 0), (0, 6))  # 1 row, 7 columns
    else:
        shard_shape = (32, 2 * 32 * 2)  # [32, 128] for COL_MAJOR
        input2_shard_shape = (32, 32 * 2)  # [32, 64] for COL_MAJOR
        block_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns for COL_MAJOR
        input2_core_range = ttnn.CoreRange((0, 0), (6, 0))  # 7 rows, 1 column for COL_MAJOR

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({input2_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, block_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_height_bcast_with_block_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 1, 4 * 32)  # height broadcast [1, 7, 1, 128]
    input3_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 32)  # [128, 32]
        input2_shard_shape = (32, 32)  # [32, 32]
        block_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
        input2_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
    else:
        shard_shape = (32, 2 * 32 * 2)  # [32, 128] for COL_MAJOR
        input2_shard_shape = (32, 32)  # [32, 32] for COL_MAJOR
        block_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns
        input2_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({input2_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    predicate_tensor = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if predicate_sharded:
        predicate_tensor = ttnn.to_memory_config(predicate_tensor, block_sharded_mem_config)

    true_tensor = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if true_sharded:
        true_tensor = ttnn.to_memory_config(true_tensor, input2_sharded_mem_config)

    false_tensor = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if false_sharded:
        false_tensor = ttnn.to_memory_config(false_tensor, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(predicate_tensor, true_tensor, false_tensor, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_scalar_bcast_with_height_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    input3_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 4 * 32)  # [128, 128]
        input2_shard_shape = (1 * 32, 32)  # [32, 32] for scalar
    else:
        shard_shape = (4 * 32, 2 * 32 * 2)  # [128, 128] for COL_MAJOR
        input2_shard_shape = (32, 1 * 32)  # [32, 32] for COL_MAJOR

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, height_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_scalar_bcast_with_width_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    input2_shape = (1, 1, 1, 1)  # scalar broadcast [1, 1, 1, 1]
    input3_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 1 * 2 * 32, 32)  # [128, 32]
        input2_shard_shape = (1 * 32, 32)  # [32, 32] for scalar
    else:
        shard_shape = (32, 2 * 1 * 2 * 32)  # [32, 128] for COL_MAJOR
        input2_shard_shape = (32, 1 * 32)  # [32, 32] for COL_MAJOR

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=7),  # 7 cores: (0,0) to (0,6)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=1),  # 1 core: (0,0) to (0,0)
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, width_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_scalar_bcast_with_block_sharding(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, shard_orientation, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    input3_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (2 * 32 * 2, 32)  # [128, 32]
        input2_shard_shape = (32, 32)  # [32, 32] for scalar
        block_core_range = ttnn.CoreRange((0, 0), (3, 6))  # 4 rows, 7 columns
        input2_core_range = ttnn.CoreRange((0, 0), (0, 6))  # 1 row, 7 columns
    else:
        shard_shape = (32, 2 * 32 * 2)  # [32, 128] for COL_MAJOR
        input2_shard_shape = (32, 32)  # [32, 32] for COL_MAJOR
        block_core_range = ttnn.CoreRange((0, 0), (6, 3))  # 7 rows, 4 columns for COL_MAJOR
        input2_core_range = ttnn.CoreRange((0, 0), (6, 0))  # 7 rows, 1 column for COL_MAJOR

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({input2_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, block_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_width_bcast_with_height_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    input2_shape = (5, 7, 2 * 32, 1)  # width broadcast [5, 7, 64, 1]
    input3_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]
    output_shape = (5, 7, 2 * 32, 4 * 32)  # [5, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    shard_shape = (11 * 32, 4 * 32)  # [352, 128] - uneven height sharding
    input2_shard_shape = (11 * 32, 32)  # [352, 32] - uneven height sharding

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, height_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_width_bcast_with_width_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    input2_shape = (1, 1, 2 * 32, 1)  # width broadcast [1, 1, 64, 1]
    input3_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]
    output_shape = (1, 2, 2 * 32, 40 * 32)  # [1, 2, 64, 1280]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    shard_shape = (32 * 2 * 2, 11 * 32)  # [128, 352] - uneven width sharding
    input2_shard_shape = (32 * 2 * 1, 32)  # [64, 32]

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),  # 1 core
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, width_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_width_bcast_with_block_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 3 * 32)  # [2, 7, 64, 96]
    input2_shape = (1, 7, 32 * 2, 1)  # width broadcast [1, 7, 64, 1]
    input3_shape = (2, 7, 32 * 2, 3 * 32)  # [2, 7, 64, 96]
    output_shape = (2, 7, 32 * 2, 3 * 32)  # [2, 7, 64, 96]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 2 * 32)  # [192, 64] - uneven block sharding
    input2_shard_shape = (32 * 2, 32)  # [64, 32]

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 4))}),  # 2 rows x 5 cols = 10 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, block_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_height_bcast_with_height_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 1, 4 * 32)  # height broadcast [1, 7, 1, 128]
    input3_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randn(input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.randn(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.randn(input3_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 4 * 32)  # [192, 128] - uneven height sharding
    input2_shard_shape = (1 * 32, 4 * 32)  # [32, 128] - uneven height sharding

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),  # 5 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, height_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_height_bcast_with_width_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    # Height broadcast: true broadcasts along height dimension
    input1_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    input2_shape = (1, 1, 1, 7 * 32)  # height broadcast [1, 1, 1, 224]
    input3_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]

    torch_input1 = torch.randint(0, 2, input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.rand(input3_shape, dtype=torch.bfloat16)

    # Uneven sharding: 224 total width, shard width = 64
    # 224 / 64 = 3.5 (uneven division)
    shard_shape = (2 * 1 * 64, 64)  # [128, 64] - uneven width sharding
    input2_shard_shape = (1 * 32, 64)  # [32, 64]

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, width_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_height_bcast_with_block_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)

    input1_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    input2_shape = (1, 7, 1, 5 * 32)  # height broadcast [1, 7, 1, 160]
    input3_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    output_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]

    torch_input1 = torch.randint(0, 2, input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.rand(input3_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 2 * 32)  # [192, 64]
    input2_shard_shape = (32 * 2, 2 * 32)  # [64, 64]

    block_core_range = ttnn.CoreRange((0, 0), (2, 4))  # 3 rows x 5 cols = 15 cores

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Use same core grid for broadcast tensor (ternary kernel limitation for height broadcast)
    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({block_core_range}),  # Same core range as main tensor
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, block_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_scalar_bcast_with_height_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    input2_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    input3_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]

    torch_input1 = torch.randint(0, 2, input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.rand(input3_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 4 * 32)  # [192, 128] - uneven height sharding
    input2_shard_shape = (2 * 32, 32)  # [64, 32]

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 4))}),  # 5 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, height_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_scalar_bcast_with_width_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    input2_shape = (1, 1, 1, 1)  # scalar broadcast [1, 1, 1, 1]
    input3_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 64, 7 * 32)  # [2, 1, 64, 224]

    torch_input1 = torch.randint(0, 2, input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.rand(input3_shape, dtype=torch.bfloat16)

    shard_shape = (2 * 1 * 64, 2 * 32)  # [128, 64] - uneven width sharding
    input2_shard_shape = (1 * 32, 32)  # [32, 32]

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))}),  # 4 cores
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),  # 1 core
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, width_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("input1_sharded", [True, False])
@pytest.mark.parametrize("input2_sharded", [True, False])
@pytest.mark.parametrize("input3_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_scalar_bcast_with_block_sharding_uneven(
    device, input1_sharded, input2_sharded, input3_sharded, out_sharded, ttnn_op
):
    torch.manual_seed(0)
    input1_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    input2_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    input3_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]
    output_shape = (2, 7, 32 * 2, 5 * 32)  # [2, 7, 64, 160]

    torch_input1 = torch.randint(0, 2, input1_shape, dtype=torch.bfloat16)
    torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)
    torch_input3 = torch.rand(input3_shape, dtype=torch.bfloat16)

    shard_shape = (3 * 32 * 2, 2 * 32)  # [192, 64] - uneven block sharding
    input2_shard_shape = (32, 32)  # [32, 32]

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 4))}),  # 15 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input2_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=input2_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),  # 7 cores
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input1, torch_input2, torch_input3)

    ttnn_tensor1 = ttnn.from_torch(
        torch_input1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input1_sharded:
        ttnn_tensor1 = ttnn.to_memory_config(ttnn_tensor1, block_sharded_mem_config)

    ttnn_tensor2 = ttnn.from_torch(
        torch_input2, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input2_sharded:
        ttnn_tensor2 = ttnn.to_memory_config(ttnn_tensor2, input2_sharded_mem_config)

    ttnn_tensor3 = ttnn.from_torch(
        torch_input3, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input3_sharded:
        ttnn_tensor3 = ttnn.to_memory_config(ttnn_tensor3, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn_op(ttnn_tensor1, ttnn_tensor2, ttnn_tensor3, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert assert_with_pcc(output_tensor, torch_output_tensor)
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize(
    "dtype_pt, dtype_tt",
    ([torch.bfloat16, ttnn.bfloat16],),
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addcmul,
    ],
)
def test_ternary_ttt_sharded_shardspec_mixed_buffer_type(dtype_pt, dtype_tt, device, ttnn_op):
    torch.manual_seed(0)
    dram_grid_size = device.dram_grid_size()
    input_shape = (1, 1, dram_grid_size.x * dram_grid_size.y * 32, 32)

    input1_pt = torch.randn(input_shape, dtype=torch.bfloat16)
    input2_pt = torch.randn(input_shape, dtype=torch.bfloat16)
    input3_pt = torch.randn(input_shape, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (2, 3))})
    N, C, H, W = input_shape
    n_cores = 12
    import math

    # L1 sharded config for predicate
    shard_spec = ttnn.ShardSpec(
        shard_grid, [math.ceil((N * C * H) / n_cores / 32) * 32, W], ttnn.ShardOrientation.ROW_MAJOR
    )
    input1_config = ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

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
    input2_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec)

    # False tensor uses DRAM interleaved
    input3_config = ttnn.DRAM_MEMORY_CONFIG

    input1_tt = ttnn.from_torch(
        input1_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input1_config,
    )
    input2_tt = ttnn.from_torch(
        input2_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input2_config,
    )
    input3_tt = ttnn.from_torch(
        input3_pt,
        dtype=dtype_tt,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input3_config,
    )

    golden_fn = ttnn.get_golden_function(ttnn_op)
    out_pt = golden_fn(input1_pt, input2_pt, input3_pt)
    out_tt = ttnn_op(input1_tt, input2_tt, input3_tt)
    assert assert_with_pcc(ttnn.to_torch(out_tt), out_pt)
