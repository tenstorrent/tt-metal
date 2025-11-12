# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, assert_with_ulp, assert_allclose
from math import isnan


def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


# TTT,  // tensor-tensor-tensor
# TTS,  // tensor-tensor-scalar
# TST,  // tensor-scalar-tensor
# TSS,  // tensor-scalar-scalar


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_no_bcast_with_block_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_predicate = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.rand(shape, dtype=torch.bfloat16)
    torch_false = torch.rand(shape, dtype=torch.bfloat16)

    shard_shape = (1024 // 2, 1024 // 4)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
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
        true_tensor = ttnn.to_memory_config(true_tensor, block_sharded_mem_config)

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
    assert output_tensor.shape == shape


@pytest.mark.parametrize(
    "predicate_sharded",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "true_sharded",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "false_sharded",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "out_sharded",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_no_bcast_with_height_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    torch.manual_seed(0)
    shape = (1, 1, 256, 256)
    torch_predicate = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.rand(shape, dtype=torch.bfloat16)
    torch_false = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (256 // 4, 256)
    else:
        shard_shape = (256, 256 // 4)

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=4, x=1),
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
        true_tensor = ttnn.to_memory_config(true_tensor, height_sharded_mem_config)

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
    assert output_tensor.shape == shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_no_bcast_with_width_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
    torch.manual_seed(0)
    shape = (1, 1, 256, 256)
    torch_predicate = torch.randint(0, 2, shape, dtype=torch.bfloat16)
    torch_true = torch.rand(shape, dtype=torch.bfloat16)
    torch_false = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (256, 256 // 4)
    else:
        shard_shape = (256 // 4, 256)

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=1, x=4),
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
        true_tensor = ttnn.to_memory_config(true_tensor, width_sharded_mem_config)

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
    assert output_tensor.shape == shape


@pytest.mark.parametrize("predicate_sharded", [True, False])
@pytest.mark.parametrize("true_sharded", [True, False])
@pytest.mark.parametrize("false_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_where_ttt_width_bcast_with_height_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
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
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
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
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
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
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
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
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_where_ttt_width_bcast_with_block_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
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
        block_core_range = ttnn.CoreRange((0, 0), (6, 6))  # 7 rows, 7 columns for COL_MAJOR
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
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_where_ttt_height_bcast_with_block_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded, shard_orientation
):
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
        block_core_range = ttnn.CoreRange((0, 0), (6, 6))  # 7 rows, 7 columns for COL_MAJOR
        true_core_range = ttnn.CoreRange((0, 0), (6, 6))  # 7 rows, 7 columns for COL_MAJOR

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
def test_where_ttt_scalar_bcast_with_height_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    torch.manual_seed(0)
    predicate_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    false_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 2 * 32, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (2 * 32 * 2, 4 * 32)  # [128, 128]
    true_shard_shape = (1 * 32, 32)  # [32, 32] for scalar

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
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
def test_where_ttt_scalar_bcast_with_width_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    torch.manual_seed(0)
    predicate_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    true_shape = (1, 1, 1, 1)  # scalar broadcast [1, 1, 1, 1]
    false_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]
    output_shape = (2, 1, 2 * 32, 7 * 32)  # [2, 1, 64, 224]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (2 * 1 * 2 * 32, 32)  # [128, 32]
    true_shard_shape = (1 * 32, 32)  # [32, 32] for scalar

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))}),
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
def test_where_ttt_scalar_bcast_with_block_sharding(
    device, predicate_sharded, true_sharded, false_sharded, out_sharded
):
    torch.manual_seed(0)
    predicate_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    true_shape = (1, 7, 1, 1)  # scalar broadcast [1, 7, 1, 1]
    false_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]
    output_shape = (2, 7, 32 * 2, 4 * 32)  # [2, 7, 64, 128]

    torch_predicate = torch.randint(0, 2, predicate_shape, dtype=torch.bfloat16)
    torch_true = torch.rand(true_shape, dtype=torch.bfloat16)
    torch_false = torch.rand(false_shape, dtype=torch.bfloat16)

    shard_shape = (2 * 32 * 2, 32)  # [128, 32]
    true_shard_shape = (32, 32)  # [32, 32] for scalar

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    true_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=true_shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
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
