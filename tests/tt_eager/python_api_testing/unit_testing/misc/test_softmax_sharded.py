# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import math

import ttnn

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero
from models.utility_functions import is_grayskull, skip_for_blackhole


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttnn.bfloat8_b,),
    ids=["bfloat8_b"],
)
def test_softmax_causal_mask(device, in_dtype, in0_mem_config):
    torch.manual_seed(0)
    sm_op = ttnn.scale_mask_softmax_in_place

    fuse_head = 2

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, num_cores_r, fuse_head * 384, 768)
    M = input_shape[2]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = 16
    # scale = 1.0
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    attention_mask = torch.rand(batch, 1, 384, 768)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = ttnn.from_torch(
        input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    grid_coord = ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = [fuse_head * 384, 768]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=8,
        block_h=12 * fuse_head,
        block_w=24,
    )

    tt_output_sharded = sm_op(in1_t_shard, scale, attention_mask_t, program_config=program_config, is_causal_mask=True)

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT, memory_config=in0_mem_config)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    attention_mask = attention_mask.repeat(1, 1, fuse_head, 1)

    golden_output_tensor = input_tensor * scale + attention_mask
    golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

    allclose, output = comp_pcc(
        tt_output_tensor,
        golden_output_tensor,
    )
    logger.info(output)
    assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "causal_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttnn.float32,
        ttnn.bfloat8_b,
    ),
    ids=["float32", "bfloat8_b"],
)
def test_softmax(device, in_dtype, in0_mem_config, causal_mask):
    if is_grayskull() and in_dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(0)
    sm_op = ttnn.scale_mask_softmax_in_place

    fuse_head = 2

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, num_cores_r, fuse_head * 384, 384)
    M = input_shape[2]
    K = input_shape[3] * batch

    hidden_dim = 1024
    num_heads = 16
    # scale = 1.0
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    if causal_mask == False:
        # attention_mask = torch.zeros(1, 1, 1, 384 * batch)
        attention_mask = torch.rand(batch, 1, 1, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask32 = pad_weight(attention_mask)
        attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        attention_mask = torch.rand(batch, 1, 384, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = ttnn.from_torch(
        input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    grid_coord = ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = [fuse_head * 384, 384]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4 if in_dtype == ttnn.float32 else 6,
        block_h=12 * fuse_head,
        block_w=12,
    )

    tt_output_sharded = sm_op(
        in1_t_shard, scale, attention_mask_t, program_config=program_config, is_causal_mask=causal_mask
    )

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT, memory_config=in0_mem_config)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    if causal_mask == False:
        attention_mask = attention_mask.reshape(batch, 1, 1, 384)
    else:
        attention_mask = attention_mask.repeat(1, 1, fuse_head, 1)

    golden_output_tensor = input_tensor * scale + attention_mask
    golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

    allclose, output = comp_pcc(
        tt_output_tensor,
        golden_output_tensor,
    )
    logger.info(output)
    assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "causal_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttnn.float32, ttnn.bfloat8_b),
    ids=["float32", "bfloat8_b"],
)
def test_scale_mask_softmax_rm(device, in_dtype, in0_mem_config, causal_mask):
    if is_grayskull() and in_dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(0)
    sm_op = ttnn.scale_mask_softmax_in_place

    fuse_head = 1

    grid_size = (8, 7)
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.fail(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    batch = grid_size[1]
    num_cores_r = grid_size[0]
    input_shape = (batch, num_cores_r, fuse_head * 384, 384)

    hidden_dim = 1024
    num_heads = 16
    # scale = 1.0
    scale = 1 / math.sqrt(hidden_dim // num_heads)

    if causal_mask == False:
        # attention_mask = torch.zeros(batch, 1, 1, 384)
        attention_mask = torch.rand(batch, 1, 1, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask = attention_mask.reshape(batch, 1, -1, 32)
        attention_mask_t = ttnn.from_torch(
            attention_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
    else:
        # attention_mask = torch.zeros(batch, 1, 384, 384)
        attention_mask = torch.rand(batch, 1, 384, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = ttnn.from_torch(
        input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    grid_coord = ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = [fuse_head * 384, 384]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4 if in_dtype == ttnn.float32 else 6,
        block_h=12 * fuse_head,
        block_w=12,
    )

    tt_output_sharded = sm_op(
        in1_t_shard, scale, attention_mask_t, program_config=program_config, is_causal_mask=causal_mask
    )

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT, memory_config=in0_mem_config)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    if causal_mask == False:
        attention_mask = attention_mask.reshape(batch, 1, 1, 384)
    else:
        attention_mask = attention_mask.repeat(1, 1, fuse_head, 1)

    golden_output_tensor = input_tensor * scale + attention_mask
    golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

    allclose, output = comp_pcc(
        tt_output_tensor,
        golden_output_tensor,
    )
    logger.info(output)
    assert allclose, f"FAILED: {output}"


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "shard_orient",
    [ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.ROW_MAJOR],
    ids=["CM", "RM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttnn.float32, ttnn.bfloat8_b),
    ids=["float32", "bfloat8_b"],
)
def test_softmax_with_sharded_mask(device, in_dtype, in0_mem_config, shard_orient):
    torch.manual_seed(0)
    sm_op = ttnn.scale_mask_softmax_in_place

    grid_size = (8, 4)
    input_shape = (1, 32, 32, 1024)
    M = input_shape[2]
    K = input_shape[3]

    scale = 1.0 / 8.0

    attention_mask = torch.rand(1, 32, 32, 1024)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask = torch.where(attention_mask == 1, torch.tensor(0.0), torch.tensor(-float("inf")))
    mask_dtype = ttnn.float32 if in_dtype == ttnn.float32 else ttnn.bfloat16
    attention_mask_t = ttnn.from_torch(
        attention_mask, dtype=mask_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    grid_coord = ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = [M, K]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orient, False)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    attention_mask_t_shard = ttnn.to_memory_config(attention_mask_t, sharded_mem_config)

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = ttnn.from_torch(
        input_tensor, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config
    )
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4 if in_dtype == ttnn.float32 else 8,
        block_h=1,
        block_w=32,
    )

    tt_output_sharded = sm_op(
        in1_t_shard, scale, attention_mask_t_shard, program_config=program_config, is_causal_mask=True
    )

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT, memory_config=in0_mem_config)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    golden_output_tensor = input_tensor * scale + attention_mask
    golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

    allclose, output = comp_pcc(
        tt_output_tensor,
        golden_output_tensor,
    )
    logger.info(output)
    assert allclose, f"FAILED: {output}"
