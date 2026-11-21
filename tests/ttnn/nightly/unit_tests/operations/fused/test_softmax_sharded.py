# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

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
)
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.fused.utility_functions import ttnn_scale_mask_softmax_in_place


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
    # Single-call (no determinism wrapper): the in-place wrapper clones the sharded input, and
    # on Blackhole that extra L1 buffer clashes with the statically-allocated circular buffers.
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
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
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

    assert_numeric_metrics(
        golden_output_tensor,
        tt_output_tensor,
        pcc_threshold=0.999,
        rtol=0.091,
        atol=0.001,
        frobenius_threshold=0.025,
    )


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
    torch.manual_seed(0)
    # Single-call (no determinism wrapper): the in-place wrapper clones the sharded input, and
    # on Blackhole that extra L1 buffer exhausts L1 / clashes with the static circular buffers.
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
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
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

    assert_numeric_metrics(
        tt_output_tensor,
        golden_output_tensor,
        pcc_threshold=0.999,
        rtol=0.091,
        atol=0.001,
        frobenius_threshold=0.025,
    )


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
    torch.manual_seed(0)
    # Single-call (no determinism wrapper): the in-place wrapper clones the sharded input, and
    # on Blackhole that extra L1 buffer exhausts L1 / clashes with the static circular buffers.
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
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
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

    assert_numeric_metrics(
        golden_output_tensor,
        tt_output_tensor,
        pcc_threshold=0.999,
        rtol=0.089,
        atol=0.001,
        frobenius_threshold=0.025,
    )


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
    sm_op = ttnn_scale_mask_softmax_in_place

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
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orient)
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

    assert_numeric_metrics(
        golden_output_tensor,
        tt_output_tensor,
        pcc_threshold=0.999,
        rtol=0.085,
        atol=0.001,
        frobenius_threshold=0.031,
    )


# Regression tests for bounty #64647: sharded softmax mask handling fixes
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_row_major_mask_with_causal_rejected(device):
    """Row-major mask + is_causal_mask=True must be rejected with a clear error, not hang the device."""
    torch.manual_seed(0)

    batch, num_cores_r = 2, 4
    input_shape = (batch, num_cores_r, 128, 128)
    input_tensor = torch.randn(input_shape).bfloat16()

    in1_t = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1))})
    shard_shape = [128, 128]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    # Row-major mask (not TILE layout)
    attention_mask = torch.rand(batch, 1, 128, 128)
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size, subblock_w=4, block_h=4, block_w=4
    )

    # Must raise a clear error about row-major + causal not being supported
    with pytest.raises(RuntimeError, match="Row-major mask with is_causal_mask=True is not supported"):
        ttnn.scale_mask_softmax_in_place(
            in1_t_shard, 1.0, attention_mask_t, program_config=program_config, is_causal_mask=True
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_non_square_causal_mask_correctness(device):
    """Non-square causal masks (block_w != mask_height) must produce correct results without out-of-bounds reads."""
    torch.manual_seed(0)

    batch = 2
    # Non-square: mask height (64) != block width (128)
    input_shape = (batch, 4, 128, 128)
    input_tensor = torch.randn(input_shape).bfloat16().float()

    in1_t = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1))})
    shard_shape = [128, 128]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    # Non-square mask: height 64, width 128
    attention_mask = torch.rand(batch, 1, 64, 128)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size, subblock_w=4, block_h=4, block_w=8
    )

    tt_output_sharded = ttnn.scale_mask_softmax_in_place(
        in1_t_shard, 1.0, attention_mask_t, program_config=program_config, is_causal_mask=True
    )

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    # Golden: repeat mask to match input height, apply causal masking
    attention_mask_repeated = attention_mask.repeat(1, 2, 1, 1)  # Repeat to match 128 height
    golden = input_tensor + attention_mask_repeated
    golden = torch.softmax(golden, dim=-1)

    assert_numeric_metrics(golden, tt_output_tensor, pcc_threshold=0.99, rtol=0.1, atol=0.01, frobenius_threshold=0.05)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_sharded_mask_orientation_mismatch_rejected(device):
    """Sharded mask with different orientation than input must be rejected."""
    torch.manual_seed(0)

    batch = 2
    input_shape = (batch, 4, 128, 128)
    input_tensor = torch.randn(input_shape).bfloat16()

    in1_t = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1))})

    # Input: COL_MAJOR orientation
    input_shard_spec = ttnn.ShardSpec(shard_grid, [128, 128], ttnn.ShardOrientation.COL_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, input_mem_config)

    # Mask: ROW_MAJOR orientation (mismatch!)
    attention_mask = torch.rand(batch, 4, 128, 128)
    mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_shard_spec = ttnn.ShardSpec(shard_grid, [128, 128], ttnn.ShardOrientation.ROW_MAJOR)
    mask_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, mask_shard_spec)
    attention_mask_t_shard = ttnn.to_memory_config(mask_t, mask_mem_config)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size, subblock_w=4, block_h=4, block_w=4
    )

    # Must raise error about orientation mismatch
    with pytest.raises(RuntimeError, match="Sharded mask shard orientation.*must match input shard orientation"):
        ttnn.scale_mask_softmax_in_place(
            in1_t_shard, 1.0, attention_mask_t_shard, program_config=program_config, is_causal_mask=False
        )


# Regression tests for bounty #64575: offset core grid fixes
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("grid_offset", [(0, 0), (1, 0), (0, 1)], ids=["origin", "offset_x", "offset_y"])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.ROW_MAJOR])
def test_softmax_offset_grid_correctness(device, grid_offset, orientation):
    """Sharded softmax must produce correct results on offset grids, not just origin-based grids."""
    torch.manual_seed(0)

    batch, num_cores_r = 2, 4
    input_shape = (batch, num_cores_r, 128, 128)
    input_tensor = torch.randn(input_shape).bfloat16().float()

    in1_t = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_grid_size = device.compute_with_storage_grid_size()
    # Create an offset shard grid
    offset_x, offset_y = grid_offset
    grid_size = [compute_grid_size.x - offset_x, compute_grid_size.y - offset_y]
    start_coord = ttnn.CoreCoord(offset_x, offset_y)
    end_coord = ttnn.CoreCoord(offset_x + grid_size[0] - 1, offset_y + grid_size[1] - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(start_coord, end_coord)})

    shard_shape = [128, 128]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, orientation)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size, subblock_w=4, block_h=4, block_w=4
    )

    tt_output_sharded = ttnn.softmax_in_place(in1_t_shard, program_config=program_config)

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    golden = torch.softmax(input_tensor, dim=-1)

    # Every shard must be updated, not just some
    assert_numeric_metrics(golden, tt_output_tensor, pcc_threshold=0.99, rtol=0.1, atol=0.01, frobenius_threshold=0.05)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("grid_offset", [(0, 0), (1, 0), (0, 1)], ids=["origin", "offset_x", "offset_y"])
def test_scale_mask_softmax_offset_grid_correctness(device, grid_offset):
    """Sharded scale_mask_softmax must produce correct results on offset grids."""
    torch.manual_seed(0)

    batch = 2
    input_shape = (batch, 4, 128, 128)
    input_tensor = torch.randn(input_shape).bfloat16().float()

    in1_t = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_grid_size = device.compute_with_storage_grid_size()
    offset_x, offset_y = grid_offset
    grid_size = [compute_grid_size.x - offset_x, compute_grid_size.y - offset_y]
    start_coord = ttnn.CoreCoord(offset_x, offset_y)
    end_coord = ttnn.CoreCoord(offset_x + grid_size[0] - 1, offset_y + grid_size[1] - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(start_coord, end_coord)})

    shard_shape = [128, 128]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    in1_t_shard = ttnn.to_memory_config(in1_t, sharded_mem_config)

    attention_mask = torch.rand(batch, 1, 128, 128)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask_t = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size, subblock_w=4, block_h=4, block_w=4
    )

    tt_output_sharded = ttnn.scale_mask_softmax_in_place(
        in1_t_shard, 1.0, attention_mask_t, program_config=program_config, is_causal_mask=False
    )

    tt_output_tensor = ttnn.to_layout(tt_output_sharded, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    golden = input_tensor + attention_mask
    golden = torch.softmax(golden, dim=-1)

    assert_numeric_metrics(golden, tt_output_tensor, pcc_threshold=0.99, rtol=0.1, atol=0.01, frobenius_threshold=0.05)
