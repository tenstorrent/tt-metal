# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import math

import tt_lib as ttl
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import print_diff_argmax, comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "casual_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["FLOAT32", "BFLOAT8_B"],
)
def test_softmax(device, in_dtype, in0_mem_config, casual_mask):
    if is_grayskull() and in_dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(0)
    sm_op = ttl.operations.primary.transformers.scale_mask_softmax_in_place

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

    if casual_mask == False:
        # attention_mask = torch.zeros(1, 1, 1, 384 * batch)
        attention_mask = torch.rand(batch, 1, 1, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask32 = tilize_to_list(pad_weight(attention_mask))
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask32,
            [batch, 1, 32, 384],
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            device,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )
    else:
        attention_mask = torch.rand(batch, 1, 384, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask32 = tilize_to_list(pad_weight(attention_mask))
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask32,
            [batch, 1, 384, 384],
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            device,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = torch2tt_tensor(input_tensor, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [fuse_head * 384, 384],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4 if in_dtype == ttl.tensor.DataType.FLOAT32 else 6,
        block_h=12 * fuse_head,
        block_w=12,
    )

    tt_output_sharded = sm_op(
        in1_t_shard, scale, attention_mask_t, program_config=program_config, is_causal_mask=casual_mask
    )

    tt_output = ttl.tensor.sharded_to_interleaved(tt_output_sharded, in0_mem_config)
    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

    if casual_mask == False:
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


@pytest.mark.parametrize(
    "casual_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttl.tensor.DataType.FLOAT32, ttl.tensor.DataType.BFLOAT8_B),
    ids=["FLOAT32", "BFLOAT8_B"],
)
def test_scale_mask_softmax_rm(device, in_dtype, in0_mem_config, casual_mask):
    if is_grayskull() and in_dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(0)
    sm_op = ttl.operations.primary.transformers.scale_mask_softmax_in_place

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

    if casual_mask == False:
        # attention_mask = torch.zeros(batch, 1, 1, 384)
        attention_mask = torch.rand(batch, 1, 1, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask = attention_mask.reshape(batch, 1, -1, 32)
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask,
            # ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.FLOAT32 if in_dtype == ttl.tensor.DataType.FLOAT32 else ttl.tensor.DataType.BFLOAT16,
        ).to(device, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1))
    else:
        # attention_mask = torch.zeros(batch, 1, 384, 384)
        attention_mask = torch.rand(batch, 1, 384, 384)
        attention_mask = (attention_mask > 0.5).float()
        attention_mask32 = tilize_to_list(pad_weight(attention_mask))
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask32,
            [batch, 1, 384, 384],
            # ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.DataType.FLOAT32 if in_dtype == ttl.tensor.DataType.FLOAT32 else ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            device,
            ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = torch2tt_tensor(input_tensor, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [fuse_head * 384, 384],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.ROW_MAJOR,
    )

    program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4 if in_dtype == ttl.tensor.DataType.FLOAT32 else 6,
        block_h=12 * fuse_head,
        block_w=12,
    )

    tt_output_sharded = sm_op(
        in1_t_shard, scale, attention_mask_t, program_config=program_config, is_causal_mask=casual_mask
    )

    tt_output = ttl.tensor.sharded_to_interleaved(tt_output_sharded, in0_mem_config)
    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

    if casual_mask == False:
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


@pytest.mark.parametrize(
    "shard_orient",
    [ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.ShardOrientation.ROW_MAJOR],
    ids=["CM", "RM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "in0_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttl.tensor.DataType.FLOAT32, ttl.tensor.DataType.BFLOAT8_B),
    ids=["FLOAT32", "BFLOAT8_B"],
)
def test_softmax_with_sharded_mask(device, in_dtype, in0_mem_config, shard_orient):
    torch.manual_seed(0)
    sm_op = ttl.operations.primary.transformers.scale_mask_softmax_in_place

    grid_size = (8, 4)
    input_shape = (1, 32, 32, 1024)
    M = input_shape[2]
    K = input_shape[3]

    scale = 1.0 / 8.0

    attention_mask = torch.rand(1, 32, 32, 1024)
    attention_mask = (attention_mask > 0.5).float()
    attention_mask = torch.where(attention_mask == 1, torch.tensor(0.0), torch.tensor(-float("inf")))
    attention_mask_t = torch2tt_tensor(
        attention_mask,
        device,
        tt_memory_config=in0_mem_config,
        tt_dtype=ttl.tensor.DataType.FLOAT32
        if in_dtype == ttl.tensor.DataType.FLOAT32
        else ttl.tensor.DataType.BFLOAT16,
    )
    attention_mask_t_shard = ttl.tensor.interleaved_to_sharded(
        attention_mask_t,
        grid_size,
        [M, K],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        shard_orient,
    )

    input_tensor = torch.randn(input_shape).bfloat16().float()
    in1_t = torch2tt_tensor(input_tensor, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M, K],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        shard_orient,
    )

    program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4 if in_dtype == ttl.tensor.DataType.FLOAT32 else 8,
        block_h=1,
        block_w=32,
    )

    tt_output_sharded = sm_op(
        in1_t_shard, scale, attention_mask_t_shard, program_config=program_config, is_causal_mask=True
    )

    tt_output = ttl.tensor.sharded_to_interleaved(tt_output_sharded, in0_mem_config)
    tt_output_tensor = tt_output.cpu().to_torch().float()
    tt_output_tensor = torch.Tensor(tt_output_tensor).reshape(input_shape)
    tt_output_tensor = untilize(tt_output_tensor)

    golden_output_tensor = input_tensor * scale + attention_mask
    golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

    allclose, output = comp_pcc(
        tt_output_tensor,
        golden_output_tensor,
    )
    logger.info(output)
    assert allclose, f"FAILED: {output}"
