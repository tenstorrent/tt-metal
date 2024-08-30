# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import numpy as np
import ttnn

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
import torch
import pytest


@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["out_DRAM"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=["in0_DRAM"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b,),
    ids=["BFLOAT8_B"],
)
def test_split_query_key_value_and_split_heads_with_program_cache(device, dtype, in0_mem_config, out_mem_config):
    torch.manual_seed(1234)

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    num_heads = 16
    grid_size = (12, 8)
    batch = grid_size[0]
    input_shape = [batch, 1, 384, 384 * grid_size[1]]
    M = input_shape[2] * batch
    K = input_shape[3]

    in0 = torch.randn(input_shape)
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=dtype)
    in0_t_shard = ttnn.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )

    q, k, v = ttnn.experimental.split_query_key_value_and_split_heads(
        in0_t_shard,
        ttnn.CoreCoord(grid_size[0], grid_size[1]),
        memory_config=sharded_mem_config,
        num_heads=num_heads,
    )

    tt_q = ttnn.sharded_to_interleaved(q, out_mem_config)
    tt_q = tt_q.cpu().to_torch().float()
    tt_q = untilize(tt_q)
    tt_k = ttnn.sharded_to_interleaved(k, out_mem_config)
    tt_k = tt_k.cpu().to_torch().float()
    tt_k = untilize(tt_k)
    tt_v = ttnn.sharded_to_interleaved(v, out_mem_config)
    tt_v = tt_v.cpu().to_torch().float()
    tt_v = untilize(tt_v)

    fused_qkv_heads = torch.split(in0, input_shape[-1] // grid_size[1], dim=-1)
    ref_q_list = []
    ref_k_list = []
    ref_v_list = []
    for head in fused_qkv_heads:
        (ref_q, ref_k, ref_v) = torch.split(head, head.shape[-1] // 3, dim=-1)
        ref_q_list.append(ref_q)
        ref_k_list.append(ref_k)
        ref_v_list.append(ref_v)

    ref_q = torch.cat(ref_q_list, dim=-1)
    ref_k = torch.cat(ref_k_list, dim=-1)
    ref_v = torch.cat(ref_v_list, dim=-1)

    height = ref_q.shape[2]
    heads = ref_q.shape[-1] // 64

    ref_q = ref_q.reshape(batch, 1, height, heads, 64).transpose(1, 3).reshape(batch, heads, height, 64)
    ref_k = ref_k.reshape(batch, 1, height, heads, 64).transpose(1, 3).reshape(batch, heads, height, 64).transpose(2, 3)
    ref_v = ref_v.reshape(batch, 1, height, heads, 64).transpose(1, 3).reshape(batch, heads, height, 64)

    passing_pcc_q, output_pcc_q = comp_pcc(tt_q, ref_q, 0.99)
    logger.info(output_pcc_q)
    passing_pcc_k, output_pcc_k = comp_pcc(tt_k, ref_k, 0.99)
    logger.info(output_pcc_k)
    passing_pcc_v, output_pcc_v = comp_pcc(tt_v, ref_v, 0.99)
    logger.info(output_pcc_v)
    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v
