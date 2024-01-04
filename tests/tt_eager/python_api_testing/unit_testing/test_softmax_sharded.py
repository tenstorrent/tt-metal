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
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "casual_mask",
    [True, False],
    ids=["causal", "no-causal"],
)
@pytest.mark.parametrize(
    "cb_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
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
    (ttl.tensor.DataType.BFLOAT8_B,),
    ids=["BFLOAT8_B"],
)
def test_softmax(device, in_dtype, cb_dtype, in0_mem_config, casual_mask):
    torch.manual_seed(0)
    sm_op = ttl.operations.primary.transformers.scale_mask_softmax_in_place

    fuse_head = 2

    grid_size = (12, 8)
    batch = grid_size[0]
    num_cores_r = grid_size[1]
    input_shape = (batch, 1, num_cores_r * fuse_head * 384, 384)
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
        attention_mask = attention_mask.reshape(batch, 1, -1, 32)
        attention_mask_t = ttl.tensor.Tensor(
            attention_mask,
            ttl.tensor.DataType.BFLOAT16,
        ).to(device, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1))
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
        [M // grid_size[1], K // grid_size[0]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    program_config = ttl.operations.primary.transformers.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=6,
        block_h=12 * fuse_head,
        block_w=12,
        math_fidelity=ttl.tensor.MathFidelity.HiFi4,
        im_data_format=cb_dtype,
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
        attention_mask = attention_mask.repeat(1, 1, 16, 1)

    for i in range(batch):
        golden_output_tensor = input_tensor[i] * scale + attention_mask[i]
        golden_output_tensor = torch.softmax(golden_output_tensor, dim=-1)

        allclose, output = comp_pcc(
            tt_output_tensor[i],
            golden_output_tensor,
        )
        logger.info(output)
        assert allclose, f"FAILED: {output}"
