# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import tt_lib as ttl
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor, pad_by_zero, get_devices_for_t3000

from torch.nn import functional as F


@pytest.mark.parametrize(
    "shard_orientation",
    (ttl.tensor.ShardOrientation.ROW_MAJOR,),
)
@pytest.mark.parametrize(
    "output_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in1_sharded",
    (False,),
)
@pytest.mark.parametrize(
    "in0_sharded",
    (True, False),
)
@pytest.mark.parametrize(
    "batch, K, seq_len, q_heads, kv_heads",
    (
        (1, 64, 128, 16, 1),  # 8 chip pre-attn matmul shapes
        # (1, 1024 + 32, 64, 16, 1),  # 8 chip post-attn matmul shapes
    ),
)
def test_group_attn_matmul(
    batch, K, seq_len, q_heads, kv_heads, in0_sharded, in1_sharded, output_sharded, shard_orientation, device
):
    torch.manual_seed(0)

    num_cores = 16
    compute_grid_size = (8, 2)
    core_range = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 1),
            ),
        }
    )

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )

    # NOTE: Mixed precision is supported as well; but might not have enough space for larger seq_len with BFLOAT16
    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    output_dtype = ttl.tensor.DataType.BFLOAT8_B

    input_shape_a = [batch, q_heads, seq_len, K]
    input_shape_b = [batch, kv_heads, K, seq_len]

    input_tensor_a = torch.randn(input_shape_a).bfloat16()
    input_tensor_b = torch.randn(input_shape_b).bfloat16()

    tt_input_tensor_a = (
        ttl.tensor.Tensor(input_tensor_a, in0_dtype).to(ttl.tensor.Layout.TILE).to(device, interleaved_mem_config)
    )
    tt_input_tensor_b = (
        ttl.tensor.Tensor(input_tensor_b, in1_dtype).to(ttl.tensor.Layout.TILE).to(device, interleaved_mem_config)
    )

    if in0_sharded:
        tt_input_tensor_a = ttl.tensor.interleaved_to_sharded(
            tt_input_tensor_a,
            compute_grid_size,
            [batch * seq_len, K],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if in1_sharded:
        tt_input_tensor_b = ttl.tensor.interleaved_to_sharded(
            tt_input_tensor_b,
            compute_grid_size,
            [kv_heads * K, seq_len],
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if output_sharded:
        output_mem_config = ttl.tensor.MemoryConfig(
            memory_layout=ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttl.tensor.BufferType.L1,
        )
    else:
        output_mem_config = interleaved_mem_config

    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    prg_cfg = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_grid_size,
        in0_block_w=K // 32,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=seq_len * 16 // num_cores // 32,
        per_core_N=seq_len // 32,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )

    tt_output_tensor_on_device = ttl.operations.primary.matmul(
        tt_input_tensor_a,
        tt_input_tensor_b,
        compute_kernel_config=compute_kernel_config,
        output_mem_config=output_mem_config,
        program_config=prg_cfg,
        output_dtype=output_dtype,
    )
    if output_sharded:
        tt_output_tensor_on_device = ttl.tensor.sharded_to_interleaved(
            tt_output_tensor_on_device, interleaved_mem_config
        )

    tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    input_tensor_a = input_tensor_a.to(torch.float)
    input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
    golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

    allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor, 0.9990)
    logger.info(f"PCC: {output}")
    assert allclose, f"FAILED: {output}"
