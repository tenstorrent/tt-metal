# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math

import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero

pytestmark = pytest.mark.skipif(is_wormhole_b0(), reason="Unsupported parallelizations for WH B0")
@pytest.mark.parametrize("in0_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize(
    "in1_in_dram, M, K, N",
    [
        (False, 3072, 1024, 3072),
        (False, 3072, 1024, 1024),
        (True, 3072, 1024, 4096),
        (True, 3072, 4096, 1024),
    ]
)
def test_sharded_matmul_2d_transposed(device, in0_sharded, out_sharded, in1_in_dram, M, K, N, function_level_defaults):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (12, 8)
    shard_shape = [M // grid_size[0], K // grid_size[1]] # shard height, width

    in0_block_w = K // grid_size[1] // 32
    in0_block_h = shard_shape[0]  // 32
    out_block_h = M // grid_size[0] // 32
    out_block_w = N // grid_size[1] // 32

    if out_block_w <= 8:
        out_subblock_w = out_block_w
        out_subblock_h = 8 // out_subblock_w
    else:
        out_subblock_h = 1
        out_subblock_w = 8 // out_subblock_h
        while out_block_w % out_subblock_w != 0:
            out_subblock_w = out_subblock_w // 2

    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )
    interleaved_mem_config_DRAM = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )
    sharded_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_L1)
    if in1_in_dram:
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM)
        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_DRAM
    else:
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_L1)
        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config_L1)[0]

    if in0_sharded:
        in0_t = ttl.tensor.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M // grid_size[0], K // grid_size[1]] ,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=True,
        fused_activation=None,
    )
    output_t = ttl.operations.primary.matmul(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=output_mem_config,
    )
    if out_sharded:
        if in1_in_dram:
            output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_DRAM)
        else:
            output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
