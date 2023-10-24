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
@pytest.mark.parametrize("fidelity", [ttl.tensor.MathFidelity.LoFi], ids=["LoFi"])
@pytest.mark.parametrize("has_bias", [False], ids=["bias"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation",
    [
        # (False, True, True, 12*128, 1024, 1024, None),
        # one core
        # (False, False, False, 128, 256, 128, None),
        # in1-L1-fusedQKV
        (False, True, True, 4608, 1024, 3072, None), # both sharded
        # (False, True, False, 4608, 1024, 3072, None), # out sharded, in0 interleaved
        # (False, False, True, 4608, 1024, 3072, None), # out interleaved, in0 sharded
        # (False, False, False, 4608, 1024, 3072, None), # out interleaved, in0 interleaved
        # # in1-dram-fusedQKV
        (True, True, True, 4608, 1024, 3072, None),
        # (True, True, False, 4608, 1024, 3072, None),
        # (True, False, True, 4608, 1024, 3072, None),
        # (True, False, False, 4608, 1024, 3072, None),
        # # in1-L1-selfout
        (False, True, True, 4608, 1024, 1024, None),
        # (False, True, False, 4608, 1024, 1024, None),
        # (False, False, True, 4608, 1024, 1024, None),
        # (False, False, False, 4608, 1024, 1024, None),
        # # in1-dram-selfout
        (True, True, True, 4608, 1024, 1024, None),
        # (True, True, False, 4608, 1024, 1024, None),
        # (True, False, True, 4608, 1024, 1024, None),
        # (True, False, False, 4608, 1024, 1024, None),
        # # in1-L1-ff1
        # (False, True, True, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # (False, True, False, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # (False, False, True, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # (False, False, False, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # # in1-dram-ff1
        # (True, True, True, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # (True, True, False, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # (True, False, True, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # (True, False, False, 4608, 1024, 4096, (ttl.tensor.FusibleActivation.GELU, True)),
        # # in1-L1-ff1 - no Gelu
        (False, True, True, 4608, 1024, 4096, None),
        # (False, True, False, 4608, 1024, 4096, None),
        # (False, False, True, 4608, 1024, 4096, None),
        # (False, False, False, 4608, 1024, 4096, None),
        # # in1-dram-ff1 - no Gelu
        (True, True, True, 4608, 1024, 4096, None),
        # (True, True, False, 4608, 1024, 4096, None),
        # (True, False, True, 4608, 1024, 4096, None),
        # (True, False, False, 4608, 1024, 4096, None),
        # # in1-L1-ff2
        (False, True, True, 4608, 4096, 1024, None),
        # (False, True, False, 4608, 4096, 1024, None),
        # (False, False, True, 4608, 4096, 1024, None),
        # (False, False, False, 4608, 4096, 1024, None),
        # # in1-dram-ff2
        (True, True, True, 4608, 4096, 1024, None),
        # (True, True, False, 4608, 4096, 1024, None),
        # (True, False, True, 4608, 4096, 1024, None),
        # (True, False, False, 4608, 4096, 1024, None),
    ]
)
def test_bert_linear(device, fidelity, in0_sharded, out_sharded, in1_in_dram, has_bias, M, K, N, activation, function_level_defaults):
    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, N]
    grid_size = (12, 8)
    # grid_size = (2, 2)
    shard_shape = [M // grid_size[0], K // grid_size[1]] # shard height, width

    in0_block_w = K // grid_size[1] // 32 # 16
    in0_block_h = M // grid_size[0]  // 32
    out_block_h = M // grid_size[0] // 32
    out_block_w = N // grid_size[1] // 32

    if out_block_w <= 8:
        out_subblock_w = out_block_w
        out_subblock_h = 8 // out_subblock_w
    else:
        out_subblock_h = 1
        out_subblock_w = 8 // out_subblock_h
        while out_block_w % out_subblock_w != 0:
            out_subblock_w = out_block_w // 2


    # in0_block_w = 1
    # out_subblock_w = 1
    # out_subblock_h = 1

    print("in0 block w h " + str(in0_block_w*32) + " " + str(in0_block_h*32))
    print("in1 block w h " + str(out_block_w*32) + " " + str(in0_block_w*32))
    print("out block w h " + str(out_block_w*32) + " " + str(out_block_h*32))
    print("out subblock w h " + str(out_subblock_w*32) + " " + str(out_subblock_h*32))

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

    if in0_sharded:
        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B)
    else:
        in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttl.tensor.DataType.BFLOAT8_B)

    if in1_in_dram:
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttl.tensor.DataType.BFLOAT8_B)
    else:
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttl.tensor.DataType.BFLOAT8_B)

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttl.tensor.DataType.BFLOAT8_B)[0]

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
        # transpose_mcast=False,
        fused_activation=activation,
    )

    if has_bias:
        output_t = ttl.operations.primary.matmul(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            output_mem_config=output_mem_config,
            math_fidelity=fidelity,
        )
    else:
        output_t = ttl.operations.primary.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            output_mem_config=output_mem_config,
            math_fidelity=fidelity,
        )

    if out_sharded:
        output_t = ttl.tensor.sharded_to_interleaved(output_t, interleaved_mem_config_L1)

    pt_out = in0 @ in1

    if has_bias:
        pt_out = pt_out + bias

    if activation != None:
        pt_out = torch.nn.functional.gelu(pt_out)
    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert True
