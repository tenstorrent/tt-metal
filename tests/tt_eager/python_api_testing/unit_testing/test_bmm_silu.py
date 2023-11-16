# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import itertools

from pathlib import Path
from models.utility_functions import pad_by_zero, tt2torch_tensor, torch2tt_tensor, torch_to_tt_tensor_rm

import numpy as np
import torch

import tt_lib as ttl
from models.utility_functions import (
    tilize_to_list,
    untilize,
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0
from loguru import logger

TILE_HEIGHT = TILE_WIDTH = 32

## parameters
# matrix sizes as number of blocks along h and w:
a_height_nblocks = [1, 7]
a_width_nblocks = [1, 7]
b_width_nblocks = [1, 7]
# block sizes as number of tiles along h and w:
a_block_height_ntiles = [4]
a_block_width_ntiles = [4]
b_block_width_ntiles = [16]
# output sublobcking per block:
out_subblock_height_ntiles = [4]  ## == a_block_height_ntiles, <= 8
out_subblock_width_ntiles = [2]  ## == b_block_width_ntiles, <= 8
tilize_a = [True, False]
untilize_out = [True, False]


def test_matmul_1d_silu(device, function_level_defaults):
    in0_shape = [1, 1, 11, 4096]
    in1_shape = [1, 1, 4096, 14336]
    bias_shape = [1, 1, 1, 14336]

    interleaved_mem_config = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.DRAM,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    ttin0_t = ttl.tensor.Tensor(
        torch.flatten(in0).tolist(), in0_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR
    )
    in0_t = ttin0_t.pad_to_tile(0).to(ttl.tensor.Layout.TILE).to(device)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16)
    ttbias = ttl.tensor.Tensor(
        torch.flatten(bias).tolist(), bias_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR
    )
    bias_t = ttbias.pad_to_tile(0).to(ttl.tensor.Layout.TILE).to(device)

    program_config = ttl.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=ttl.tensor.FusibleActivation.SILU,
        mcast_in0=True,
    )
    output_t = ttl.operations.primary.matmul_1d(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        output_mem_config=interleaved_mem_config,
        output_dtype=ttl.tensor.DataType.BFLOAT16,
    )

    pt_out = torch.nn.functional.silu(in0 @ in1 + bias)

    tt_out = tt2torch_tensor(output_t)
    tt_out = tt_out[:, :, :11, :]

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing
