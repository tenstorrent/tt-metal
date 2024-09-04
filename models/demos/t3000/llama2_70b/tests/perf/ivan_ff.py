# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import torch
import pytest
from loguru import logger

import ttnn

from models.utility_functions import torch2tt_tensor, tt2torch_tensor

FF_DIM = int(32 * 1024 / 8)
USE_ACC = True

DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
BFP8_DTYPE = ttnn.bfloat8_b
WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)


class TtFF1:
    def __init__(self, device):
        self.weight = torch2tt_tensor(
            torch.randn(8 * 1024, FF_DIM),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )

    def __call__(self, x, prog_config):
        # Assume interleaved input
        ff_out = ttnn.matmul(
            x,
            self.weight,
            program_config=prog_config,
            memory_config=WIDTH_SHARDED_MEMCFG,
            dtype=BFP8_DTYPE,
        )
        x.deallocate()

        return ff_out


def run_test_ff1(
    device,
):
    n_cores = 32
    start_idx = (0, 0)
    # end_idx = (7, 3)
    end_idx = (3, 7)
    compute_grid = (4, 8)
    cols1_tiles = int(8 * 1024 / 32)
    cols2_tiles = int(FF_DIM / 32)
    # common factors of the above variables
    if not (cols1_tiles % n_cores == 0 and cols2_tiles % n_cores == 0):
        print(f"num_cores: {n_cores}. core_range {start_idx}, {end_idx} not valid")
        assert False

    print(f"num_cores: {n_cores}. core_range {start_idx}, {end_idx}")

    in0_block_w = int(cols1_tiles / n_cores)
    per_core_N = int(cols2_tiles / n_cores)
    max_dst_size = 4 if USE_ACC else 8
    out_subblock_w = max([i for i in range(1, max_dst_size + 1) if (per_core_N % i) == 0])

    prog_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        # compute_with_storage_grid_size=(8,4),
        compute_with_storage_grid_size=compute_grid,
        in0_block_w=in0_block_w,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=out_subblock_w,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=per_core_N,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fuse_batch=True,
        fused_activation=ttnn.UnaryOpType.SILU,
        mcast_in0=True,
    )

    inp_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(*start_idx),
                        ttnn.CoreCoord(*end_idx),
                    ),
                }
            ),
            [
                32,
                int(8 * 1024 / n_cores),
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    pt_in = torch.randn(1, 32, 8 * 1024)
    tt_in = torch2tt_tensor(pt_in, device, tt_memory_config=inp_mem_config)

    # TT hardware execution -------------------------------------------------------------
    tt_model = TtFF1(device)

    tt_out = tt_model(tt_in, prog_config)
    tt_out.deallocate()


def test_ff1(
    device,
):
    run_test_ff1(
        device,
    )
