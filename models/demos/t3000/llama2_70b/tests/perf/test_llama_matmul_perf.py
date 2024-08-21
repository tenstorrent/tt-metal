# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import torch
import pytest
from loguru import logger

import ttnn

from models.utility_functions import torch2tt_tensor, tt2torch_tensor

DMODEL = 8 * 1024
FF_DIM = 32 * 1024
USE_ACC = True

DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
BFP8_DTYPE = ttnn.bfloat8_b
WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_FP16_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


class Decode_FF1:
    def __init__(self, device):
        self.weight = torch2tt_tensor(
            torch.randn(DMODEL, FF_DIM // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )

        self.prog_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
        )

    def __call__(self, x):
        ff_out = ttnn.matmul(
            x,
            self.weight,
            program_config=self.prog_config,
            memory_config=WIDTH_SHARDED_MEMCFG,
            dtype=BFP8_DTYPE,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
        )

        return ff_out


class Decode_FF2:
    def __init__(self, device):
        self.weight = torch2tt_tensor(
            torch.randn(FF_DIM, DMODEL // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )

        self.prog_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def __call__(self, x):
        # Assume interleaved input
        ff_out = ttnn.matmul(
            x,
            self.weight,
            program_config=self.prog_config,
            memory_config=WIDTH_SHARDED_MEMCFG,
            dtype=BFP8_DTYPE,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
        )

        return ff_out


class Prefill_MLP_128:
    def __init__(self, device):
        self.w1 = torch2tt_tensor(
            torch.randn(DMODEL, FF_DIM // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )
        self.w3 = torch2tt_tensor(
            torch.randn(DMODEL, FF_DIM // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )
        self.w2 = torch2tt_tensor(
            torch.randn(FF_DIM, DMODEL // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )

        self.prog_config1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # how much inner dim you take each time
            out_subblock_h=2,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
            fuse_batch=True,
        )

        self.prog_config3 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # how much inner dim you take each time
            out_subblock_h=2,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size
            fused_activation=None,
            mcast_in0=True,
            fuse_batch=True,
        )

        self.prog_config2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=32,  # how much inner dim you take each time
            out_subblock_h=4,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=4,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=1,  # N / TILE_WIDTH / Grid_Size
            fused_activation=None,
            mcast_in0=True,
            fuse_batch=True,
        )

    def __call__(self, x, x_allgather):
        # Assume interleaved input
        ff1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.prog_config1,
            dtype=BFP8_DTYPE,
            memory_config=WIDTH_SHARDED_MEMCFG,
            compute_kernel_config=COMPUTE_KERNEL_FP16_CONFIG,
        )
        ff3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.prog_config3,
            dtype=BFP8_DTYPE,
            memory_config=WIDTH_SHARDED_MEMCFG,
            compute_kernel_config=COMPUTE_KERNEL_FP16_CONFIG,
        )

        out = ttnn.mul(
            ff1_out,
            ff3_out,
            memory_config=WIDTH_SHARDED_MEMCFG,
        )

        out.deallocate()
        x.deallocate()
        ff2_out = ttnn.matmul(
            x_allgather,
            self.w2,
            program_config=self.prog_config2,
            dtype=BFP8_DTYPE,
            memory_config=WIDTH_SHARDED_MEMCFG,
            compute_kernel_config=COMPUTE_KERNEL_FP16_CONFIG,
        )
        return out


def run_prefill_MLP_128(
    device,
):
    n_cores = 32
    start_idx = (0, 0)
    end_idx = (7, 3)

    inp_shape = (1, 128, DMODEL)
    inp_allgather_shape = (1, 128, FF_DIM)

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
                inp_shape[-2],
                inp_shape[-1] // n_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    inp_allgather_mem_config = ttnn.MemoryConfig(
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
                inp_allgather_shape[-2],
                inp_allgather_shape[-1] // n_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    pt_in = torch.randn(*inp_shape)
    tt_in = torch2tt_tensor(pt_in, device, tt_memory_config=inp_mem_config)
    pt_in_allgather = torch.randn(*inp_allgather_shape)
    tt_in_allgather = torch2tt_tensor(pt_in_allgather, device, tt_memory_config=inp_allgather_mem_config)
    # TT hardware execution -------------------------------------------------------------
    tt_model = Prefill_MLP_128(device)

    tt_out = tt_model(tt_in, tt_in_allgather)


class Prefill_MLP_2k:
    def __init__(self, device):
        self.w1 = torch2tt_tensor(
            torch.randn(DMODEL, FF_DIM // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )
        self.w3 = torch2tt_tensor(
            torch.randn(DMODEL, FF_DIM // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )
        self.w2 = torch2tt_tensor(
            torch.randn(FF_DIM, DMODEL // 8),
            device,
            tt_memory_config=DRAM_MEMCFG,
            tt_dtype=BFP8_DTYPE,
        )

        self.block_sharded = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 7),
                        ),
                    }
                ),
                [
                    2048 // 8,
                    4096 // 8,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        in0_block_w = 4

        self.prog_config1 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            # [2048 x 8192] x [8192 x 4096]
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in0_block_w,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            fused_activation=ttnn.UnaryOpType.SILU,
            transpose_mcast=False,
        )

        self.prog_config3 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            # [2048 x 8192] x [8192 x 4096]
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in0_block_w,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=16,  # N / TILE_WIDTH / Grid_Size
            fused_activation=None,
            transpose_mcast=False,
        )

        self.prog_config2 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            # [2048 x 32768] x [32768 x 1024]
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=in0_block_w,  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=8,  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size
            fused_activation=None,
            transpose_mcast=False,
        )

    def __call__(self, x, x_allgather):
        # Assume interleaved input
        ff1_out = ttnn.matmul(
            x,
            self.w1,
            program_config=self.prog_config1,
            memory_config=self.block_sharded,
            compute_kernel_config=COMPUTE_KERNEL_FP16_CONFIG,
        )
        ff3_out = ttnn.matmul(
            x,
            self.w3,
            program_config=self.prog_config3,
            memory_config=self.block_sharded,
            compute_kernel_config=COMPUTE_KERNEL_FP16_CONFIG,
        )

        out = ttnn.mul(ff1_out, ff3_out, memory_config=self.block_sharded)
        ff1_out.deallocate()
        ff3_out.deallocate()
        out.deallocate()

        ff2_out = ttnn.matmul(
            x_allgather,
            self.w2,
            program_config=self.prog_config2,
            compute_kernel_config=COMPUTE_KERNEL_FP16_CONFIG,
        )
        return


def run_prefill_MLP_2k(
    device,
):
    n_cores = 32
    start_idx = (0, 0)
    end_idx = (7, 3)

    inp_shape = (1, 2048, DMODEL)
    inp_allgather_shape = (1, 2048, FF_DIM)

    pt_in = torch.randn(*inp_shape)
    tt_in = torch2tt_tensor(pt_in, device, tt_memory_config=DRAM_MEMCFG)
    pt_in_allgather = torch.randn(*inp_allgather_shape)
    tt_in_allgather = torch2tt_tensor(pt_in_allgather, device, tt_memory_config=DRAM_MEMCFG)
    # TT hardware execution -------------------------------------------------------------
    tt_model = Prefill_MLP_2k(device)

    tt_out = tt_model(tt_in, tt_in_allgather)


def run_decode_ff1(
    device,
):
    n_cores = 32
    start_idx = (0, 0)
    end_idx = (7, 3)

    inp_shape = (1, 32, DMODEL)

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
                inp_shape[-2],
                inp_shape[-1] // n_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    pt_in = torch.randn(*inp_shape)
    tt_in = torch2tt_tensor(pt_in, device, tt_memory_config=inp_mem_config)

    # TT hardware execution -------------------------------------------------------------
    tt_model = Decode_FF1(device)

    tt_out = tt_model(tt_in)


def run_decode_ff2(
    device,
):
    n_cores = 32
    start_idx = (0, 0)
    end_idx = (7, 3)

    inp_shape = (1, 32, FF_DIM)

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
                inp_shape[-2],
                inp_shape[-1] // n_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    pt_in = torch.randn(*inp_shape)
    tt_in = torch2tt_tensor(pt_in, device, tt_memory_config=inp_mem_config)

    # TT hardware execution -------------------------------------------------------------
    tt_model = Decode_FF2(device)

    tt_out = tt_model(tt_in)


def test_decode_ff1(
    device,
):
    run_decode_ff1(
        device,
    )


def test_decode_ff2(
    device,
):
    run_decode_ff2(
        device,
    )


def test_prefill_ff1ff3_128(
    device,
):
    run_prefill_MLP_128(
        device,
    )


def test_prefill_2k(
    device,
):
    run_prefill_MLP_2k(
        device,
    )
