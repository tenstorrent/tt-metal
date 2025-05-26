# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn

# duration 4212916ns
# theoretical considerations:
# ideal cycles compute = ( M * N * K (in tiles) / num_cores ) * fidelity
#  fidelity:
#   HiFi4: 64
#   HiFi3: 48
#   HiFi2: 32
#   LoFi: 16
# Our  case:
# M = 1024 / 32 = 32
# N = 18432 / 32 = 576
# K = 4608 / 32 = 144
#  num_cores = 64
# ideal_cycles_compute = (32 * 576 * 144 / 64) * 16 = 663552ns
# utilisation = ~15%
# Mem bandwidth considerations:
# Dram bandwidth = 288 GB/s
# in0 has to be read from dram, in1 (weights) has to be read from dram, output has to be written to dram
# If we are running at 100% dram bandwidth:
# to read in0: 1 * 1 * 1024 * 18432 * 2 (float16) / 288GB/s = 131072ns <- in0 sharded makes this go away
# to read in1: 1 * 1 * 18432 * 4608 * 1 (bfp8) / 288GB/s = 294912ns
# to write out: 1 * 1 * 1024 * 4608 * 2 (float16) / 288GB/s = 32768ns <- output sharding makes this go away

# in0, in1 reads is overlapped with compute
# output write is not overlapped with compute, i.e. the output is written once it is fully computed
# From above, we can see that the compute is the bottleneck, not the memory bandwidth.
# This is an example from Falcon7B FF2 MLP matmul in 1k seq_len prefill.


# This went the matmul1d path, and duration is: 4106952ns
def test_matmul_base_no_config(device):
    in0_shape = [1, 1, 1024, 18432]
    in1_shape = [1, 1, 18432, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
    )


# program config:
# matmul2d works the following way (interlaved one, block-sharded is similar):
# we have leftmost column -> it reads in0, and broadcasts it to all cores in the row
# we have topmost row -> it reads in1, and broadcasts it to all cores in it's column
# So types of kernels are:
# - topmost leftmost kernel: reads in0, and in1
# - leftmost column kernels: read in0, and broadcast it to all cores in the row
# - topmost row kernels: read in1, and broadcast it to all cores in the column
# - all other kernels: recieve in0 and in1 from senders
# in0 reader kernels read in total [per_core_M * K] tiles and broadcast to it's row
# in1 reader kernels read in total [per_core_N * K] tiles and broadcast to it's column
# per_core_M  == M / num_cores_column == in case of 8x8 grid, it is M / 8. In program configs, this is expressed in tiles
# per_core_N  == N / num_cores_row == in case of 8x8 grid, it is N / 8   . In program configs, this is expressed in tiles

# MatmulMultiCoreReuseMultiCastProgramConfig    is the program config for matmul2d
# MatmulMultiCoreReuseMultiCast1DProgramConfig  is the program config for matmul1d


# This is the basic matmul2d, with all perf configs set to ones.
# 4746466ns is the duration, a little slower then the first one.
def test_matmul_base_2d(device):
    in0_shape = [1, 1, 1024, 18432]
    in1_shape = [1, 1, 18432, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]  # = 32 / 8 = 4
    per_core_N = (4608 // 32) // grid_size[1]  # = 144 / 8 = 18
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,  # gelu goes here
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


# in0_block_w parameter is used to control how many number of "partial" matmuls are we going to have.
# It is expressed in tiles.
# K must be divsible by in0_block_w
# In our case K (inner dim) is 18432 or 576 tiles.
# If in0_block_w is 2, we have 576 / 2 partial matmuls, with their results beign accumulated over time.
# So this means:
# - the bigger the in0_block_w the less "matmul loops" we have, and the number of blocks is smaller.
# - valid in0_block_w values are divisors of K (576) = 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, etc.
# A block is what is being computed on a single core at a time.
# General rule is that the bigger the in0_block_w the better the performance will be, but there are some edge cases here as well :)
# So, each core will compute, at a time:
# 1) [per_core_M, in0_block_w] * [in0_block_w, per_core_N] matrix multiply
# 2) produce a partial results of size [per_core_M, per_core_N]
# 3) accumulate the partial results into the output matrix

# in0_block_w dictates the block size, and blocks must be stored in the L1 CB's, therefore that parameter is constrained by the L1 size, and should be upped as much as we have L1 space for it.
# Matmul2D main memory concerns:
# - we need a input0 CB size of [per_core_M, in0_block_w]
# - we need a input1 CB size of [in0_block_w, per_core_N]
# - we need a output CB size of [per_core_M, per_core_N]
# - if output is say bfp8, then we need as well a [per_core_M, per_core_N] CB of float16 in order to accumulate the results, in addition to the output CB of [per_core_M, per_core_N] bfp8. This is because, if we accumulate in bfloat16, and output is bfloat16, the "accumulation" and output CB buffers are aliased, and in the bfp8 case they are not,


# in0_block_w also affects data movement performance.
# Task: experiment with different in0_block_w values and see perf:
# 18 is the biggest value that fits in our case, giving us: 2378019ns of perf
def test_matmul_2d_in0_block_w(device):
    in0_shape = [1, 1, 1024, 18432]
    in1_shape = [1, 1, 18432, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]  # = 32 / 8 = 4
    per_core_N = (4608 // 32) // grid_size[1]  # = 144 / 8 = 18
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=18,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,  # gelu goes here
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


# When computing a matmul we are doing a [per_core_m, in0_block_w] * [in0_block_w, per_core_N] matrix multiply at a time on core.
# out_subblock_h must divide per_core_M
# out_subblock_w must divide per_core_N
# out_subblock_h and out_subblock_w define what parts of the matmul loop are done in one go on the compute engine itself. I'm not 100% sure of this, but think of it like this:
# subblocks do NOT influcence the memory requirements.
# (1, 1) is the worst combination -> in that case matmul is trisc bounded, and we can do some experiments here to confirm that:
# increase fidelities in the above case:
# HiFi4 duration 2899608ns
# HiFi3 duration 2379767ns
# HiFi2 duration 2366617ns
# LoFi duration (baseline) 2385454ns
# HiFi4 is 64 cycles per tile, HiFi3 is 48 cycles per tile, HiFi2 is 32 cycles per tile, LoFi is 16 cycles per tile.
# We can see that only HiFi4 is having some perf degradation compared to others, as it pushes it away from the trisc bound.


# out_subblock_h * out_subblock_w needs to be < 8.
# via brute fore, we find the best combiation :)
# There is no rule except that generally the bigger the h * w the better perf, but may not be the case.
# Experiment: sweep variations subblocks and see perf
# I've picked the (1, 6) subblock size, giving us the: 1251439ns duration.
# Experiment: when pickes the best subblock combo, sweep fidelites and see perf.
# At this point, matmul is: 1251439ns duration which is >50% util.
def test_matmul_subblocks(device):
    in0_shape = [1, 1, 1024, 18432]
    in1_shape = [1, 1, 18432, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]  # = 32 / 8 = 4
    per_core_N = (4608 // 32) // grid_size[1]  # = 144 / 8 = 18
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=18,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,  # gelu goes here
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


# At the above point, we have saturated the compute engine, and matmul is unpacker bound :(.
# The only way to go downward from this is to reduce the amount of data that is being unpacked.
# We can do that by using bfp4 for weights.
# If we just switch that, we get duration of 882478ns. This is > 70% utilisation.
def test_matmul_bfp4(device):
    in0_shape = [1, 1, 1024, 18432]
    in1_shape = [1, 1, 18432, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat4_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]  # = 32 / 8 = 4
    per_core_N = (4608 // 32) // grid_size[1]  # = 144 / 8 = 18
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=18,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,  # gelu goes here
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


# If we reduced in1 df to bfp4, we also get more memory in L1, as for input 1 we have to store [in0_block_w, per_core_N]. This is ~2x smaller than the above case, and we can leverage that.
# This is giving us the best perf here possible: 863915ns.
# Experiments - try sharding here (or commenting in0_reads, and output writes to see how it affects perf):
def test_matmul_updated_in0_block_w(device):
    in0_shape = [1, 1, 1024, 18432]
    in1_shape = [1, 1, 18432, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    height_sharded_memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat4_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]  # = 32 / 8 = 4
    per_core_N = (4608 // 32) // grid_size[1]  # = 144 / 8 = 18
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=32,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,  # gelu goes here
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


# Todo: try tunning this with matmul1d, and see what perf can we get.
# Todo: try using the new features (such as using 16 tiles for subblock_h * w)
# Todo: investigate out_block_h <- reduces L1
# Todo: run with bf16 weights

# source issue: https://github.com/tenstorrent/tt-metal/issues/9723
