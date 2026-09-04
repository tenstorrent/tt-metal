# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for matmul/linear operations in the MLA (Multi-Head Latent Attention) module.
Tests each matmul operation independently with the same configurations as used in mla.py.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_PER_CHIP

PCC_REQUIRED = 0.99

# Available core grid is 12x10, but due to di/dt and throttling problems, use 11x10 temporarily
compute_with_storage_grid_size_bh_orig = (12, 10)
compute_with_storage_grid_size_11x10 = (11, 10)

# DeepSeek-V3 geometry. The Kimi generations that also run this chunk length carry 64 or 96 heads,
# so only q_a_proj and kv_a_proj_with_mqa (head-count independent) share a per-chip shape with the
# MLA_MATMUL_CONFIG[...][640] entries; the other four are wider here and are tuned in this file.
HIDDEN_SIZE = 7168
NUM_HEADS = 128

# Production prefill chunk. The parametrized shapes below are global; SP sharding on dim 2 over the
# 8 rows of the 8x4 mesh is what makes M per chip 640 rows == 20 tiles.
SEQ_LEN = PREFILL_CHUNK_TOKENS

# Every per_core_M below is sized for exactly that M. Fail at collection with a readable message if
# the production chunk moves, rather than deep inside the matmul validator.
assert PREFILL_CHUNK_TOKENS_PER_CHIP == 640, (
    f"program configs below are tuned for 640 rows/chip, got {PREFILL_CHUNK_TOKENS_PER_CHIP}; retune "
    "per_core_M (and the batched matmuls' block count) before changing the chunk size"
)

# Every program config below is tuned at that per-chip M. K and N do not depend on the sequence
# length, so they carry over from the previously tuned shapes; only the M split changes. per_core_M
# = 2 covers 20 tiles over the 10 grid rows for the 2D mcast matmuls, which measured flat in the
# output subblock width.
#
# The two batched matmuls use MatmulMultiCoreReuse, which is what MLA_MATMUL_CONFIG picks for them
# at 640 and which measured 3.8x / 2.7x faster here than the multicast config the retired 6400-row
# rows used. That factory spreads batch * (M_t / per_core_M) * (N_t / per_core_N) output blocks over
# the grid and writes one block per core, so asking for more blocks than the 110 cores hold leaves
# the tail of the output silently never written (#54798 -- the existing divisibility guard only
# fires for L1-sharded inputs, and these are interleaved). per_core_M = 10 holds both at 32 * 2 = 64
# blocks.
#
# Tile counts in the comments are per chip: [M, K] * [K, N].

# q_a_proj: [20, 56] * [56, 48]. Same per-chip shape as MLA_MATMUL_CONFIG["q_a_proj"][640],
# whose tiling this mirrors.
prog_config_mm0_bh = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=5,
    per_core_M=2,
    per_core_N=5,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# q_b_proj: [20, 48] * [48, 192]
prog_config_mm1_bh = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=6,
    per_core_M=2,
    per_core_N=18,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# wkv_b1: batch 32, [20, 4] * [4, 16]. 64 output blocks.
prog_config_mm2_bh = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
    in0_block_w=4,
    out_subblock_h=2,
    out_subblock_w=4,
    per_core_M=10,
    per_core_N=16,
)

# kv_a_proj_with_mqa: [20, 56] * [56, 18]. Same per-chip shape as
# MLA_MATMUL_CONFIG["kv_a_proj_with_mqa"][640], whose tiling this mirrors.
prog_config_mm3_bh = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
    in0_block_w=14,
    out_subblock_h=2,
    out_subblock_w=1,
    per_core_M=2,
    per_core_N=2,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)

# wkv_b2: batch 32, [20, 16] * [16, 4]. 64 output blocks.
prog_config_mm4_bh = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=10,
    per_core_N=4,
)

# o_proj: [20, 128] * [128, 224]
prog_config_mm5_bh = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=7,
    per_core_M=2,
    per_core_N=21,
    transpose_mcast=False,
    fuse_batch=False,
    fused_activation=None,
)


# Mesh configuration: (sp_axis=0, tp_axis=1)
# SP (Sequence Parallelism) on axis 0, TP (Tensor Parallelism) on axis 1
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    ids=["8x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [torus_xy_device_params()],
    ids=["torus-xy"],
    indirect=True,
)
@pytest.mark.parametrize(
    "in0_x, in0_y, in0_z, in0_w, in0_sp_sharded, in0_tp_sharded, in0_tp_shard_dim, in0_dtype, in1_x, in1_y, in1_z, in1_w, in1_tp_sharded, in1_tp_shard_dim, in1_dtype, out_dtype, prog_config, act_mem_config, out_mem_config",
    [
        # mm0 -- q_a_proj
        (
            1,
            1,
            SEQ_LEN,
            HIDDEN_SIZE,
            True,
            True,
            3,
            ttnn.bfloat16,
            1,
            1,
            HIDDEN_SIZE,
            1536,
            True,
            2,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            prog_config_mm0_bh,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
        # mm1 -- q_b_proj
        (
            1,
            1,
            SEQ_LEN,
            1536,
            True,
            False,
            None,
            ttnn.bfloat16,
            1,
            1,
            1536,
            24576,
            True,
            3,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            prog_config_mm1_bh,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
        # mm2 -- wkv_b1
        (
            1,
            NUM_HEADS,
            SEQ_LEN,
            128,
            True,
            True,
            1,
            ttnn.bfloat16,
            1,
            NUM_HEADS,
            128,
            512,
            True,
            1,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            prog_config_mm2_bh,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.L1_MEMORY_CONFIG,
        ),
        # mm3 -- kv_a_proj_with_mqa
        (
            1,
            1,
            SEQ_LEN,
            HIDDEN_SIZE,
            True,
            True,
            3,
            ttnn.bfloat16,
            1,
            1,
            HIDDEN_SIZE,
            576,
            True,
            2,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            prog_config_mm3_bh,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        # mm4 -- wkv_b2
        (
            1,
            NUM_HEADS,
            SEQ_LEN,
            512,
            True,
            True,
            1,
            ttnn.bfloat16,
            1,
            NUM_HEADS,
            512,
            128,
            True,
            1,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            prog_config_mm4_bh,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        # mm5 -- o_proj
        (
            1,
            1,
            SEQ_LEN,
            16384,
            True,
            True,
            3,
            ttnn.bfloat16,
            1,
            1,
            16384,
            7168,
            True,
            2,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            prog_config_mm5_bh,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ],
)
@pytest.mark.parametrize(
    "skip_host_comparison",
    [False],
)
def test_mla_mm(
    mesh_device,
    in0_x,
    in0_y,
    in0_z,
    in0_w,
    in0_sp_sharded,
    in0_tp_sharded,
    in0_tp_shard_dim,
    in0_dtype,
    in1_x,
    in1_y,
    in1_z,
    in1_w,
    in1_tp_sharded,
    in1_tp_shard_dim,
    in1_dtype,
    out_dtype,
    prog_config,
    act_mem_config,
    out_mem_config,
    skip_host_comparison,
):
    torch.manual_seed(42)
    hidden_states = torch.randn(in0_x, in0_y, in0_z, in0_w, dtype=torch.bfloat16)
    weight = torch.randn(in1_x, in1_y, in1_z, in1_w, dtype=torch.bfloat16) * 0.02

    print("Compute grid size is: ", mesh_device.compute_with_storage_grid_size())

    sp_axis = 0
    tp_axis = 1

    # Convert input to TTNN with sharding over (seq_len, hidden_size)
    in0_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    if in0_sp_sharded or in0_tp_sharded:
        shard_dims = [None, None]
        if in0_sp_sharded:
            shard_dims[sp_axis] = 2
        if in0_tp_sharded:
            assert in0_tp_shard_dim is not None
            shard_dims[tp_axis] = in0_tp_shard_dim
        in0_mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=shard_dims,
        )
    tt_input = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=act_mem_config,
        mesh_mapper=in0_mesh_mapper,
    )

    in1_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    if in1_tp_sharded:
        shard_dims = [None, None]
        shard_dims[tp_axis] = in1_tp_shard_dim
        in1_mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=shard_dims,
        )
    tt_weight = ttnn.from_torch(
        weight,
        device=mesh_device,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=in1_mesh_mapper,
    )

    # Compute kernel config as in mla.py
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    print("tt_input.shape: ", tt_input.shape)
    print("tt_weight.shape: ", tt_weight.shape)

    # Perform matmul
    tt_output = ttnn.linear(
        tt_input,
        tt_weight,
        memory_config=out_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=out_dtype,
        program_config=prog_config,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Successfully completed matmul test, output shape: {tt_output.shape}")

    if skip_host_comparison == False:
        # Compute reference output on host
        # Direct matmul: output = input @ weight
        # For the matmul shapes: (in0_x, in0_y, in0_z, in0_w) @ (in1_x, in1_y, in1_z, in1_w)
        # Result: (in0_x, in0_y, in0_z, in1_w)
        reference_output = torch.matmul(hidden_states, weight)

        # Convert TT output back to torch
        # Debug: print output tensor info
        logger.info(f"tt_output shape: {tt_output.shape}")

        # Determine concat dimensions based on sharding configuration
        # Output inherits sharding from input's non-contracted dimensions
        # For matmul (in0 @ in1), output shape is (in0_x, in0_y, in0_z, in1_w)
        # - Dim 0-2 from in0: inherit in0's sharding on these dims
        # - Dim 3 from in1: if in1 is sharded on output dim (dim3), output is sharded there
        concat_dims = [None, None]

        # sp_axis sharding: input is always sharded on dim2 (seq_len) if sp_sharded
        if in0_sp_sharded:
            concat_dims[sp_axis] = 2

        # tp_axis sharding for output depends on both operands
        # Case 1: Both sharded on contraction dim → need to sum partial results
        # Case 2: Input sharded on non-contraction dim → output inherits this sharding
        # Case 3: Weight sharded on output dim → output is sharded on output dim
        need_tp_sum = (
            in0_tp_sharded
            and in0_tp_shard_dim == 3
            and in1_tp_sharded  # input sharded on contraction dim
            and in1_tp_shard_dim == 2  # weight sharded on contraction dim
        )

        if in0_tp_sharded and in0_tp_shard_dim != 3:
            # Input sharded on dim1 or dim2 (not contraction) → output inherits this
            concat_dims[tp_axis] = in0_tp_shard_dim
        elif in1_tp_sharded and in1_tp_shard_dim == 3:
            # Weight sharded on output features → output sharded on dim3
            concat_dims[tp_axis] = 3
        elif need_tp_sum:
            # Both sharded on contraction → concat on dim3 then sum
            concat_dims[tp_axis] = 3

        # After matmul, concat on determined dimensions
        tt_output_torch_full = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=mesh_device.shape),
        )

        if need_tp_sum:
            # Sum the partial results from tp_axis devices
            tp_mesh_size = mesh_device.shape[tp_axis]
            tt_output_torch = tt_output_torch_full.reshape(in0_x, in0_y, in0_z, tp_mesh_size, in1_w).sum(dim=3)
        else:
            tt_output_torch = tt_output_torch_full

        # Compare outputs
        logger.info(f"Comparing outputs: TTNN shape={tt_output_torch.shape}, Reference shape={reference_output.shape}")
        passing, pcc = comp_pcc(reference_output, tt_output_torch, PCC_REQUIRED)
        logger.info(f"PCC: {pcc:.6f}, Required: {PCC_REQUIRED}")

        assert (
            passing
        ), f"MLA matmul test failed: PCC {pcc:.6f} < {PCC_REQUIRED} for shapes in0=[{in0_x}, {in0_y}, {in0_z}, {in0_w}], in1=[{in1_x}, {in1_y}, {in1_z}, {in1_w}]"

        logger.info(f"✓ MLA matmul test passed with PCC={pcc:.6f}")
