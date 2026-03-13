# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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

PCC_REQUIRED = 0.99

# Available core grid is 12x10, but due to di/dt and throttling problems, use 11x10 temporarily
compute_with_storage_grid_size_bh_orig = (12, 10)

# [128, 128] * [128, 224]
compute_with_storage_grid_size_11x10 = (11, 10)
prog_config_mm5_bh = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=7,
                    per_core_M=13,
                    per_core_N=21,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,)

# [100, 128] * [128, 224]
prog_config_mm5_bh_25k = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=7,
                    per_core_M=10,
                    per_core_N=21,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,)

# [128, 56] * [56, 18]
prog_config_mm3_bh = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=13,
                    per_core_N=2,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,)

# [100, 56] * [56, 18]
prog_config_mm3_bh_25k = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
                    in0_block_w=8,
                    out_subblock_h=2,
                    out_subblock_w=2,
                    per_core_M=10,
                    per_core_N=2,
                    transpose_mcast=False,
                    fuse_batch=False,
                    fused_activation=None,)

# [32, 128, 16] * [32, 16, 4]
prog_config_mm4_bh = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
                    in0_block_w=8,
                    out_subblock_h=2,
                    out_subblock_w=4,
                    per_core_M=2,
                    per_core_N=4,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=False,)

# [32, 100, 16] * [32, 16, 4]
prog_config_mm4_bh_25k = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=compute_with_storage_grid_size_11x10,
                    in0_block_w=16,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=4,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=False,)

HIDDEN_SIZE = 7168
NUM_HEADS = 128

# Obtained by dividing 128 * 1024 over 4 galaxies
SEQ_LEN_32K = 32768

# Obtained by dividing 100 * 1024 over 4 galaxies
SEQ_LEN_25K = 25600

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
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("in0_x, in0_y, in0_z, in0_w, in0_sp_sharded, in0_tp_sharded, in0_tp_shard_dim, in0_dtype, in1_x, in1_y, in1_z, in1_w, in1_tp_sharded, in1_tp_shard_dim, in1_dtype, out_dtype, prog_config, act_mem_config, out_mem_config",
    [
        (1, 1, SEQ_LEN_32K, HIDDEN_SIZE, True, True, 3, ttnn.bfloat16, 1, 1, HIDDEN_SIZE, 1536, True, 2, ttnn.bfloat8_b, ttnn.bfloat16, None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, SEQ_LEN_32K, 1536, True, False, None, ttnn.bfloat16, 1, 1, 1536, 24576, True, 3, ttnn.bfloat8_b, ttnn.bfloat16, None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, NUM_HEADS, SEQ_LEN_32K, 128, True, True, 1, ttnn.bfloat16, 1, NUM_HEADS, 128, 512, True, 1, ttnn.bfloat8_b, ttnn.bfloat16, None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, SEQ_LEN_32K, HIDDEN_SIZE, True, True, 3, ttnn.bfloat16, 1, 1, HIDDEN_SIZE, 576, True, 2, ttnn.bfloat8_b, ttnn.bfloat16, prog_config_mm3_bh, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, NUM_HEADS, SEQ_LEN_32K, 512, True, True, 1, ttnn.bfloat16, 1, NUM_HEADS, 512, 128, True, 1, ttnn.bfloat8_b, ttnn.bfloat8_b, prog_config_mm4_bh, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, SEQ_LEN_32K, 16384, True, True, 3, ttnn.bfloat16, 1, 1, 16384, 7168, True, 2, ttnn.bfloat8_b, ttnn.bfloat16, prog_config_mm5_bh, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),

        (1, 1, SEQ_LEN_25K, HIDDEN_SIZE, True, True, 3, ttnn.bfloat16, 1, 1, HIDDEN_SIZE, 1536, True, 2, ttnn.bfloat8_b, ttnn.bfloat16, None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, SEQ_LEN_25K, 1536, True, False, None, ttnn.bfloat16, 1, 1, 1536, 24576, True, 3, ttnn.bfloat8_b, ttnn.bfloat16, None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, NUM_HEADS, SEQ_LEN_25K, 128, True, True, 1, ttnn.bfloat16, 1, NUM_HEADS, 128, 512, True, 1, ttnn.bfloat8_b, ttnn.bfloat16, None, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, SEQ_LEN_25K, HIDDEN_SIZE, True, True, 3, ttnn.bfloat16, 1, 1, HIDDEN_SIZE, 576, True, 2, ttnn.bfloat8_b, ttnn.bfloat16, prog_config_mm3_bh_25k, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, NUM_HEADS, SEQ_LEN_25K, 512, True, True, 1, ttnn.bfloat16, 1, NUM_HEADS, 512, 128, True, 1, ttnn.bfloat8_b, ttnn.bfloat8_b, prog_config_mm4_bh_25k, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (1, 1, SEQ_LEN_25K, 16384, True, True, 3, ttnn.bfloat16, 1, 1, 16384, 7168, True, 2, ttnn.bfloat8_b, ttnn.bfloat16, prog_config_mm5_bh_25k, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ]
)
@pytest.mark.parametrize(
    "skip_host_comparison",
    [False],
)
def test_mla_mm(request, mesh_device, in0_x, in0_y, in0_z, in0_w, in0_sp_sharded, in0_tp_sharded, in0_tp_shard_dim, in0_dtype, in1_x, in1_y, in1_z, in1_w, in1_tp_sharded, in1_tp_shard_dim, in1_dtype, out_dtype, prog_config, act_mem_config, out_mem_config, skip_host_comparison):
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
            in0_tp_sharded and in0_tp_shard_dim == 3 and  # input sharded on contraction dim
            in1_tp_sharded and in1_tp_shard_dim == 2       # weight sharded on contraction dim
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

        assert passing, f"MLA matmul test failed: PCC {pcc:.6f} < {PCC_REQUIRED} for shapes in0=[{in0_x}, {in0_y}, {in0_z}, {in0_w}], in1=[{in1_x}, {in1_y}, {in1_z}, {in1_w}]"

        logger.info(f"✓ MLA matmul test passed with PCC={pcc:.6f}")