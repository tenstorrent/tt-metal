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


SEQ_LEN = 128 * 1024
HIDDEN_SIZE = 7168
NUM_HEADS = 128
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
@pytest.mark.parametrize("in0_x, in0_y, in0_z, in0_w, in0_sp_sharded, in0_tp_sharded, in0_tp_shard_dim, in0_dtype, in1_x, in1_y, in1_z, in1_w, in1_tp_sharded, in1_tp_shard_dim, in1_dtype, out_dtype",
    [
        (1, 1, SEQ_LEN, HIDDEN_SIZE, True, True, 3, ttnn.bfloat16, 1, 1, HIDDEN_SIZE, 1536, True, 2, ttnn.bfloat8_b, ttnn.bfloat16),
        (1, 1, SEQ_LEN, 1536, True, False, None, ttnn.bfloat16, 1, 1, 1536, 24576, True, 3, ttnn.bfloat8_b, ttnn.bfloat16),
        (1, NUM_HEADS, SEQ_LEN, 128, True, True, 1, ttnn.bfloat16, 1, NUM_HEADS, 128, 512, True, 1, ttnn.bfloat8_b, ttnn.bfloat16),
        (1, 1, SEQ_LEN, HIDDEN_SIZE, True, True, 3, ttnn.bfloat16, 1, 1, HIDDEN_SIZE, 576, True, 2, ttnn.bfloat8_b, ttnn.bfloat16),
        (1, NUM_HEADS, SEQ_LEN, 512, True, True, 1, ttnn.bfloat16, 1, NUM_HEADS, 512, 128, True, 2, ttnn.bfloat8_b, ttnn.bfloat8_b),
        (1, 1, SEQ_LEN, 32768, True, True, 3, ttnn.bfloat16, 1, 1, 32768, 7168, True, 2, ttnn.bfloat8_b, ttnn.bfloat16),
    ]
)
def test_mla_mm(request, mesh_device, in0_x, in0_y, in0_z, in0_w, in0_sp_sharded, in0_tp_sharded, in0_tp_shard_dim, in0_dtype, in1_x, in1_y, in1_z, in1_w, in1_tp_sharded, in1_tp_shard_dim, in1_dtype, out_dtype):
    torch.manual_seed(42)
    hidden_states = torch.randn(in0_x, in0_y, in0_z, in0_w, dtype=torch.bfloat16)
    weight = torch.randn(in1_x, in1_y, in1_z, in1_w, dtype=torch.bfloat16) * 0.02

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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    # Perform matmul
    print("tt_input.shape: ", tt_input.shape)
    print("tt_weight.shape: ", tt_weight.shape)
    tt_output = ttnn.linear(
        tt_input,
        tt_weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        dtype=out_dtype,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Successfully completed matmul test, output shape: {tt_output.shape}")