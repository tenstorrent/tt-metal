# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Stress test for the MLA q-projection block: linear -> all-reduce -> rms_norm -> linear.

Loops the block n_iterations times on a 2x4 mesh (SP=2, TP=4) with random weights
to verify CCL semaphore cycling and numerical stability across many iterations.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl


@pytest.mark.timeout(0)
@pytest.mark.parametrize("n_iterations", [1, 2500000], ids=["1iter", "250iter"])
@pytest.mark.parametrize("seq_len_local", [6400])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE * 2),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_mla_q_proj_block(mesh_device, device_params, num_links, topology, n_iterations, seq_len_local):
    """
    Stress-test the MLA q-projection block:
      1. linear (q_a_proj)           - TP-partial matmul
      2. reduce_scatter + all_gather - all-reduce across TP
      3. rms_norm                    - normalise latent
      4. linear (q_b_proj)          - project to full head dims
    """
    torch.manual_seed(42)

    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]

    ccl_num_links = num_links
    ccl_topology = topology

    # CCL semaphore manager
    tt_ccl = get_tt_ccl(mesh_device)

    # Compute kernel config (matches mla.py)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # --- Mesh mappers ---
    shard_dims_tp0 = [None, None]
    shard_dims_tp0[tp_axis] = 0
    mapper_tp0 = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims_tp0)

    shard_dims_tp1 = [None, None]
    shard_dims_tp1[tp_axis] = 1
    mapper_tp1 = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims_tp1)

    # --- Create random weights ---
    # q_a_proj: [hidden_size, q_lora_rank], TP-sharded on dim 0
    tt_q_a_proj_weight = ttnn.from_torch(
        torch.randn(DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.Q_LORA_RANK, dtype=torch.bfloat16) * 0.02,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper_tp0,
    )

    # q_a_layernorm: [1, 1, q_lora_rank // TILE_SIZE, TILE_SIZE], replicated
    tt_q_a_layernorm_weight = ttnn.from_torch(
        torch.randn(1, 1, DeepSeekV3Config.Q_LORA_RANK // ttnn.TILE_SIZE, ttnn.TILE_SIZE, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # q_b_proj: [q_lora_rank, num_heads * qk_head_dim], TP-sharded on dim 1
    tt_q_b_proj_weight = ttnn.from_torch(
        torch.randn(
            DeepSeekV3Config.Q_LORA_RANK,
            DeepSeekV3Config.NUM_ATTENTION_HEADS
            * (DeepSeekV3Config.QK_NOPE_HEAD_DIM + DeepSeekV3Config.QK_ROPE_HEAD_DIM),
            dtype=torch.bfloat16,
        )
        * 0.02,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper_tp1,
    )

    # --- Create random input ---
    # hidden_states: [1, 1, sp_factor * seq_len_local, hidden_size]
    # Sharded: SP on dim 2, TP on dim 3
    shard_dims_input = [None, None]
    shard_dims_input[sp_axis] = 2
    shard_dims_input[tp_axis] = 3
    mapper_input = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims_input)

    tt_hidden_states = ttnn.from_torch(
        torch.randn(1, 1, sp_factor * seq_len_local, DeepSeekV3Config.EMB_SIZE, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper_input,
    )

    logger.info(
        f"Running q-projection block for {n_iterations} iterations on {mesh_device.shape} mesh, "
        f"seq_len_local={seq_len_local}"
    )

    # --- Run iterations ---
    for i in range(n_iterations):
        # 1. q_a_proj linear (TP-partial)
        tt_q = ttnn.linear(
            tt_hidden_states,
            tt_q_a_proj_weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        # 2. All-reduce: reduce_scatter + all_gather
        if tp_factor > 1:
            tt_q = ttnn.experimental.reduce_scatter_minimal_async(
                tt_q,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=tp_axis),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=tp_axis),
                num_links=ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl_topology,
                cluster_axis=tp_axis,
            )
            tt_q = ttnn.experimental.all_gather_async(
                tt_q,
                dim=3,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=tp_axis),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=tp_axis),
                num_links=ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ccl_topology,
                cluster_axis=tp_axis,
            )

        # 3. RMS norm
        tt_q = ttnn.rms_norm(
            tt_q,
            weight=tt_q_a_layernorm_weight,
            epsilon=DeepSeekV3Config.RMS_NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )

        # 4. q_b_proj linear
        tt_q = ttnn.linear(
            tt_q,
            tt_q_b_proj_weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        ttnn.synchronize_device(mesh_device)

        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"Iteration {i + 1}/{n_iterations} completed, output shape: {tt_q.shape}")

    # --- Sanity checks on final output ---
    expected_last_dim = (
        DeepSeekV3Config.NUM_ATTENTION_HEADS
        * (DeepSeekV3Config.QK_NOPE_HEAD_DIM + DeepSeekV3Config.QK_ROPE_HEAD_DIM)
        // tp_factor
    )
    assert tt_q.shape == [
        1,
        1,
        seq_len_local,
        expected_last_dim,
    ], f"Unexpected output shape: {tt_q.shape}, expected [1, 1, {seq_len_local}, {expected_last_dim}]"

    logger.info(f"All {n_iterations} iterations completed successfully, output shape: {tt_q.shape}")
