# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal repro: MoE dispatch (edge-column vs edge-row cores) back-to-back with ring-joint SDPA.

Loops two ops on a 2x4 FABRIC_2D mesh: the prefill dispatch op with its sender+untilize cores
forced onto a single worker edge line (first_col / first_row via core_grid_override), immediately
followed by ring-joint SDPA. No correctness check — this isolates a suspected dispatch<->ring-SDPA
hang, so it only measures hang vs no-hang. first_row is the control.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_gate_outputs,
    get_max_payload_size,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.tt_dit.utils.padding import get_padded_vision_seq_len
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

# Dispatch shape (matches test_prefill_dispatch_subdevices.py, the config that reproduces the
# column-dispatch hang at 5K+ ISL).
NUM_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 4
DISPATCH_BUFFER_CAPACITY_FACTOR = 4
EMB_DIM = DeepSeekV3Config.EMB_SIZE

# SDPA shape (MLA prefill, scaled from the 32x4 production config as in test_ring_joint_mla.py).
PRODUCTION_UP = 4
NHQ_V_PRODUCTION = 128
NHK = 1
HEAD_DIM_Q_K = 576
HEAD_DIM_V = 128
Q_CHUNK_SIZE = 32
K_CHUNK_SIZE = 32
Q_DTYPE = ttnn.bfloat16
KV_DTYPE = ttnn.bfloat8_b


def _dispatch_edge_core_range_set(grid_x, grid_y, edge):
    if edge == "first_col":
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, grid_y - 1))})
    if edge == "first_row":
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 0))})
    raise ValueError(f"unknown dispatch_edge: {edge}")


def run_dispatch_sdpa_loop(mesh_device, dispatch_edge, n_iters, seq_len_per_chip):
    torch.manual_seed(42)

    rp_axis, up_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    rp_factor = mesh_shape[rp_axis]
    up_factor = mesh_shape[up_axis]

    grid = mesh_device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y

    # ---- Ring-joint SDPA setup (verbatim from run_ring_joint_sdpa) ----
    nhq_v = (NHQ_V_PRODUCTION // PRODUCTION_UP) * up_factor
    base_seq_len = seq_len_per_chip * rp_factor
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, rp_factor)
    joint_seq_len = 0

    sdpa_compute_grid = (grid_x - 1, grid_y)
    ccl_core_grid_offset = (grid_x - 1, 0)

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # One semaphore pair, reused every iteration to mirror the model reusing the ring-attention
    # semaphores across all layers. Fresh per-iteration handles would mask any cross-layer state.
    ccl_semaphore_handles = [
        [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)] for _ in range(1)
    ]

    ag_output_shape_k = (1, NHK, padded_seq_len, HEAD_DIM_Q_K)
    ag_output_shape_v = (1, nhq_v, padded_seq_len, HEAD_DIM_V)

    kv_shard_dims = [None, None]
    kv_shard_dims[up_axis] = 1
    persistent_k_output_shard_dims = [None, None]

    persistent_output_buffers = [
        [
            ttnn.as_tensor(
                torch.zeros(ag_output_shape_k),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=KV_DTYPE,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_output_shard_dims
                ),
            ),
            ttnn.as_tensor(
                torch.zeros(ag_output_shape_v),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=KV_DTYPE,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims
                ),
            ),
        ]
        for _ in range(1)
    ]

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=Q_CHUNK_SIZE,
        k_chunk_size=K_CHUNK_SIZE,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(1, nhq_v, base_seq_len, HEAD_DIM_Q_K)
    K = fa_rand(1, NHK, base_seq_len, HEAD_DIM_Q_K)
    V = fa_rand(1, nhq_v, base_seq_len, HEAD_DIM_V)
    padded_Q = torch.cat([Q, torch.zeros(1, nhq_v, padded_seq_len - base_seq_len, HEAD_DIM_Q_K)], dim=2)
    padded_K = torch.cat([K, torch.zeros(1, NHK, padded_seq_len - base_seq_len, HEAD_DIM_Q_K)], dim=2)
    padded_V = torch.cat([V, torch.zeros(1, nhq_v, padded_seq_len - base_seq_len, HEAD_DIM_V)], dim=2)

    joint_Q = fa_rand(1, nhq_v, joint_seq_len, HEAD_DIM_Q_K)
    joint_K = fa_rand(1, NHK, joint_seq_len, HEAD_DIM_Q_K)
    joint_V = fa_rand(1, nhq_v, joint_seq_len, HEAD_DIM_V)

    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[rp_axis] = 2
    sdpa_input_shard_dims[up_axis] = 1

    sdpa_joint_shard_dims = [None, None]
    sdpa_joint_shard_dims[up_axis] = 1

    sdpa_k_input_shard_dims = [None, None]
    sdpa_k_input_shard_dims[rp_axis] = 2
    sdpa_k_input_shard_dims[up_axis] = None if NHK == 1 else 1

    tt_Q = ttnn.as_tensor(
        padded_Q,
        dtype=Q_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
        ),
    )
    tt_K = ttnn.as_tensor(
        padded_K,
        dtype=KV_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_input_shard_dims
        ),
    )
    tt_V = ttnn.as_tensor(
        padded_V,
        dtype=KV_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
        ),
    )
    tt_joint_Q = ttnn.from_torch(
        joint_Q,
        dtype=Q_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
        ),
    )
    joint_k_mesh_mapper = (
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_input_shard_dims)
        if NHK > 1
        else ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_joint_K = ttnn.from_torch(
        joint_K,
        dtype=KV_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=joint_k_mesh_mapper,
    )
    tt_joint_V = ttnn.from_torch(
        joint_V,
        dtype=KV_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
        ),
    )

    # ---- Dispatch setup (verbatim from run_dispatch), cores forced onto the edge line ----
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    num_devices = mesh_device.get_num_devices()

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        NUM_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        num_devices,
        dispatch_group_size,
        DISPATCH_BUFFER_CAPACITY_FACTOR,
    )

    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=EMB_DIM,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=NUM_ROUTED_EXPERTS,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    expert_offsets, _, _, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        NUM_ROUTED_EXPERTS,
        experts_per_chip,
        seq_len_per_chip,
        NUM_EXPERTS_PER_TOK,
        expert_dispatch_table=expert_dispatch_table,
    )

    dispatch_mesh_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    tt_x = ttnn.from_torch(
        x, mesh_mapper=dispatch_mesh_mapper, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=dispatch_mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=dispatch_mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint16
    )
    tt_expert_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    edge_core_range_set = _dispatch_edge_core_range_set(grid_x, grid_y, dispatch_edge)
    logger.info(f"dispatch_edge={dispatch_edge} core_grid_override={edge_core_range_set}")

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=EMB_DIM,
        cluster_axis=sp_axis,
        num_links=1,
        topology=ttnn.Topology.Linear,
        core_grid_override=edge_core_range_set,
    )

    # ---- The loop under test: dispatch then ring-joint SDPA, per iter ----
    try:
        for i in range(n_iters):
            logger.info(f"[iter {i}/{n_iters}] ring_joint_sdpa")
            ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_output_buffers[0][0],
                persistent_output_buffer_v=persistent_output_buffers[0][1],
                joint_strategy="rear",
                logical_n=base_seq_len,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_semaphore_handles[0],
                num_links=1,
                cluster_axis=rp_axis,
                mesh_device=mesh_device,
                topology=ttnn.Topology.Linear,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=ccl_core_grid_offset,
                use_column_major_ccl=True,
                is_causal=True,
                is_balanced=False,
            )

            ttnn.synchronize_device(mesh_device)
            logger.info(f"[iter {i}/{n_iters}] synchronized after sdpa")

            logger.info(f"[iter {i}/{n_iters}] dispatch ({dispatch_edge})")
            tt_dispatch_module(tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table)

            ttnn.synchronize_device(mesh_device)
            logger.info(f"[iter {i}/{n_iters}] synchronized")
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(sub_device_manager)

    logger.info(f"✅ completed {n_iters} dispatch+sdpa iterations (dispatch_edge={dispatch_edge})")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        (
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
            },
        ),
    ],
    indirect=["mesh_device", "device_params"],
    ids=["fabric2d-mesh-2x4"],
)
@pytest.mark.parametrize("dispatch_edge", ["first_col", "first_row"])
@pytest.mark.parametrize("seq_len_per_chip", [128, 2560], ids=["seq128", "seq2560"])
@pytest.mark.parametrize("n_iters", [2, 20], ids=["iters2", "iters20"])
@pytest.mark.timeout(0)
def test_dispatch_sdpa_loop(mesh_device, device_params, dispatch_edge, seq_len_per_chip, n_iters):
    run_dispatch_sdpa_loop(mesh_device, dispatch_edge, n_iters, seq_len_per_chip)
