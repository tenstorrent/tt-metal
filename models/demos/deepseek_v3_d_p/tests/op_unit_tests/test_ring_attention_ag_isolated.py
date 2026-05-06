# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Isolated unit test for the ring-attention all-gather op.

ring_joint_scaled_dot_product_attention internally invokes
ttnn.experimental.ring_attention_all_gather_async to gather K and V across the
SP-axis ring before the attention compute. This test calls that AG op directly
with the same shapes/sharding/dtypes/mesh/fabric/topology that MLA's SDPA would
feed it, but without the SDPA compute. If a hang reproduces here, the AG path
itself is implicated; if not, the bug lives in SDPA's compute path or in the
AG↔compute interaction inside ring_joint_scaled_dot_product_attention.

Shape derivation matches `test_mla_sdpa` in test_ring_joint_mla.py: production
shape (32, 4) is scaled to whatever mesh the test runs on by the same formula.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.tt_dit.utils.padding import get_padded_vision_seq_len


def create_global_semaphores(mesh_device, cores, initial_value):
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def create_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor):
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    return mesh_device.create_submesh(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))


def run_isolated_ag(
    submesh,
    *,
    b,
    num_heads_v,
    num_heads_k,
    padded_seq_len,
    head_dim_k,
    head_dim_v,
    kv_dtype,
    num_links,
    rp_axis,
    up_axis,
    all_gather_topology,
    n_iters,
):
    """Drive ring_attention_all_gather_async with K- and V-shaped inputs that
    match what ring_joint_scaled_dot_product_attention feeds it inside MLA."""

    full_compute_grid = submesh.compute_with_storage_grid_size()
    logger.debug(f"full_compute_grid: {full_compute_grid}")
    logger.debug(f"submesh shape: {submesh.shape}")

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # Per-iteration global semaphores (mirrors what the SDPA test does so the AG
    # op gets fresh handles each iteration, identical to one MLA forward pass).
    ccl_semaphore_handles = [create_global_semaphores(submesh, ccl_sub_device_crs, 0) for _ in range(n_iters)]

    # Persistent output buffers: full padded_seq_len gathered across rp_axis.
    # K is replicated across up_axis when num_heads_k == 1 (MLA absorbed-K shape);
    # V is sharded across up_axis on the heads dim.
    ag_output_shape_k = (b, num_heads_k, padded_seq_len, head_dim_k)
    ag_output_shape_v = (b, num_heads_v, padded_seq_len, head_dim_v)

    persistent_k_dims = [None, None]
    if num_heads_k != 1:
        persistent_k_dims[up_axis] = 1
    persistent_v_dims = [None, None]
    persistent_v_dims[up_axis] = 1

    persistent_output_buffers = [
        [
            ttnn.from_torch(
                torch.zeros(ag_output_shape_k),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=kv_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=persistent_k_dims),
            ),
            ttnn.from_torch(
                torch.zeros(ag_output_shape_v),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=kv_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=persistent_v_dims),
            ),
        ]
        for _ in range(n_iters)
    ]

    # Inputs sharded on rp_axis (sequence dim 2). K is replicated across up_axis
    # when num_heads_k == 1; otherwise sharded on heads. V is always sharded on heads.
    k_input_dims = [None, None]
    k_input_dims[rp_axis] = 2
    if num_heads_k != 1:
        k_input_dims[up_axis] = 1

    v_input_dims = [None, None]
    v_input_dims[rp_axis] = 2
    v_input_dims[up_axis] = 1

    torch.manual_seed(0)
    K_full = torch.randn(b, num_heads_k, padded_seq_len, head_dim_k, dtype=torch.bfloat16)
    V_full = torch.randn(b, num_heads_v, padded_seq_len, head_dim_v, dtype=torch.bfloat16)

    tt_K = ttnn.from_torch(
        K_full,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=kv_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=k_input_dims),
    )
    tt_V = ttnn.from_torch(
        V_full,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=kv_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=v_input_dims),
    )

    logger.info(
        f"AG inputs ready: K={tuple(tt_K.shape)} V={tuple(tt_V.shape)} "
        f"persistent_K={ag_output_shape_k} persistent_V={ag_output_shape_v}"
    )

    for i in range(n_iters):
        logger.info(f"--- AG iter {i + 1}/{n_iters} ---")
        ttnn.experimental.ring_attention_all_gather_async(
            [tt_K, tt_V],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=2,  # gather along sequence
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            cluster_axis=rp_axis,
            mesh_device=submesh,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
        )
        ttnn.synchronize_device(submesh, sub_device_ids=sub_device_stall_group)

    logger.info(f"✓ AG isolated loop completed: {n_iters} iterations")
    ttnn.distributed_context_barrier()


# Production shape (32, 4): SP=32, TP=4. Same scaling formula as test_mla_sdpa.
# Inputs (b, num_heads_v, num_heads_k, head_dim_q_k, head_dim_v) match MLA's
# absorbed-K shape: K has 1 head dim head_dim=576 (kv_lora_rank+rope), V has
# 128 heads with head_dim=128.
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat8_b], ids=["kv_bf8"])
@pytest.mark.parametrize(
    "seq_len",
    [128 * 1024, 100 * 1024],
    ids=["seq128k", "seq100k"],
)
@pytest.mark.parametrize(
    "b, num_heads_v, num_heads_k, head_dim_k, head_dim_v",
    [(1, 128, 1, 576, 128)],
)
@pytest.mark.parametrize("n_iters", [1, 25, 50], ids=["iter1", "iter25", "iter50"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
            },
            ttnn.Topology.Linear,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            ttnn.Topology.Linear,
        ),
    ],
    ids=["line_original", "line_emb"],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4), (2, 4)],
    ids=["8x4", "2x4"],
    indirect=True,
)
@pytest.mark.parametrize("rp_axis, up_axis", [[0, 1]], ids=["rpxup"])
@pytest.mark.timeout(0)
def test_ring_attention_ag_isolated(
    mesh_device,
    b,
    num_heads_v,
    num_heads_k,
    head_dim_k,
    head_dim_v,
    seq_len,
    kv_dtype,
    n_iters,
    num_links,
    rp_axis,
    up_axis,
    all_gather_topology,
):
    production_shape = [32, 4]
    mesh_device_shape = list(mesh_device.shape)

    rp_factor = mesh_device_shape[rp_axis]
    up_factor = mesh_device_shape[up_axis]

    submesh = create_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    # Same scaling as test_mla_sdpa: scale seq_len by mesh / production rp_factor,
    # scale heads by mesh / production up_factor.
    seq_len = (seq_len // production_shape[0]) * rp_factor
    num_heads_v = (num_heads_v // production_shape[1]) * up_factor
    padded_seq_len = get_padded_vision_seq_len(seq_len, mesh_device_shape[rp_axis])

    logger.info(
        f"AG isolated: mesh={tuple(mesh_device.shape)} submesh={tuple(submesh.shape)} "
        f"rp={rp_factor} up={up_factor} seq_len={seq_len} padded={padded_seq_len} "
        f"num_heads_v={num_heads_v} n_iters={n_iters}"
    )

    run_isolated_ag(
        submesh,
        b=b,
        num_heads_v=num_heads_v,
        num_heads_k=num_heads_k,
        padded_seq_len=padded_seq_len,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        kv_dtype=kv_dtype,
        num_links=num_links,
        rp_axis=rp_axis,
        up_axis=up_axis,
        all_gather_topology=all_gather_topology,
        n_iters=n_iters,
    )
