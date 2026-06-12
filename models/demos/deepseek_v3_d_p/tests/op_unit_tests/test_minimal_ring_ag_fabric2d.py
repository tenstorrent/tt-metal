# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal smoke: ttnn.experimental.ring_attention_all_gather_async on (8,4) Blackhole
Galaxy, cluster_axis=0 (SP, 8-long axis), across three fabric configs. Strips away
MLA/SDPA/FFN — just creates K/V-shaped tensors, runs the AG, and asserts it completes
(no hang under the @timeout) with the expected gathered shapes.

Fabric configs exercised:
  - FABRIC_2D       : Topology.Linear. The original hang repro — a hang here surfaces at
                      synchronize_device with the e0,8/e0,9 ethernet-retrain signature.
  - FABRIC_1D_RING  : Topology.Ring (1D ring on the SP axis).
  - FABRIC_2D_TORUS_Y: Topology.Ring over a 2D-routed torus — the single-galaxy column
                      ring case.

FABRIC_2D_TORUS_Y closes the 8-axis (mesh dim 0 = the SP/row axis; TORUS_Y wraps dim 0,
TORUS_X would wrap the 4-wide dim 1) into a ring on one galaxy. FABRIC_2D_TORUS_Y
auto-selects single_bh_galaxy_torus_y_graph_descriptor.textproto, whose uniform
`channels { count: 2 }` gives the RING-closing row-7<->row-0 edge the same 2-link width
as the LINE edges (validated against the 110-c78 wiring: that wrap is a normal 2-link on
every column; the extra row-2<->row-5 chord and chan-8/9 cables are left undeclared).
The op requests Topology.Ring on cluster_axis=0; get_usable_topology (ccl_common.cpp)
retains Ring because the device coords span the full 8-axis (first==0, last==7 ->
BoundaryMode::WRAP). Ring-retention is guaranteed by that code path but is NOT asserted
at runtime (no python binding for get_usable_topology). The 4-axis (cluster_axis=1) stays
LINE, so column separation is preserved.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size

# DeepSeek-V3 dims (mirrors MLA shapes at isl_total=1024 on (8,4) mesh, tp_factor=4)
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
NUM_HEADS = 128
ISL_TOTAL = 1024


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="FABRIC_2D",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="FABRIC_1D_RING",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="FABRIC_2D_TORUS_Y",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["1link", "2link"])
@pytest.mark.timeout(180)
def test_minimal_ring_ag(mesh_device, device_params, num_links):
    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]
    isl_per_chip = ISL_TOTAL // sp_factor
    num_heads_per_tp = NUM_HEADS // tp_factor

    # Ring-capable fabrics gather the 8-long SP axis as a ring; plain FABRIC_2D stays
    # Linear. For FABRIC_2D_TORUS_Y, get_usable_topology keeps Topology.Ring on the
    # SP axis (coords span the full 8-axis -> WRAP); the 4-axis stays Linear.
    ring_fabrics = (
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    )
    topology = ttnn.Topology.Ring if device_params["fabric_config"] in ring_fabrics else ttnn.Topology.Linear

    logger.info(
        f"sp_factor={sp_factor}, tp_factor={tp_factor}, "
        f"isl_per_chip={isl_per_chip}, num_heads_per_tp={num_heads_per_tp}, "
        f"num_links={num_links}, topology={topology}"
    )

    # Sub-device + semaphores (matches MLA's tt_ccl pattern)
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    ccl_sem_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    # K full output shape: [1, 1, ISL_TOTAL, KV_LORA_RANK + QK_ROPE_HEAD_DIM]
    # V full output shape: [1, num_heads_per_tp, ISL_TOTAL, V_HEAD_DIM]
    k_output_shape = (1, 1, ISL_TOTAL, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    v_output_shape = (1, num_heads_per_tp, ISL_TOTAL, V_HEAD_DIM)

    logger.info(f"K output shape: {k_output_shape}")
    logger.info(f"V output shape: {v_output_shape}")

    # Input shard: along SP axis (cluster_axis=0). V also sharded along TP for the head dim.
    k_input_dims = [None, None]
    k_input_dims[sp_axis] = 2  # shard ISL along SP
    v_input_dims = [None, None]
    v_input_dims[sp_axis] = 2  # shard ISL along SP
    v_input_dims[tp_axis] = 1  # shard heads along TP

    torch.manual_seed(0)
    k_full = torch.randn(k_output_shape, dtype=torch.bfloat16)
    v_full = torch.randn((1, NUM_HEADS, ISL_TOTAL, V_HEAD_DIM), dtype=torch.bfloat16)

    k_input = ttnn.from_torch(
        k_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=k_input_dims),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v_input = ttnn.from_torch(
        v_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=v_input_dims),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Persistent buffers (match MLA's allocation pattern in mla.py:304-313)
    persistent_v_dims = [None, None]
    persistent_v_dims[tp_axis] = 1  # shard heads along TP
    persistent_k_dims = [None, None]

    persistent_k = ttnn.from_torch(
        torch.zeros(k_output_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_dims),
    )
    persistent_v = ttnn.from_torch(
        torch.zeros((1, NUM_HEADS, ISL_TOTAL, V_HEAD_DIM)),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_v_dims),
    )

    logger.info("Calling ring_attention_all_gather_async on cluster_axis=0 (SP)...")
    out_tensors = ttnn.experimental.ring_attention_all_gather_async(
        [k_input, v_input],
        persistent_output_buffer=[persistent_k, persistent_v],
        dim=2,
        multi_device_global_semaphore=ccl_sem_handles,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=worker_sub_device_id,
    )
    logger.info(
        f"Returned from AG; out_tensors[0].shape={out_tensors[0].shape}, out_tensors[1].shape={out_tensors[1].shape}"
    )

    logger.info("Calling synchronize_device...")
    ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
    logger.success("AG completed under this fabric config — no hang.")

    # Structural check: the gather expands each per-chip SP shard back to the full
    # ISL_TOTAL on every chip. Catches a degenerate "ran but gathered nothing/wrong"
    # without needing a PCC reference (numerical PCC lives in the dispatch/combine suite).
    assert tuple(out_tensors[0].shape) == k_output_shape, f"K gather shape {out_tensors[0].shape} != {k_output_shape}"
    assert tuple(out_tensors[1].shape) == v_output_shape, f"V gather shape {out_tensors[1].shape} != {v_output_shape}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
