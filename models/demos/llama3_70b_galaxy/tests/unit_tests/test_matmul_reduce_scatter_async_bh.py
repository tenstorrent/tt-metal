# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone Blackhole-Galaxy validation for ttnn.experimental.matmul_reduce_scatter_async.

Goal: prove the *fused* matmul + reduce-scatter (built on the BH-working reduce_scatter_minimal_async
primitive, with the matmul->RS overlap signaler) runs on the 8x4 Blackhole Galaxy 2D-torus fabric with
a column-axis (cluster_axis=1) reduce-scatter and produces correct numerics -- WITHOUT the Wormhole
llama_reduce_scatter path (which deadlocks on BH).

This isolates the op from the model's tensor-parallel FF layout: we control every shape here. A K-parallel
matmul is sharded on the contraction dim across the 4 column devices; each device computes a partial
[M, N]; the fused reduce-scatter sums the partials across the 4-device column ring and scatters N.

Run (on a Blackhole Galaxy):
    pytest -svq models/demos/llama3_70b_galaxy/tests/unit_tests/test_matmul_reduce_scatter_async_bh.py
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import DECODE_FABRIC_CONFIG as _FABRIC_CONFIG


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _FABRIC_CONFIG,
            "trace_region_size": 200000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "M, K_full, N",
    [
        (256, 2048, 2048),
    ],
)
@pytest.mark.parametrize("num_links", [1])
def test_matmul_reduce_scatter_async_bh(mesh_device, M, K_full, N, num_links, reset_seeds):
    mesh_shape = (8, 4)
    cluster_axis = 1
    num_col_devices = mesh_shape[1]  # cluster_axis=1 -> 4 devices in a column ring
    assert K_full % num_col_devices == 0
    assert N % num_col_devices == 0
    K_per_dev = K_full // num_col_devices

    # ----- sub-device (full worker grid) + semaphores -----
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"compute_with_storage_grid_size = ({grid.x}, {grid.y})")
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    rs_semaphores = [ttnn.create_global_semaphore(mesh_device, ccl_crs, 0) for _ in range(3)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, ccl_crs, 0)

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # ----- persistent buffers required by the fused RS -----
    # intermediate = matmul output per device [1,1,M,N]; output = scattered [1,1,M,N/num_col_devices]
    persistent_intermediate = ttnn.from_torch(
        torch.zeros([1, 1, M, N]),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    persistent_output = ttnn.from_torch(
        torch.zeros([1, 1, M, N // num_col_devices]),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ----- weights: [1,1,K_full,N], K sharded across the 4 column devices, replicated across rows -----
    weights_torch = torch.randn([1, 1, K_full, N]).bfloat16()
    weight_tt = ttnn.from_torch(
        weights_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(2)], ttnn.MeshShape(*mesh_shape)),
        ),
    )

    # ----- input: [1,1,M,K_full], K sharded across the 4 column devices, replicated across rows -----
    input_torch = torch.rand([1, 1, M, K_full]).bfloat16()
    input_tt = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(*mesh_shape)),
        ),
    )

    # ----- 2D multicast matmul program config (fused op requires MatmulMultiCoreReuseMultiCast) -----
    core_grid = (8, 4)
    in0_block_w = max(1, min(4, K_per_dev // 32 // core_grid[0]))
    per_core_M = max(1, math.ceil(M / 32 / core_grid[1]))
    per_core_N = max(1, math.ceil(N / 32 / core_grid[0]))
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # ----- torch reference: full K-reduced matmul, then scatter N across the column ring -----
    ref_full = torch.matmul(input_torch.float(), weights_torch.float())  # [1,1,M,N]
    ref_scatter = torch.chunk(ref_full, num_col_devices, dim=3)  # each [1,1,M,N/4]

    logger.info("Dispatching fused matmul_reduce_scatter_async (cluster_axis=1) ...")
    mm_out, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        input_tt,
        weight_tt,
        persistent_intermediate_buffer=persistent_intermediate,
        persistent_output_buffer=persistent_output,
        dim=3,
        multi_device_global_semaphore=rs_semaphores,
        reduce_scatter_core_grid_offset=(0, core_grid[1]),
        barrier_semaphore=barrier_semaphore,
        num_links=num_links,
        memory_config_rs=dram,
        memory_config_mm=dram,
        topology=ttnn.Topology.Ring,
        subdevice_id=worker_sub_device_id,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[worker_sub_device_id])
    logger.info("Fused op completed (no hang).")

    # ----- verify matmul partial-sum output -----
    mm_torch = ttnn.to_torch(
        ttnn.from_device(mm_out),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    # dims=(0,3): mesh rows -> tensor dim0 (8 identical copies), mesh cols -> tensor dim3 (the 4 partials).
    # Summing the 4 column partials along dim3 reconstructs the full reduced matmul.
    mm_row0 = mm_torch[0:1]  # one column-row copy: [1,1,M,N*? ] -> actually [1,1,M,N] with 4 partials concatenated
    mm_reduced = torch.sum(torch.stack(torch.chunk(mm_row0, num_col_devices, dim=3)), dim=0)
    ok_mm, pcc_mm = comp_pcc(mm_reduced, ref_full, 0.99)
    logger.info(f"matmul PCC: {pcc_mm}")

    # ----- verify reduce-scatter output -----
    rs_torch = ttnn.to_torch(
        ttnn.from_device(rs_out),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    rs_row0 = rs_torch[0:1]  # [1,1,M,N] = the 4 scattered shards concatenated
    ref_rs_full = torch.cat(ref_scatter, dim=3)
    ok_rs, pcc_rs = comp_pcc(rs_row0, ref_rs_full, 0.99)
    logger.info(f"reduce-scatter PCC: {pcc_rs}")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    assert ok_mm, f"matmul PCC failed: {pcc_mm}"
    assert ok_rs, f"reduce-scatter PCC failed: {pcc_rs}"
