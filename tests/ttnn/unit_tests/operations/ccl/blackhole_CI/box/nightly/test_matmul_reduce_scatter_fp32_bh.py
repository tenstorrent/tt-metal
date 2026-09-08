# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the fp32 fused matmul + reduce-scatter path on a 4 chip Blackhole box.

With fp32 tiles a page is 4kB, so only one tile fits in a fabric packet and
num_tiles_to_write_per_packet becomes 1. The ring reduce-scatter writer used to prime its scatter
packet header with that value, which is below NOC_SCATTER_WRITE_MIN_CHUNKS (2), and the fabric
ASSERT in populate_unicast_scatter_write_fields tripped the watcher. See #53329.

Shapes and program config mirror the Qwen3.6-27B GDN prefill out projection
(models/demos/blackhole/qwen36/tt/tp_common.py::matmul_reduce_scatter_prefill).
"""
import math

import pytest
import torch

import ttnn

TILE = 32


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_fused_matmul_reduce_scatter_atomic_barrier(mesh_device):
    nd = mesh_device.get_num_devices()
    M, K_local, N = 128, 1536, 5120
    grid = (8, 8)
    rs_dtype = ttnn.float32

    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))})
    rs_sems = [ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(3)]
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    interm = ttnn.from_torch(
        torch.zeros(1, 1, M, N),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=rs_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_buf = ttnn.from_torch(
        torch.zeros(1, 1, M, N // nd),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=rs_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    x = ttnn.from_torch(
        torch.randn(1, 1, M, K_local),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(K_local, N),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    per_core_N = max(1, math.ceil(N / TILE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE // grid[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=crs,
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    for _ in range(5):
        _, rs = ttnn.experimental.matmul_reduce_scatter_async(
            x,
            w,
            persistent_intermediate_buffer=interm,
            persistent_output_buffer=out_buf,
            dim=3,
            multi_device_global_semaphore=rs_sems,
            reduce_scatter_core_grid_offset=(0, 8),
            barrier_semaphore=barrier_sem,
            num_links=2,
            memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            subdevice_id=None,
            memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            compute_kernel_config=ckc,
        )
        ttnn.deallocate(ttnn.clone(rs, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ttnn.synchronize_device(mesh_device)
