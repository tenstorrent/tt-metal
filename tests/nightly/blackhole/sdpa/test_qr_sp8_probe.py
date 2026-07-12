# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: can this LoudBox form an 8-chip SP ring? Tries FABRIC_1D_RING + Topology.Ring on an 8-device mesh
and runs one all_gather. If the ethernet ring won't train, this fails fast at mesh-open (router-sync timeout,
~10s self-abort) rather than hanging. Mirrors the galaxy_torus smoke config.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import create_global_semaphores


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(1, 8), (8, 1)], indirect=True, ids=["1x8", "8x1"])
def test_sp8_ring_probe(mesh_device):
    shape = tuple(mesh_device.shape)
    sp = max(shape)
    cluster_axis = shape.index(sp)
    logger.info(f"OPENED mesh {shape}, SP={sp}, cluster_axis={cluster_axis} — fabric ring trained OK")

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    sem = create_global_semaphores(mesh_device, sp, crs, 0)

    dims = [None, None]
    dims[cluster_axis] = 2  # shard seq dim across the SP axis
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=shape)
    T = 8192
    full = torch.rand([1, 1, T, 576]).bfloat16()
    inp = ttnn.from_torch(
        full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    def run():
        return ttnn.experimental.all_gather_async(
            inp,
            dim=2,
            multi_device_global_semaphore=sem,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            subdevice_id=sub_id,
        )

    out = run()
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
    logger.info(f"all_gather over {sp} devices SUCCEEDED, out shape {tuple(out.shape)}")
    t0 = time.perf_counter()
    for _ in range(5):
        o = run()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        ttnn.deallocate(o)
    ms = (time.perf_counter() - t0) / 5 * 1e3
    logger.info(f"SP={sp} all_gather(KVPE T={T}) median-ish {ms:.3f} ms")
    assert tuple(out.shape)[2] == T, "gathered seq must be full T"
