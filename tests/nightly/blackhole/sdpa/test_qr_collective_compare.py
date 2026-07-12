# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate the COLLECTIVE's contribution to the KV-gather cost.

The production report (Galaxy SP=8xTP=4, FABRIC_2D) measures the KV gather as AllBroadcastDeviceOperation =
~36 ms @500K. My earlier number (7.26 ms) used a ring all_gather_async — a different, more efficient
collective. Same DATA (the MQA-shared KVPE is head/TP-independent, so one SP=8 ring gathers the same bytes
whether on this box or on one of the Galaxy's 4 TP rings). This measures both collectives for the identical KV
tensor on the same SP=8 ring, so the gap that remains vs 36 ms is contention (4 rings share the Galaxy fabric,
not reproducible on 8 chips) + real cache/DRAM overhead — NOT data volume.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import create_global_semaphores

K_DIM = 576


def _shard_kv(mesh_device, sp, ca, T):
    dims = [None, None]
    dims[ca] = 2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.rand([1, 1, T, K_DIM]).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _median(fn, sub_id, mesh_device, iters=6, warmup=2):
    ts = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        dt = (time.perf_counter() - t0) * 1e3
        if isinstance(out, list):
            for o in out:
                ttnn.deallocate(o)
        else:
            ttnn.deallocate(out)
        if i >= warmup:
            ts.append(dt)
    ts.sort()
    return ts[len(ts) // 2]


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["sp8"])
def test_collective_compare(mesh_device):
    sp, ca = 8, 1
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    sem = create_global_semaphores(mesh_device, sp, crs, 0)

    logger.info(f"KV-gather collective comparison, SP={sp} ring — same KV tensor, two collectives")
    logger.info(f"{'T':>9} {'all_gather(ring,2L)':>20} {'all_broadcast(prod)':>21} {'bcast/ag':>9}")
    for T in [65536, 131072, 262144, 524288]:
        kv = _shard_kv(mesh_device, sp, ca, T)

        def ag():
            return ttnn.experimental.all_gather_async(
                kv,
                dim=2,
                multi_device_global_semaphore=sem,
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                cluster_axis=ca,
                subdevice_id=sub_id,
            )

        def bc():
            return ttnn.all_broadcast(
                kv,
                cluster_axis=ca,
                num_links=1,
                topology=ttnn.Topology.Linear,
                subdevice_id=sub_id,
            )

        ag_ms = _median(ag, sub_id, mesh_device)
        try:
            bc_ms = _median(bc, sub_id, mesh_device)
            ratio = f"{bc_ms/ag_ms:.2f}x"
        except Exception as e:  # noqa: BLE001
            bc_ms, ratio = float("nan"), f"ERR:{type(e).__name__}"
            logger.warning(f"all_broadcast failed at T={T}: {e}")
        logger.info(f"{T:>9} {ag_ms:>20.3f} {bc_ms:>21.3f} {ratio:>9}")
        ttnn.deallocate(kv)
