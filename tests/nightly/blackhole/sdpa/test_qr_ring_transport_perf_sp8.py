# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""qr-ring transport perf at SP=8 — the full 8-chip LoudBox ring (FABRIC_1D_RING + Topology.Ring).

Same measurement as test_qr_ring_transport_perf.py but over all 8 devices in one ring. SP=8 is the plan's
production sequence-parallel width, and its ring fraction (7/8) is worse for KV-gather than SP=4's (3/4), so
the O(T) KV-gather slope is steeper and the crossover drops.
"""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import create_global_semaphores

K_DIM, V_DIM, H, SQ = 576, 512, 128, 512


def _timed(mesh_device, sp, cluster_axis, shape, dim, dtype, num_links, iters=6, warmup=2):
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    sem = create_global_semaphores(mesh_device, sp, crs, 0)
    dims = [None, None]
    dims[cluster_axis] = dim
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
    torch.manual_seed(0)
    inp = ttnn.from_torch(
        torch.rand(shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )

    def run():
        return ttnn.experimental.all_gather_async(
            inp,
            dim=dim,
            multi_device_global_semaphore=sem,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=cluster_axis,
            subdevice_id=sub_id,
        )

    ts = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        o = run()
        ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
        dt = (time.perf_counter() - t0) * 1e3
        ttnn.deallocate(o)
        if i >= warmup:
            ts.append(dt)
    ttnn.deallocate(inp)
    ts.sort()
    return ts[len(ts) // 2]


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
# NOTE: only the full 8-chip ring trains on this LoudBox — a (1,4) ring subset fails fabric router sync
# (device 4 can't complete the ethernet handshake). So SP=8 is measured as a ring here; the SP=4 baseline
# in test_qr_ring_transport_perf.py is LINEAR (native 2x4 axis). Ring vs linear is a real confound — see notes.
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["sp8"])
def test_qr_transport_sp8(mesh_device):
    sp, ca, nl = tuple(mesh_device.shape)[1], 1, 2
    logger.info(f"SP={sp} ring (num_links={nl})")
    q_ms = _timed(mesh_device, sp, ca, [1, H, SQ, K_DIM], 2, ttnn.bfloat16, nl)
    o_ms = _timed(mesh_device, sp, ca, [1, H, SQ, V_DIM], 2, ttnn.bfloat16, nl)
    qr = q_ms + o_ms
    rows = []
    for T in [4096, 16384, 65536, 131072, 262144, 524288]:
        ms = _timed(mesh_device, sp, ca, [1, 1, T, K_DIM], 2, ttnn.bfloat16, nl)
        rows.append((T, ms))
        logger.info(f"KV-gather T={T:>7} {ms:8.3f} ms")
    logger.info("===== qr-ring transport @ SP=8 (measured, LoudBox 8-chip ring) =====")
    logger.info(f"Q-gather all_gather(Q) {q_ms:.3f} ms + reduce(O) {o_ms:.3f} ms = {qr:.3f} ms  [FLAT in T]")
    logger.info(f"{'T (ctx)':>10} {'KV ms':>10} {'qr ms':>9} {'speedup':>9}")
    for T, ms in rows:
        logger.info(f"{T:>10} {ms:>10.3f} {qr:>9.3f} {ms / qr:>8.2f}x")
    assert rows[-1][1] > rows[0][1], "KV-gather must grow with T"
