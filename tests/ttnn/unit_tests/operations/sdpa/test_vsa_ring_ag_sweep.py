# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How fast is the multi-worker all_gather_async at the 15 s K-shard shape as a function of workers per link?
Every sender core is taken from the VSA grid, so the fused op wants the fewest workers that still saturate the links."""
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.test import ring_params_8k_req_exact_devices


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {**ring_params_8k_req_exact_devices, "l1_small_size": 65536}, id="4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_ag_workers_sweep(mesh_device, reset_seeds):
    sp_axis, tp_axis, num_links = 1, 0, 2
    heads_total, t_local, dim = 56, 226 * 64, 128
    mesh_shape = tuple(mesh_device.shape)
    shard = [None, None]
    shard[tp_axis] = 1
    shard[sp_axis] = 2
    heads_only = [None, None]
    heads_only[tp_axis] = 1
    t_total = t_local * mesh_shape[sp_axis]
    k = torch.randn(1, heads_total, t_total, dim, dtype=torch.bfloat16)
    to_dev = lambda t, dims: ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    tt_k = to_dev(k, shard)
    tt_gk = to_dev(torch.zeros_like(k), heads_only)
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [[ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2)] for _ in range(2)]
    calls = [0]
    results = {}
    for workers in (None, 1, 2, 4):
        kw = {} if workers is None else {"num_workers_per_link": workers}

        def ag():
            calls[0] += 1
            return ttnn.experimental.all_gather_async(
                tt_k,
                persistent_output_buffer=tt_gk,
                dim=2,
                multi_device_global_semaphore=sems[calls[0] % 2],
                num_links=num_links,
                topology=ttnn.Topology.Ring,
                cluster_axis=sp_axis,
                **kw,
            )

        ag()
        ttnn.synchronize_device(mesh_device)
        iters = 6
        t0 = time.perf_counter()
        for _ in range(iters):
            ag()
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) * 1e3 / iters
        results[workers] = ms
        logger.info(f"all_gather_async K-shard workers_per_link={workers}: {ms:.2f} ms")
    print("\nAG_SWEEP " + " ".join(f"w{w}={ms:.2f}" for w, ms in results.items()))
