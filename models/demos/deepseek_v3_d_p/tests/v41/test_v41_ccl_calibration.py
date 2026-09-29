# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CCL link-rate calibration for the V4.1 perf model on the LoudBox 2x4 (bead 8y7.9.3).

``utils/v41_perf_model.collective_ns`` charges ``bottleneck-edge bytes / (link rate x links) + hops x hop latency``
with 25 GB/s per link. The G2 profile measured the TP all-to-alls at 1.65-2.5x that "ideal", so the rate is too low
for at least that path. This test times the collectives V4.1 uses, on both mesh axes, at several per-chip sizes and
1 / 2 links, so ``rate`` and ``latency`` can be fitted per (kind, axis, links) against the model's edge bytes.

Collectives (bf16, DRAM, TILE, ``[1, 1, rows, 1024]`` per chip):

* TP (axis 1, 4 chips): ``V41Collectives.tp_all_gather`` / ``tp_reduce_scatter`` / ``tp_all_to_all`` (dim 3 -> 2);
* SP (axis 0, 2 chips): ``all_gather_async`` and ``all_to_all_async_generic`` with ``cluster_axis=0``.

Timing: ``ITERS`` back-to-back calls captured in one trace, one warm replay, then ``REPLAYS`` timed replays with a
device synchronize around them; time per call = wall / (REPLAYS x ITERS). Host dispatch is excluded by the trace;
inter-op gaps inside the trace are included (they are part of a traced layer too). One ``CCL_CAL {json}`` log line
per case; ``scripts/ccl_fit.py`` (artifacts) fits the rate and latency.
"""

import json
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import MESH
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS, TP_AXIS

WIDTH = 1024  # bf16 columns per chip: 2 KiB per row
ROWS = (128, 512, 2048, 8192)  # 256 KiB .. 16 MiB per chip
LINKS = (1, 2)
ITERS = 8
REPLAYS = 5
CASES = ("tp_all_gather", "tp_reduce_scatter", "tp_all_to_all", "sp_all_gather", "sp_all_to_all")


def _call(coll: V41Collectives, case: str, x, links: int):
    coll.num_links = links
    if case == "tp_all_gather":
        return coll.tp_all_gather(x, dim=3)
    if case == "tp_reduce_scatter":
        return coll.tp_reduce_scatter(x, dim=3)
    if case == "tp_all_to_all":
        return coll.tp_all_to_all(x, in_dim=3, out_dim=2)
    if case == "sp_all_gather":
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            multi_device_global_semaphore=coll.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
            barrier_semaphore=coll.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=SP_AXIS),
            num_links=links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=coll.sp_topology,
            cluster_axis=SP_AXIS,
        )
    if case == "sp_all_to_all":
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=3,
            out_dim=2,
            num_links=links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=coll.sp_topology,
            cluster_axis=SP_AXIS,
        )
    raise ValueError(case)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_ccl_calibration(mesh_device, device_params, case):
    assert tuple(mesh_device.shape) == (2, 4)
    coll = V41Collectives(mesh_device)
    axis = TP_AXIS if case.startswith("tp") else SP_AXIS
    n = mesh_device.shape[axis]
    for links in LINKS:
        for rows in ROWS:
            t0 = time.perf_counter()
            x = ttnn.from_torch(
                torch.randn(1, 1, rows, WIDTH, dtype=torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            y = _call(coll, case, x, links)  # compile + correctness of the output shape
            want = [1, 1, rows, WIDTH]
            if "all_gather" in case:
                want[3] *= n
            elif "reduce_scatter" in case:
                want[3] //= n
            else:
                want[3], want[2] = WIDTH * n, rows // n
            assert list(ttnn.get_device_tensors(y)[0].shape) == want, (case, list(y.shape), want)
            ttnn.synchronize_device(mesh_device)
            outs = []
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(ITERS):
                outs.append(_call(coll, case, x, links))
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)  # warm replay
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            us = (time.perf_counter() - start) * 1e6 / (REPLAYS * ITERS)
            ttnn.release_trace(mesh_device, tid)
            rec = dict(case=case, axis=axis, chips=n, links=links, rows=rows, in_bytes=rows * WIDTH * 2, us=us)
            logger.info(f"CCL_CAL {json.dumps(rec)} (case {time.perf_counter() - t0:.1f}s)")
            for t in outs + [x, y]:
                ttnn.deallocate(t)
