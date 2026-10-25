# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Two-op concurrency repro for the two-chunk pipeline's deadlock matrix (notes/pipeline/deadlock-matrix.md).

One traced loop issues op A on sub-device SD_A and op B on SD_B back to back, PAIR_N times, with no ordering between
them, so the two run concurrently on every chip. The trace is replayed PAIR_R times. A watchdog thread kills the process
(os._exit(124)) when one replay + synchronize takes longer than PAIR_WATCHDOG_S, so a fabric deadlock costs seconds.

Manager: {SD0 = x0..10, SD1 = x11} (the persistent pipeline layout). Ops (PAIR_A / PAIR_B = "<op>@<sd>", sd 0|1):
  rs1r / rs1l  reduce_scatter_minimal_async [1,1,640,6144] bf16, axis 1, Ring / Linear (1 worker/link on SD1)
  ag1r / ag1l  all_gather_async [1,1,640,1536] -> 6144, axis 1, Ring / Linear (1 worker/link on SD1)
  ag0r / ag0l  all_gather_async [1,1,640,1536] -> 6144 on axis 0 (the dispatch axis), Ring / Linear
  hb1          high_bw_all_gather [1,1,640,512] -> dim 3 x4 on axis 1 (the kv-stem / q_a gathers)
  mm           2D matmul [640,2048] x [2048,4096] on SD0 (no fabric; control)
Example: PAIR_A=rs1l@1 PAIR_B=rs1r@0 pytest ... -k torus-xy-8x4
"""

import os
import threading
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

NUM_LINKS = 2


def _watchdog(deadline_box, label):
    while True:
        time.sleep(0.5)
        d = deadline_box[0]
        if d is not None and time.monotonic() > d:
            logger.error(f"[pair] WATCHDOG: replay of {label} exceeded its deadline -> hang; exiting 124")
            os._exit(124)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=12352, l1_small_size=1536, trace_region_size=64 * 1024 * 1024),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(600)
def test_pipe_pair_repro(mesh_device, device_params):
    spec_a, spec_b = os.environ["PAIR_A"], os.environ["PAIR_B"]
    n, r = int(os.environ.get("PAIR_N", "16")), int(os.environ.get("PAIR_R", "20"))
    wd = float(os.environ.get("PAIR_WATCHDOG_S", "20"))
    tt_ccl = get_tt_ccl(mesh_device)
    g = mesh_device.compute_with_storage_grid_size()
    sd_cores = [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 2, g.y - 1))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(g.x - 1, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))}),
    ]
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([sd_cores[0]]), ttnn.SubDevice([sd_cores[1]])], 0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(shape, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            torch.randn(shape) * 0.01,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    def make(spec):
        name, sd = spec.split("@")
        sd = int(sd)
        sd_id = ttnn.SubDeviceId(sd)
        w = {"num_workers_per_link": 1} if sd == 1 else {}
        if name in ("rs1r", "rs1l"):
            x = dev((1, 1, 640, 6144))
            topo = ttnn.Topology.Ring if name == "rs1r" else ttnn.Topology.Linear
            mid = ttnn.from_torch(
                torch.zeros(([2] if topo == ttnn.Topology.Linear else []) + [1, 1, 640, 6144]),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            sems = [tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1) for _ in range(2)]
            bars = [tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1) for _ in range(2)]
            state = {"i": 0}

            def fn():
                i = state["i"] % 2
                state["i"] += 1
                return ttnn.experimental.reduce_scatter_minimal_async(
                    x,
                    persistent_output_buffers=[mid],
                    dim=3,
                    multi_device_global_semaphore=sems[i],
                    barrier_semaphore=bars[i],
                    num_links=NUM_LINKS,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topo,
                    cluster_axis=1,
                    subdevice_id=sd_id,
                    **w,
                )

            return fn
        if name in ("ag1r", "ag1l", "ag0r", "ag0l"):
            axis = 1 if name.startswith("ag1") else 0
            topo = ttnn.Topology.Ring if name.endswith("r") else ttnn.Topology.Linear
            x = dev((1, 1, 640, 1536))
            sems = [tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=axis) for _ in range(2)]
            bars = [tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=axis) for _ in range(2)]
            state = {"i": 0}

            def fn():
                i = state["i"] % 2
                state["i"] += 1
                return ttnn.experimental.all_gather_async(
                    x,
                    dim=-1,
                    cluster_axis=axis,
                    multi_device_global_semaphore=sems[i],
                    barrier_semaphore=bars[i],
                    num_links=NUM_LINKS,
                    topology=topo,
                    subdevice_id=sd_id,
                    **w,
                )

            return fn
        if name == "hb1":
            x = dev((1, 1, 640, 512))
            out = dev((1, 1, 640, 2048))

            def fn():
                ttnn.experimental.high_bw_all_gather(
                    x,
                    dim=3,
                    output_tensor=out,
                    cluster_axis=1,
                    num_links=NUM_LINKS,
                    subdevice_id=sd_id,
                    sub_core_grids=sd_cores[sd],
                )
                return None

            return fn
        if name == "mm":
            assert sd == 0
            a, b = dev((1, 1, 640, 2048)), dev((1, 1, 2048, 4096), ttnn.bfloat8_b)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(g.x - 1, g.y),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=2,
                per_core_N=12,
                transpose_mcast=False,
                fuse_batch=False,
                fused_activation=None,
            )

            def fn():
                return ttnn.matmul(a, b, program_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_device_id=sd_id)

            return fn
        raise ValueError(spec)

    mesh_device.load_sub_device_manager(mgr)
    try:
        fa, fb = make(spec_a), make(spec_b)
        label = f"{spec_a} || {spec_b}"
        # eager compile, each alone then together once
        for f in (fa, fb):
            o = f()
            ttnn.synchronize_device(mesh_device)
            if o is not None:
                ttnn.deallocate(o)
        outs = []
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(n):
            outs += [fa(), fb()]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        deadline = [None]
        threading.Thread(target=_watchdog, args=(deadline, label), daemon=True).start()
        times = []
        for _ in range(r):
            deadline[0] = time.monotonic() + wd
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6 / n)
            deadline[0] = None
        ttnn.release_trace(mesh_device, tid)
        times.sort()
        logger.info(f"[pair] RESULT {label}: OK, per pair-iteration median {times[len(times) // 2]:.1f} us (n={n}, r={r})")
        for o in outs:
            if o is not None:
                ttnn.deallocate(o)
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(mgr)
