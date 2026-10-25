# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pipeline E1b: GLM-5.3 TP collectives with their workers confined to the persistent comm column (x = 11).

8x4 BH Galaxy, FABRIC_2D_TORUS_XY, num_links=2, 640 rows/chip. Each case is timed as DB_K back-to-back launches in one
trace, replayed DB_R times (median per launch, us). Placements:
  full = default manager (whole 12x10 grid; today's placement)
  main = persistent-layout manager {SD0 x0..10, SD1 x=11}, workers on SD0
  col  = same manager, workers on SD1 (the 10-core comm column)
Cases (env E1B_CASES, comma list; default all):
  rs_ring / rs_linear: reduce_scatter_minimal_async [1,1,640,6144] bf16 on axis 1 (o_proj / post-combine RS shape),
      default workers and num_workers_per_link=1 (the column holds 10 cores: 2 links x 2 dirs x (1 worker + 1 mux))
  gate_hbag: the gate all-reduce's all-gather leg, high_bw_all_gather [1,1,640,256] bf16 dim 1 on axis 1
  ag_x: the MoE x all-gather, all_gather_async [1,1,640,1536] -> 6144 on axis 1
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

NUM_LINKS = 2


def _time_trace(mesh_device, fn, k, r, label):
    fn()  # compile (eager)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(k):
        fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(r):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e6 / k)
    ttnn.release_trace(mesh_device, tid)
    times.sort()
    med = times[len(times) // 2]
    logger.info(f"[e1b] RESULT {label}: per_launch_us median={med:.1f} min={times[0]:.1f} max={times[-1]:.1f}")
    return med


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=12352, l1_small_size=1536, trace_region_size=256 * 1024 * 1024),
            id="torus-xy-8x4",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(900)
def test_e1_comm_ccl_bench(mesh_device, device_params):
    k, r = int(os.environ.get("DB_K", "8")), int(os.environ.get("DB_R", "5"))
    cases = [c for c in os.environ.get("E1B_CASES", "rs_ring,rs_linear,gate_hbag,ag_x").split(",") if c]
    tt_ccl = get_tt_ccl(mesh_device)
    g = mesh_device.compute_with_storage_grid_size()
    main = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 2, g.y - 1))})
    col = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(g.x - 1, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([main]), ttnn.SubDevice([col])], 0)
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

    rs_x = dev((1, 1, 640, 6144))
    gate_x = dev((1, 1, 640, 256))
    gate_out = dev((1, 4, 640, 256))
    ag_in = dev((1, 1, 640, 1536))
    results = {}

    placements = [("full", None, None), ("main", ttnn.SubDeviceId(0), main), ("col", ttnn.SubDeviceId(1), col)]
    for pname, sd, cores in placements:
        if sd is None:
            mesh_device.clear_loaded_sub_device_manager()
        else:
            mesh_device.load_sub_device_manager(mgr)
        sd_kw = {} if sd is None else {"subdevice_id": sd}
        for case in cases:
            variants = []
            if case in ("rs_ring", "rs_linear"):
                topo = ttnn.Topology.Ring if case == "rs_ring" else ttnn.Topology.Linear
                for w in (None, 1):
                    if pname == "col" and w is None:
                        continue  # the default worker count does not fit 10 cores

                    def fn(topo=topo, w=w):
                        out = ttnn.experimental.reduce_scatter_minimal_async(
                            rs_x,
                            persistent_output_buffers=None,
                            dim=3,
                            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
                            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
                            num_links=NUM_LINKS,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            topology=topo,
                            cluster_axis=1,
                            num_workers_per_link=w,
                            **sd_kw,
                        )
                        ttnn.deallocate(out)

                    variants.append((f"{case} w={w or 'dflt'}", fn))
            elif case == "gate_hbag":

                def fn():
                    ttnn.experimental.high_bw_all_gather(
                        gate_x,
                        dim=1,
                        output_tensor=gate_out,
                        cluster_axis=1,
                        num_links=NUM_LINKS,
                        **({} if sd is None else {"subdevice_id": sd, "sub_core_grids": cores}),
                    )

                variants.append(("gate_hbag", fn))
            elif case == "ag_x":

                def fn():
                    out = ttnn.experimental.all_gather_async(
                        ag_in,
                        dim=-1,
                        cluster_axis=1,
                        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
                        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
                        num_links=NUM_LINKS,
                        topology=ttnn.Topology.Ring,
                        **sd_kw,
                    )
                    ttnn.deallocate(out)

                variants.append(("ag_x ring", fn))
            else:
                raise ValueError(f"E1B_CASES item {case!r}")
            for vname, fn in variants:
                label = f"{pname} {vname}"
                try:
                    results[label] = _time_trace(mesh_device, fn, k, r, label)
                except Exception as ex:  # one failing variant must not hide the others
                    logger.error(f"[e1b] FAIL {label}: {type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}")
                    results[label] = None
                    ttnn.synchronize_device(mesh_device)
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(mgr)
    logger.info(f"[e1b] SUMMARY {results}")
