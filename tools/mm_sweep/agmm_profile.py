# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 3/4 overlap evidence for all_gather_regime_a_matmul_async via the device profiler.
#
# Runs the fused op under TT_METAL_DEVICE_PROFILER=1, reads per-RISC kernel zones from
# generated/profiler/.logs/profile_log_device.csv, and — per profiled device (a common cycle timebase across
# that device's cores) — compares the fabric INJECTOR (BRISC-KERNEL) end against the earliest COMPUTE
# (TRISC-KERNEL) start. If compute starts before the injector finishes the gather, the matmul is overlapping
# the still-arriving remote shards. Streaming (default) should overlap; TT_AGMM_FULL_GATHER=1 should not.
#
# Usage (one mode per process; the mode is a build-time env flag so programs must not share a cache):
#   TT_METAL_DEVICE_PROFILER=1 python tools/mm_sweep/agmm_profile.py <mode> <D> <mesh_y> <mesh_x> <M> <K> <N>
#   (mode is a label only; set TT_AGMM_FULL_GATHER=1 in the env for the no-overlap baseline.)

import csv
import gc
import json
import os
import sys
from collections import defaultdict

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0.0


def parse_overlap():
    import statistics
    rows = list(csv.reader(open(CSV)))
    # per (device, core_x, core_y, zone): ordered list of (start,end) pairs, one per profiled run.
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[0].strip(), row[1].strip(), row[2].strip(), row[10].strip())].append((row[11].strip(), int(row[5])))
    runs = defaultdict(list)  # (dev,x,y,zone) -> [(start,end), ...] per run
    for k, l in ev.items():
        st = None
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                runs[k].append((st, c))
                st = None
    # number of steady-state runs to score (drop warmup run 0): min pairs across zones, capped.
    nz = [len(p) for p in runs.values() if p]
    nruns = min(nz) if nz else 0
    if nruns <= 1:
        run_ids = list(range(nruns))
    else:
        run_ids = list(range(1, nruns))  # drop warmup
    # per device, per run: total span (max end - min start over all zones), compute span, injector span.
    trisc_cores = defaultdict(set)
    for (dev, x, y, zone) in runs:
        if zone == "TRISC-KERNEL":
            trisc_cores[dev].add((x, y))
    devs = sorted({k[0] for k in runs})
    out = {}
    for dev in devs:
        n_trisc = len(trisc_cores[dev])
        if n_trisc == 0:
            continue
        totals, comps, injs = [], [], []
        for r in run_ids:
            allpairs = [runs[k][r] for k in runs if k[0] == dev and len(runs[k]) > r]
            if not allpairs:
                continue
            tmin = min(s for s, _ in allpairs)
            tmax = max(e for _, e in allpairs)
            totals.append((tmax - tmin) / FREQ * 1e6)
            cp = [runs[k][r] for k in runs if k[0] == dev and k[3] == "TRISC-KERNEL" and len(runs[k]) > r]
            if cp:
                comps.append((max(e for _, e in cp) - min(s for s, _ in cp)) / FREQ * 1e6)
            ip = [runs[k][r] for k in runs
                  if k[0] == dev and k[3] == "BRISC-KERNEL" and (k[1], k[2]) not in trisc_cores[dev]
                  and len(runs[k]) > r]
            if ip:  # injector core = BRISC with no TRISC; take the latest-ending such core's span
                s0, e0 = max(ip, key=lambda p: p[1])
                injs.append((e0 - s0) / FREQ * 1e6)
        if not totals:
            continue
        out[dev] = {
            "n_compute_cores": n_trisc,
            "runs": len(totals),
            "total_span_us_median": round(_median(totals), 2),
            "total_span_us_min": round(min(totals), 2),
            "total_span_us_max": round(max(totals), 2),
            "compute_span_us_median": round(_median(comps), 2) if comps else None,
            "injector_span_us_median": round(_median(injs), 2) if injs else None,
        }
    return out


def main():
    mode, D, my, mx, M, K, N = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                                int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    try:
        os.remove(CSV)
    except OSError:
        pass
    import torch
    import ttnn
    from models.common.utility_functions import comp_pcc

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full = ttnn.open_mesh_device(ttnn.MeshShape(my, mx))
    try:
        md = full.create_submesh(ttnn.MeshShape(1, D))
        grid = md.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        mgr = md.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
        md.load_sub_device_manager(mgr)
        md.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
        torch.manual_seed(0)
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        ref = (t0.float() @ t1.float())[0, 0]
        a = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=md, dtype=ttnn.bfloat16, mesh_mapper=ttnn.create_mesh_mapper(
            md, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, D))))
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, md)
        b = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=md, dtype=ttnn.bfloat16, memory_config=wcfg,
                            mesh_mapper=ttnn.create_mesh_mapper(md, ttnn.MeshMapperConfig(
                                [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, D))))
        cfg = None  # auto-picker (config=None) so any corpus shape, incl. narrow-N, is handled
        sems = [ttnn.create_global_semaphore(md, crs, 0) for _ in range(2 * D)]
        out = None
        for _ in range(12):
            for s in sems:
                ttnn.reset_global_semaphore_value(s, 0)
            out = ttnn.experimental.all_gather_regime_a_matmul_async(
                a, b, config=cfg, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=1,
                num_workers_per_link=1, multi_device_global_semaphore=sems)
            ttnn.synchronize_device(md)
        per = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        _, pcc = comp_pcc(ref, per[0].float().reshape(ref.shape), 0.999)
        ttnn.ReadDeviceProfiler(md)  # ReadMeshDeviceProfilerResults; CSV flushes on the following clean close
        # Clean teardown so the profiler flushes: fully release the submesh's sub-device manager + stall group,
        # close the SUBMESH, then the parent. (An abort here leaves profile_log_device.csv unwritten.)
        a = b = out = per = None
        md.reset_sub_device_stall_group()
        md.clear_loaded_sub_device_manager()
        md.remove_sub_device_manager(mgr)
        ttnn.close_mesh_device(md)
        md = None
        gc.collect()
        ttnn.close_mesh_device(full)
        full = None
        gc.collect()
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ov = parse_overlap()  # parse AFTER close so the device CSV is on disk
        print("PROFJSON " + json.dumps({"mode": mode, "D": D, "shape": [M, K, N], "pcc": float(pcc),
                                        "per_device": ov}))
    finally:
        if full is not None:
            try:
                ttnn.close_mesh_device(full)
            except Exception:
                pass
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
