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

ROOT = "/data/cglagovich/tt-metal/.claude/worktrees/resilient-marinating-piglet"
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9


def parse_overlap():
    rows = list(csv.reader(open(CSV)))
    # per (device, core_x, core_y, risc, zone): list of (type, cycle)
    ev = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        dev = row[0].strip()
        ev[(dev, row[1].strip(), row[2].strip(), row[10].strip())].append((row[11].strip(), int(row[5])))
    # last run's span per (dev,core,zone): (start, end)
    spans = {}
    for k, l in ev.items():
        st = None
        pairs = []
        for t, c in l:
            if t == "ZONE_START":
                st = c
            elif t == "ZONE_END" and st is not None:
                pairs.append((st, c))
                st = None
        if pairs:
            spans[k] = pairs[-1]  # last (steady-state) run
    # per device: injector BRISC end (max), earliest compute TRISC start (min), latest compute TRISC end
    per_dev = defaultdict(lambda: {"inj_start": None, "inj_end": None, "comp_start": None, "comp_end": None,
                                   "n_trisc": 0})
    # heuristic: the injector is the lone BRISC-KERNEL core that is NOT also running a TRISC-KERNEL (compute
    # cores run reader+compute; the injector core runs only the injector DM kernel). We approximate the injector
    # as the BRISC core with the LATEST end that has no TRISC zone, and compute as all TRISC cores.
    trisc_cores = defaultdict(set)
    for (dev, x, y, zone), (s, e) in spans.items():
        if zone == "TRISC-KERNEL":
            trisc_cores[dev].add((x, y))
    for (dev, x, y, zone), (s, e) in spans.items():
        d = per_dev[dev]
        if zone == "TRISC-KERNEL":
            d["n_trisc"] += 1
            d["comp_start"] = s if d["comp_start"] is None else min(d["comp_start"], s)
            d["comp_end"] = e if d["comp_end"] is None else max(d["comp_end"], e)
        elif zone == "BRISC-KERNEL" and (x, y) not in trisc_cores[dev]:
            # candidate injector core (no compute on it)
            if d["inj_end"] is None or e > d["inj_end"]:
                d["inj_end"] = e
                d["inj_start"] = s
    out = {}
    for dev, d in per_dev.items():
        if d["n_trisc"] == 0 or d["inj_end"] is None:
            continue
        overlap_cyc = d["inj_end"] - d["comp_start"]  # >0 => compute started before injector finished gather
        out[dev] = {
            "n_compute_cores": d["n_trisc"],
            "inj_span_us": round((d["inj_end"] - d["inj_start"]) / FREQ * 1e6, 2),
            "compute_span_us": round((d["comp_end"] - d["comp_start"]) / FREQ * 1e6, 2),
            "compute_start_before_inj_end_us": round(overlap_cyc / FREQ * 1e6, 2),
            "overlap": overlap_cyc > 0,
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
        md.load_sub_device_manager(md.create_sub_device_manager([ttnn.SubDevice([crs])], 0))
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
        cfg = ttnn.RegimeAMatmulConfig(k_slices=3, n_slices=1, m_slices=1, k_block_tiles=4, n_subblock_tiles=6)
        sems = [ttnn.create_global_semaphore(md, crs, 0) for _ in range(D + 1)]
        out = None
        for _ in range(6):
            for s in sems:
                ttnn.reset_global_semaphore_value(s, 0)
            out = ttnn.experimental.all_gather_regime_a_matmul_async(
                a, b, config=cfg, cluster_axis=1, topology=ttnn.Topology.Ring, num_links=1,
                num_workers_per_link=1, multi_device_global_semaphore=sems)
            ttnn.synchronize_device(md)
        per = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        _, pcc = comp_pcc(ref, per[0].float().reshape(ref.shape), 0.999)
        try:
            ttnn.ReadMeshDeviceProfiler(md)
        except AttributeError:
            ttnn.ReadDeviceProfiler(md)
        # Parse + emit BEFORE teardown so a submesh-close hiccup never loses the profiling result.
        ov = parse_overlap()
        print("PROFJSON " + json.dumps({"mode": mode, "D": D, "shape": [M, K, N], "pcc": float(pcc),
                                        "per_device": ov}))
    finally:
        a = b = out = per = None
        md = None
        gc.collect()  # release the submesh's CQ hold before closing the parent mesh
        ttnn.close_mesh_device(full)
        gc.collect()
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
