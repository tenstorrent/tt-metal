# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 3 overlap PROOF for all_gather_regime_a_matmul_async via custom device-profiler markers.
#
# The streaming-ring kernels emit four named zones (DeviceZoneScopedN), all on the same per-device cycle
# timebase:
#   AGMM_FIRST_LOCAL_MM   (in0 reader) - first step-0 in0 block whose gate passed for a LOCAL shard block
#   AGMM_FIRST_REMOTE_MM  (in0 reader) - first step-0 in0 block whose gate passed for a REMOTE shard block
#   AGMM_LOCAL_TX_DONE    (relay)      - this device's OWN shard's last block has been forwarded to the neighbour
#   AGMM_GATHER_DONE      (relay)      - every shard's every block is now stored locally (all-gather complete)
#
# Overlap is PROVEN, per device, per run, when:
#   (a) first_local_mm  <  local_tx_done   (compute on local data starts before local-shard transmit finishes)
#   (b) first_remote_mm <  gather_done      (compute on a remote shard starts before the full gather finishes)
#
# Usage:  TT_METAL_DEVICE_PROFILER=1 python tools/mm_sweep/agmm_markers.py <D> <mesh_y> <mesh_x> <M> <K> <N>

import csv
import gc
import json
import os
import sys
from collections import defaultdict

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9

MARKS = ("AGMM_FIRST_LOCAL_MM", "AGMM_FIRST_REMOTE_MM", "AGMM_LOCAL_TX_DONE", "AGMM_GATHER_DONE")


def parse_marker_starts():
    """runs[(dev, zone)] -> sorted list of per-run earliest ZONE_START cycles across that device's cores."""
    rows = list(csv.reader(open(CSV)))
    # per (dev, core_x, core_y, zone): ordered ZONE_START cycles (one per program invocation).
    per_core = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12:
            continue
        zone = row[10].strip()
        if zone not in MARKS or row[11].strip() != "ZONE_START":
            continue
        per_core[(row[0].strip(), row[1].strip(), row[2].strip(), zone)].append(int(row[5]))
    # group by (dev, zone); align runs by invocation index and take the earliest core per run.
    by_devzone = defaultdict(dict)  # (dev,zone) -> {(x,y): [cycles per run]}
    for (dev, x, y, zone), cyc in per_core.items():
        by_devzone[(dev, zone)][(x, y)] = cyc
    out = {}
    for (dev, zone), cores in by_devzone.items():
        nruns = min(len(v) for v in cores.values())
        out[(dev, zone)] = [min(v[r] for v in cores.values()) for r in range(nruns)]
    return out


def prove_overlap():
    starts = parse_marker_starts()
    devs = sorted({d for (d, _z) in starts})
    result = {}
    for dev in devs:
        got = {m: starts.get((dev, m), []) for m in MARKS}
        nruns = min((len(v) for v in got.values()), default=0)
        if nruns == 0 or any(len(got[m]) == 0 for m in MARKS):
            result[dev] = {"error": "missing markers", "counts": {m: len(got[m]) for m in MARKS}}
            continue
        run_ids = list(range(1, nruns)) if nruns > 1 else [0]  # drop warmup run 0 when possible
        a_ok, b_ok, rows = True, True, []
        for r in run_ids:
            fl, fr = got["AGMM_FIRST_LOCAL_MM"][r], got["AGMM_FIRST_REMOTE_MM"][r]
            lt, gd = got["AGMM_LOCAL_TX_DONE"][r], got["AGMM_GATHER_DONE"][r]
            a = fl < lt
            b = fr < gd
            a_ok, b_ok = a_ok and a, b_ok and b
            rows.append(
                {
                    "run": r,
                    "first_local_before_local_tx_done": a,
                    "first_remote_before_gather_done": b,
                    "local_lead_us": round((lt - fl) / FREQ * 1e6, 3),
                    "remote_lead_us": round((gd - fr) / FREQ * 1e6, 3),
                }
            )
        result[dev] = {
            "runs_scored": len(run_ids),
            "overlap_a_all_runs": a_ok,
            "overlap_b_all_runs": b_ok,
            "per_run": rows,
        }
    return result


def main():
    D, my, mx, M, K, N = (
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
    )
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
        a = ttnn.from_torch(
            t0,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.create_mesh_mapper(
                md, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, D))
            ),
        )
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, md)
        b = ttnn.from_torch(
            t1,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            dtype=ttnn.bfloat16,
            memory_config=wcfg,
            mesh_mapper=ttnn.create_mesh_mapper(
                md, ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, D))
            ),
        )
        cfg = None  # auto-picker
        sems = [ttnn.create_global_semaphore(md, crs, 0) for _ in range(2 * D)]
        out = None
        for _ in range(12):
            for s in sems:
                ttnn.reset_global_semaphore_value(s, 0)
            out = ttnn.experimental.all_gather_regime_a_matmul_async(
                a,
                b,
                config=cfg,
                cluster_axis=1,
                topology=ttnn.Topology.Ring,
                num_links=1,
                num_workers_per_link=1,
                multi_device_global_semaphore=sems,
            )
            ttnn.synchronize_device(md)
        per = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
        _, pcc = comp_pcc(ref, per[0].float().reshape(ref.shape), 0.999)
        ttnn.ReadDeviceProfiler(md)
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
        ov = prove_overlap()
        print("MARKJSON " + json.dumps({"D": D, "shape": [M, K, N], "pcc": float(pcc), "per_device": ov}))
    finally:
        if full is not None:
            try:
                ttnn.close_mesh_device(full)
            except Exception:
                pass
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
