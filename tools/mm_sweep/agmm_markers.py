# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 3 overlap PROOF for all_gather_regime_a_matmul_async via custom device-profiler markers.
#
# The store-and-forward ring kernels emit markers on one per-device cycle timebase:
#   AGMM_MM_LOCAL   (in0 reader, DeviceTimestampedData, data=Pk band kk) - first step-0 in0 block whose gate
#                    passed for a LOCAL shard block, per band
#   AGMM_MM_REMOTE  (in0 reader, DeviceTimestampedData, data=Pk band kk) - first step-0 REMOTE-shard block, per band
#   AGMM_LOCAL_TX_DONE (relay, zone) - this device's OWN shard's last block has been forwarded to the neighbour
#   AGMM_GATHER_DONE   (relay, zone) - every shard's every block is now stored locally (all-gather complete)
#
# Overlap is PROVEN, per device, per run, PER Pk BAND, when:
#   - every band that owns local K begins a LOCAL block before its own first REMOTE block (local-first schedule),
#   - first local math  <  local_tx_done   (local compute starts before local-shard transmit finishes),
#   - first remote math <  gather_done      (remote compute starts before the full gather finishes).
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

RELAY = ("AGMM_LOCAL_TX_DONE", "AGMM_GATHER_DONE")  # DeviceZoneScopedN zones on the relay core
BANDMK = ("AGMM_MM_LOCAL", "AGMM_MM_REMOTE")  # DeviceTimestampedData(name, band_kk) on the compute readers


def _zone_starts():
    """(dev, zone) -> [earliest ZONE_START cycle per run across that device's cores]."""
    rows = list(csv.reader(open(CSV)))
    per_core = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 12 or row[11].strip() != "ZONE_START" or row[10].strip() not in RELAY:
            continue
        per_core[(row[0].strip(), row[10].strip(), row[1].strip(), row[2].strip())].append(int(row[5]))
    by = defaultdict(dict)
    for (dev, zone, x, y), cyc in per_core.items():
        by[(dev, zone)][(x, y)] = cyc
    out = {}
    for (dev, zone), cores in by.items():
        n = min(len(v) for v in cores.values())
        out[(dev, zone)] = [min(v[r] for v in cores.values()) for r in range(n)]
    return out


def _band_starts():
    """(dev, band_kk, kind) -> [earliest TS_DATA cycle per run across that band's cores]; kind in {local,remote}."""
    rows = list(csv.reader(open(CSV)))
    per_core = defaultdict(list)  # (dev, kk, kind, x, y) -> [cycles per run]
    for row in rows[2:]:
        if len(row) < 12 or row[11].strip() != "TS_DATA" or row[10].strip() not in BANDMK:
            continue
        kind = "local" if row[10].strip() == "AGMM_MM_LOCAL" else "remote"
        per_core[(row[0].strip(), int(row[6]), kind, row[1].strip(), row[2].strip())].append(int(row[5]))
    by = defaultdict(dict)
    for (dev, kk, kind, x, y), cyc in per_core.items():
        by[(dev, kk, kind)][(x, y)] = cyc
    out = {}
    for (dev, kk, kind), cores in by.items():
        n = min(len(v) for v in cores.values())
        out[(dev, kk, kind)] = [min(v[r] for v in cores.values()) for r in range(n)]
    return out


def prove_overlap():
    zones = _zone_starts()
    bands = _band_starts()
    devs = sorted({d for (d, _z) in zones} | {d for (d, _kk, _k) in bands})
    result = {}
    for dev in devs:
        lt = zones.get((dev, "AGMM_LOCAL_TX_DONE"), [])
        gd = zones.get((dev, "AGMM_GATHER_DONE"), [])
        kks = sorted({kk for (d, kk, _k) in bands if d == dev})
        counts = {"AGMM_LOCAL_TX_DONE": len(lt), "AGMM_GATHER_DONE": len(gd), "bands": kks}
        if not lt or not gd or not kks:
            result[dev] = {"error": "missing markers", "counts": counts}
            continue
        # per-band local ordering + device overlap, scored per run (drop warmup run 0 when possible).
        nruns = min([len(lt), len(gd)] + [len(v) for (d, kk, k), v in bands.items() if d == dev])
        run_ids = list(range(1, nruns)) if nruns > 1 else [0]
        per_band = {}
        dev_a_ok, dev_b_ok = True, True  # a: first-local < local_tx_done ; b: first-remote < gather_done
        band_local_first_ok = True  # every band with local ownership begins local before its own first remote
        for kk in kks:
            loc = bands.get((dev, kk, "local"), [])
            rem = bands.get((dev, kk, "remote"), [])
            has_local = len(loc) > 0
            rows = []
            for r in run_ids:
                lv = loc[r] if r < len(loc) else None
                rv = rem[r] if r < len(rem) else None
                lb = (lv is not None) and (rv is None or lv <= rv)  # band begins local before remote
                a = (lv is not None) and lv < lt[r]
                b = (rv is not None) and rv < gd[r]
                if has_local:
                    band_local_first_ok = band_local_first_ok and lb
                    dev_a_ok = dev_a_ok and a
                if rv is not None:
                    dev_b_ok = dev_b_ok and b
                rows.append(
                    {
                        "run": r,
                        "local_before_remote": lb if has_local else None,
                        "first_local_before_local_tx_done": a if has_local else None,
                        "first_remote_before_gather_done": b if rv is not None else None,
                    }
                )
            per_band[kk] = {"has_local_ownership": has_local, "per_run": rows}
        result[dev] = {
            "runs_scored": len(run_ids),
            "bands_with_local_begin_local_first": band_local_first_ok,
            "first_local_before_local_tx_done_all": dev_a_ok,
            "first_remote_before_gather_done_all": dev_b_ok,
            "per_band": per_band,
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
    os.environ["TT_AGMM_TRANSPORT"] = "ring_store_forward"  # markers only exist in the store-and-forward relay
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
