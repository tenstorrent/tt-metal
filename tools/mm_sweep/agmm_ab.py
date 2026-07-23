# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Task 3 A/B: interleaved multi-relaunch of the three Phase-A transports, reporting per-transport total
# device-span (median + min-max) so drift is shared across all three arms.
#
#   ring_store_forward - DRAM-staged store-and-forward ring (neighbor relay, per-kb-block overlap)
#   source_to_all      - source unicasts its whole shard to every peer (default transport)
#   full_wait          - diagnostic: source_to_all AND the in0 reader waits for the COMPLETE gather before matmul
#
# The three run interleaved within ONE process (transport_mode is a hashed op attribute read from
# TT_AGMM_TRANSPORT at invoke(), so the program cache holds all three); run index r maps to ARMS[r % 3].
# Total device-span (max zone-end - min zone-start across a device's cores per run) is the wall we compare.
#
# Usage:  TT_METAL_DEVICE_PROFILER=1 python tools/mm_sweep/agmm_ab.py <D> <mesh_y> <mesh_x> <M> <K> <N> [rounds]

import csv
import gc
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
ARMS = ("ring_store_forward", "source_to_all", "full_wait")


def parse_spans():
    """dev -> {'total': [us per run], 'injector': [us per run]}, in invocation order.
    total = max zone-end - min zone-start across the device's cores; injector = span of the fabric relay/injector
    BRISC-KERNEL (the BRISC core with no TRISC-KERNEL = the relay core)."""
    rows = list(csv.reader(open(CSV)))
    ev = defaultdict(list)  # (dev,x,y,zone) -> [(marker, cycle), ...]
    for row in rows[2:]:
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        ev[(row[0].strip(), row[1].strip(), row[2].strip(), row[10].strip())].append((row[11].strip(), int(row[5])))
    pairs = defaultdict(list)  # (dev,x,y,zone) -> [(start,end) per run]
    for k, lst in ev.items():
        st = None
        for m, c in lst:
            if m == "ZONE_START":
                st = c
            elif m == "ZONE_END" and st is not None:
                pairs[k].append((st, c))
                st = None
    devs = sorted({k[0] for k in pairs})
    trisc = defaultdict(set)  # dev -> {(x,y) with a TRISC-KERNEL = compute cores}
    for dev, x, y, zone in pairs:
        if zone == "TRISC-KERNEL":
            trisc[dev].add((x, y))
    out = {}
    for dev in devs:
        nz = [len(pairs[k]) for k in pairs if k[0] == dev]
        nruns = min(nz) if nz else 0
        total, inj = [], []
        for r in range(nruns):
            ap = [pairs[k][r] for k in pairs if k[0] == dev and len(pairs[k]) > r]
            total.append((max(e for _, e in ap) - min(s for s, _ in ap)) / FREQ * 1e6)
            ip = [
                pairs[k][r]
                for k in pairs
                if k[0] == dev and k[3] == "BRISC-KERNEL" and (k[1], k[2]) not in trisc[dev] and len(pairs[k]) > r
            ]
            if ip:  # relay core = BRISC without a TRISC; the latest-ending such core is the fabric relay
                s0, e0 = max(ip, key=lambda p: p[1])
                inj.append((e0 - s0) / FREQ * 1e6)
        out[dev] = {"total": total, "injector": inj}
    return out


def summarize(rounds_warmup):
    spans = parse_spans()
    result = {}
    for dev, sp in spans.items():
        by_arm = defaultdict(lambda: {"total": [], "injector": []})
        for r, v in enumerate(sp["total"]):
            if r // len(ARMS) < rounds_warmup:  # drop the first `rounds_warmup` full interleaved rounds
                continue
            arm = ARMS[r % len(ARMS)]
            by_arm[arm]["total"].append(v)
            if r < len(sp["injector"]):
                by_arm[arm]["injector"].append(sp["injector"][r])
        arm_stat = {}
        for arm, d in by_arm.items():
            if d["total"]:
                st = {
                    "runs": len(d["total"]),
                    "total_median_us": round(statistics.median(d["total"]), 2),
                    "total_min_us": round(min(d["total"]), 2),
                    "total_max_us": round(max(d["total"]), 2),
                    "total_all_us": [round(x, 2) for x in d["total"]],
                }
                if d["injector"]:
                    st.update(
                        {
                            "inj_median_us": round(statistics.median(d["injector"]), 2),
                            "inj_min_us": round(min(d["injector"]), 2),
                            "inj_max_us": round(max(d["injector"]), 2),
                        }
                    )
                arm_stat[arm] = st
        result[dev] = arm_stat
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
    rounds = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    warmup = 1
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
        sems = [ttnn.create_global_semaphore(md, crs, 0) for _ in range(2 * D)]
        pccs = {}
        for _ in range(rounds):
            for arm in ARMS:  # interleave the three transports every round
                os.environ["TT_AGMM_TRANSPORT"] = arm
                for s in sems:
                    ttnn.reset_global_semaphore_value(s, 0)
                out = ttnn.experimental.all_gather_regime_a_matmul_async(
                    a,
                    b,
                    config=None,
                    cluster_axis=1,
                    topology=ttnn.Topology.Ring,
                    num_links=1,
                    num_workers_per_link=1,
                    multi_device_global_semaphore=sems,
                )
                ttnn.synchronize_device(md)
                if arm not in pccs:
                    per = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                    _, pccs[arm] = comp_pcc(ref, per[0].float().reshape(ref.shape), 0.999)
                    per = None
                out = None
        ttnn.ReadDeviceProfiler(md)
        a = b = None
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
        ab = summarize(warmup)
        print(
            "ABJSON "
            + json.dumps(
                {
                    "D": D,
                    "shape": [M, K, N],
                    "rounds": rounds,
                    "warmup_rounds": warmup,
                    "pcc": {k: float(v) for k, v in pccs.items()},
                    "per_device": ab,
                }
            )
        )
    finally:
        if full is not None:
            try:
                ttnn.close_mesh_device(full)
            except Exception:
                pass
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
