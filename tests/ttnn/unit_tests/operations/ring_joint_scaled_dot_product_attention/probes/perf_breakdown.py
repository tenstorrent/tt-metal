# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-tournament probe (not collected: no test_ prefix): per-chip, per-stage breakdown of ONE call of the
perf-FOCUS config exactly (Wan2.2: bf16 Q / K/V, fp32_dest_acc_en=False, HiFi2, 1x4 ring, 2 links,
logical_n = 32760), from the device profiler's KERNEL spans and the op's MaybeDeviceZoneScope zones.

  TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh tests/.../probes/perf_breakdown.py -s
  python tests/.../probes/perf_breakdown.py [out.json]     # host-side analysis (the CSV is flushed at mesh close)

Env: FOCUS_HEADS (40), FOCUS_N (8192 rows per device), FOCUS_LINKS (2), FOCUS_FP32 (0), FOCUS_FID (HiFi2; fidelity
ablation only), PERF_OUT (json path).
Prints per chip: op span (first kernel start -> last kernel end), compute-core span quantiles, port spans, and per
zone (count, mean / median ns per execution, per RISC).
"""
import json
import os

import pandas as pd
import pytest
import torch
import ttnn

from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention
from ttnn.operations.ring_joint_scaled_dot_product_attention import (
    ring_joint_scaled_dot_product_attention_program_descriptor as pdmod,
)

H = int(os.environ.get("FOCUS_HEADS", "40"))
N = int(os.environ.get("FOCUS_N", "8192"))
LINKS = int(os.environ.get("FOCUS_LINKS", "2"))
FP32 = os.environ.get("FOCUS_FP32", "0") == "1"
D = int(os.environ.get("FOCUS_DH", "128"))
JOINT_L = int(os.environ.get("FOCUS_JOINT_L", "0"))  # replicated joint tokens (0 = none)
TOPO = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}[os.environ.get("FOCUS_TOPO", "ring")]
PAD = int(os.environ.get("FOCUS_PAD", "8"))  # logical_n = G * N - PAD
REPS = int(os.environ.get("FOCUS_REPS", "1"))  # measured calls (each read separately; see spans())
FID = getattr(ttnn.MathFidelity, os.environ.get("FOCUS_FID", "HiFi2"))  # ablation only: the focus is HiFi2
FREQ_GHZ = 1.35
LOG = "generated/profiler/.logs/profile_log_device.csv"


def _analyze(chips=None):
    df = pd.read_csv(LOG, skiprows=1, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    run_col = "run host ID" if "run host ID" in df.columns else "run ID"
    df = df[df[run_col] == df.groupby("PCIe slot")[run_col].transform("max")]
    tcol = "time[cycles since reset]"
    port_cores = {}
    for coord, ch in chips.items():
        port_cores[coord] = {(c.x, c.y) for c in ch["ports"].values()}
    out = {}
    for dev, d in df.groupby("PCIe slot"):
        k = d[d["zone name"].astype(str).str.contains("KERNEL")]
        span = k.groupby(["core_x", "core_y"])[tcol].agg(["min", "max"])
        t0 = span["min"].min()
        rel = (span - t0) / FREQ_GHZ
        rel["dur"] = rel["max"] - rel["min"]
        res = {
            "op_span_ns": float(rel["max"].max()),
            "cores": int(len(rel)),
            "core_end_q_ns": rel["max"].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).round(0).tolist(),
            "core_start_q_ns": rel["min"].quantile([0.0, 0.5, 1.0]).round(0).tolist(),
        }
        z = d[~d["zone name"].astype(str).str.contains("KERNEL") & d["type"].isin(["ZONE_START", "ZONE_END"])]
        zones = {}
        for (zn, risc), g in z.groupby(["zone name", "RISC processor type"]):
            g = g.sort_values(["core_x", "core_y", tcol])
            durs, last_end = [], []
            for _, gc in g.groupby(["core_x", "core_y"]):
                st = gc[gc["type"] == "ZONE_START"][tcol].values
                en = gc[gc["type"] == "ZONE_END"][tcol].values
                n = min(len(st), len(en))
                durs += list((en[:n] - st[:n]) / FREQ_GHZ)
                if n:
                    last_end.append((en[n - 1] - t0) / FREQ_GHZ)
            if not durs:
                continue
            s = pd.Series(durs)
            zones[f"{zn}@{risc}"] = {
                "n": int(len(s)),
                "mean_ns": round(float(s.mean()), 0),
                "median_ns": round(float(s.median()), 0),
                "max_ns": round(float(s.max()), 0),
                "sum_per_core_ns": round(
                    float(s.sum()) / max(1, g[["core_x", "core_y"]].drop_duplicates().shape[0]), 0
                ),
                "last_end_max_ns": round(max(last_end), 0) if last_end else None,
            }
        # marker budget check
        cnt = z.groupby(["core_x", "core_y", "RISC processor type"]).size()
        res["max_markers_per_risc"] = int(cnt.max()) if len(cnt) else 0
        res["zones"] = zones
        out[int(dev)] = res
    return out


@pytest.mark.parametrize("mesh_shape,fabric_config", [((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)])
def test_perf_breakdown(mesh_shape, fabric_config, mesh_device):
    G = 4
    mk = lambda: ttnn.from_torch(  # noqa: E731
        torch.randn((1, H, G * N, D), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_shape),
    )
    q, k, v = mk(), mk(), mk()
    joint = [None, None, None]
    if JOINT_L:
        joint = [
            ttnn.from_torch(
                torch.randn((1, H, JOINT_L, D), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for _ in range(3)
        ]
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=FID, fp32_dest_acc_en=FP32)
    call = lambda: ring_joint_scaled_dot_product_attention(  # noqa: E731
        q,
        k,
        v,
        *joint,
        logical_n=G * N - PAD,
        cluster_axis=1,
        topology=TOPO,
        num_links=LINKS,
        compute_kernel_config=cfg,
    )
    call()  # compile
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
    for _ in range(REPS):
        call()
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
    chips, ring, G_, nl = pdmod.plan(mesh_device, cluster_axis=1, topology=TOPO, num_links=LINKS)
    for coord, ch in chips.items():
        print("PORTS", coord, {k_: (c.x, c.y) for k_, c in ch["ports"].items()}, "compute", len(ch["compute"]))


def report(res):
    for dev, r in sorted(res.items()):
        print(
            f"\nCHIP {dev}: op span {r['op_span_ns']/1e3:.1f} us, cores {r['cores']}, max markers/RISC "
            f"{r['max_markers_per_risc']}"
        )
        print("  core end quantiles (us):", [round(x / 1e3, 1) for x in r["core_end_q_ns"]])
        print("  core start quantiles (us):", [round(x / 1e3, 1) for x in r["core_start_q_ns"]])
        for zn, zs in sorted(r["zones"].items()):
            print(
                f"  {zn:45s} n={zs['n']:6d} mean={zs['mean_ns']:10.0f} med={zs['median_ns']:10.0f} "
                f"max={zs['max_ns']:10.0f} per_core_sum={zs['sum_per_core_ns']:12.0f} last_end={zs['last_end_max_ns']}"
            )


def spans():
    """Op span (first kernel start -> last kernel end, us) of every op call in the log, per chip: the runs whose
    zones include compute_unit (the ethernet-probe programs and the compile call are dropped)."""
    df = pd.read_csv(LOG, skiprows=1, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    run_col = "run host ID" if "run host ID" in df.columns else "run ID"
    tcol = "time[cycles since reset]"
    out = {}
    for dev, d in df.groupby("PCIe slot"):
        runs = []
        for r, g in d.groupby(run_col):
            if not (g["zone name"] == "compute_unit").any():
                continue
            k = g[g["zone name"].astype(str).str.contains("KERNEL")]
            runs.append((r, (k[tcol].max() - k[tcol].min()) / FREQ_GHZ / 1e3))
        out[int(dev)] = [round(x, 1) for _, x in sorted(runs)[1:]]  # drop the compile call
    return out


if __name__ == "__main__":
    import sys

    if os.environ.get("SPANS"):
        sp = spans()
        print("SPANS_US", json.dumps(sp))
        allv = [x for v in sp.values() for x in v]
        per_call_max = [max(v[i] for v in sp.values()) for i in range(min(len(v) for v in sp.values()))]
        print("PER_CALL_SLOWEST_US", per_call_max, "median", sorted(per_call_max)[len(per_call_max) // 2])
        print("PER_CALL_FASTEST_US", [min(v[i] for v in sp.values()) for i in range(len(per_call_max))])
        sys.exit(0)
    res = _analyze({})
    report(res)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump({str(k_): v_ for k_, v_ in res.items()}, f, indent=1)
