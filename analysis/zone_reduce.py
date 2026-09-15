# SPDX-License-Identifier: Apache-2.0
"""Reduce a tracy profile_log_device.csv from analysis/zone_sweep.py into per-run tables.

Per run host ID:
  wall_dev_cycles  : device wall = max(KERNEL zone end) - min(KERNEL zone start) over all cores/RISCs
  wall core        : the core whose KERNEL zone ends last (the wall-setting core)
  per-core, per-RISC KERNEL spans
  TS_DATA records  : custom accumulate-zone sums (name) and occurrence counts (name_N), per core/RISC
  ZONE_TOTAL       : native DeviceZoneScopedSumN totals, per core/RISC
  raw zones        : DeviceZoneScopedN start/end pairs (name, duration) per core/RISC
  perf counters    : 9090 records via analysis/post_process.load_counters (if present)

Usage: python analysis/zone_reduce.py <csv> [--out <prefix>]
  writes <prefix>_runs.csv (one row per run), <prefix>_cores.csv (per run/core/RISC spans + zone sums),
  <prefix>_raw.csv (per raw zone instance), <prefix>_counters.csv (per run/core counters, if any).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from post_process import load_counters  # noqa: E402

PERF_ID = 9090
PERF_REF_ID = 9091


def load(path: Path):
    zones = []  # KERNEL + raw zones
    data = []  # TS_DATA / ZONE_TOTAL
    with open(path) as f:
        f.readline()
        header = f.readline().strip().split(", ")
        ix = {c: i for i, c in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split(",", len(header) - 1)
            if len(p) < len(header):
                continue
            try:
                tid = int(p[ix["timer_id"]])
            except ValueError:
                continue
            if tid in (PERF_ID, PERF_REF_ID):
                continue
            typ = p[ix["type"]].strip()
            rec = {
                "core_x": int(p[ix["core_x"]]),
                "core_y": int(p[ix["core_y"]]),
                "risc": p[ix["RISC processor type"]],
                "time": int(p[ix["time[cycles since reset]"]]),
                "run_id": int(p[ix["run host ID"]]) if p[ix["run host ID"]] else -1,
                "zone": p[ix["zone name"]],
                "type": typ,
                "line": p[ix["source line"]],
            }
            if typ in ("ZONE_START", "ZONE_END"):
                zones.append(rec)
            elif typ in ("TS_DATA", "ZONE_TOTAL"):
                rec["data"] = int(p[ix["data"]]) if p[ix["data"]] else 0
                data.append(rec)
    return pd.DataFrame(zones), pd.DataFrame(data)


def pair_zones(z: pd.DataFrame) -> pd.DataFrame:
    """Pair ZONE_START/ZONE_END per (run, core, risc, zone) in order -> durations."""
    out = []
    for key, g in z.groupby(["run_id", "core_x", "core_y", "risc", "zone"], sort=False):
        g = g.sort_values("time")
        stack = []
        for _, r in g.iterrows():
            if r["type"] == "ZONE_START":
                stack.append(r["time"])
            elif stack:
                t0 = stack.pop()
                out.append(
                    dict(
                        run_id=key[0],
                        core_x=key[1],
                        core_y=key[2],
                        risc=key[3],
                        zone=key[4],
                        start=t0,
                        end=r["time"],
                        dur=r["time"] - t0,
                    )
                )
    return pd.DataFrame(out)


def reduce(path: Path):
    z, d = load(path)
    if z.empty:
        raise SystemExit(f"no zone rows in {path}")
    kern = z[z["zone"].str.contains("KERNEL")]
    pairs = pair_zones(z)
    kp = pairs[pairs["zone"].str.contains("KERNEL")]
    raw = pairs  # includes the KERNEL pair per RISC so absolute kernel start/end are available
    runs, cores = [], []
    for i, rid in enumerate(sorted(kern["run_id"].unique())):
        k = kern[kern["run_id"] == rid]
        t_min = k[k["type"] == "ZONE_START"]["time"].min()
        t_max = k[k["type"] == "ZONE_END"]["time"].max()
        last = k[(k["type"] == "ZONE_END") & (k["time"] == t_max)].iloc[0]
        kpr = kp[kp["run_id"] == rid]
        t1 = kpr[kpr["risc"] == "TRISC_1"]["dur"]
        rec = dict(
            file=path.name,
            run_idx=i,
            run_id=rid,
            wall_dev_cycles=int(t_max - t_min),
            wall_core_x=int(last["core_x"]),
            wall_core_y=int(last["core_y"]),
            wall_core_risc=last["risc"],
            n_cores=int(kpr.groupby(["core_x", "core_y"]).ngroups),
            trisc1_mean=float(t1.mean()) if len(t1) else float("nan"),
            trisc1_median=float(t1.median()) if len(t1) else float("nan"),
            trisc1_max=float(t1.max()) if len(t1) else float("nan"),
            kernel_start_skew=int(k[k["type"] == "ZONE_START"]["time"].max() - t_min),
        )
        runs.append(rec)
        # per core/risc rows
        dd = d[d["run_id"] == rid] if len(d) else d
        for (cx, cy, risc), g in kpr.groupby(["core_x", "core_y", "risc"]):
            row = dict(
                run_idx=i,
                run_id=rid,
                core_x=cx,
                core_y=cy,
                risc=risc,
                kernel_start=int(g["start"].iloc[0] - t_min),
                kernel_end=int(g["end"].iloc[0] - t_min),
                kernel_dur=int(g["dur"].iloc[0]),
                is_wall_core=int(cx == rec["wall_core_x"] and cy == rec["wall_core_y"]),
            )
            if len(dd):
                sub = dd[(dd["core_x"] == cx) & (dd["core_y"] == cy) & (dd["risc"] == risc)]
                for _, r in sub.iterrows():
                    name = r["zone"] if r["type"] == "TS_DATA" else "SUM:" + r["zone"]
                    row[name] = row.get(name, 0) + int(r["data"])
            cores.append(row)
    runs = pd.DataFrame(runs)
    cores = pd.DataFrame(cores)
    cnt = None
    try:
        cnt = load_counters(path)
    except Exception:
        cnt = None
    return runs, cores, raw, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = Path(a.csv)
    runs, cores, raw, cnt = reduce(path)
    pd.set_option("display.width", 250)
    print(runs.to_string())
    if a.out:
        runs.to_csv(a.out + "_runs.csv", index=False)
        cores.to_csv(a.out + "_cores.csv", index=False)
        raw.to_csv(a.out + "_raw.csv", index=False)
        if cnt is not None and len(cnt):
            cnt.to_csv(a.out + "_counters.csv", index=False)
        print("wrote", a.out + "_{runs,cores,raw,counters}.csv")


if __name__ == "__main__":
    main()
