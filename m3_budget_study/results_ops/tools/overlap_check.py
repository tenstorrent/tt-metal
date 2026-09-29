#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Did the shared expert run concurrently with dispatch? Per MoE layer of the profiled forward, from a tracy ops
CSV (DEVICE FW START / END CYCLE per op and device):

  dispatch   span of the ops in <layer>/mlp/dispatch (minus its dispatch_v2_prep sub-zone)
  shared     span of the ops in <layer>/mlp/shared_expert (minus tp_reduce_scatter / tp_allreduce)
  window     span of both; overlap = the two spans' intersection; saved = dispatch + shared - window

Spans are wall time on the chip (first start to last end), not summed kernel time. Cycles are converted with the
clock implied by the ops' own DEVICE FW DURATION [ns]. One row per layer: the median over devices and the worst
device (largest window). Also lists the SUB DEVICE IDs seen per zone.

  overlap_check.py <ops_perf_results_*.csv> [--json out.json]
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)
ROOT = "profiled_chunk"
EXCLUDE = {"dispatch": ("dispatch_v2_prep",), "shared_expert": ("tp_reduce_scatter", "tp_allreduce")}


def spans(path):
    """{(layer, zone, device): [start_cycle, end_cycle]}, {(zone): set(sub-device ids)}, ns per cycle."""
    stack, out, sds = [], {}, defaultdict(set)
    ns_per_cycle = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code = row.get("OP CODE") or ""
            if row.get("OP TYPE") == "signpost":
                if code.startswith("M3_ZONE_START"):
                    stack.append(code.split(None, 1)[1].strip())
                elif code.startswith("M3_ZONE_END"):
                    end = code.split(None, 1)[1].strip()
                    while stack and stack.pop() != end:
                        pass
                continue
            if not stack or stack[0] != ROOT or len(stack) < 4 or stack[2] != "mlp":
                continue
            layer, zone, sub = stack[1], stack[3], stack[4:]
            if zone not in EXCLUDE or (sub and sub[0] in EXCLUDE[zone]):
                continue
            try:
                s, e, dev = int(row["DEVICE FW START CYCLE"]), int(row["DEVICE FW END CYCLE"]), int(row["DEVICE ID"])
            except (KeyError, TypeError, ValueError):
                continue
            key = (layer, zone, dev)
            if key in out:
                out[key][0], out[key][1] = min(out[key][0], s), max(out[key][1], e)
            else:
                out[key] = [s, e]
            sds[zone].add(row.get("SUB DEVICE ID") or "-")
            try:
                ns = float(row["DEVICE FW DURATION [ns]"])
                if e > s and ns > 0:
                    ns_per_cycle.append(ns / (e - s))
            except (KeyError, TypeError, ValueError):
                pass
    return out, sds, (statistics.median(ns_per_cycle) if ns_per_cycle else 1 / 1.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json")
    args = ap.parse_args()
    sp, sds, npc = spans(args.csv)
    layers = sorted({k[0] for k in sp})
    rows = []
    print(
        f"{args.csv}\n  clock {1 / npc:.3f} GHz, sub-device ids: "
        + ", ".join(f"{z}={sorted(v)}" for z, v in sds.items())
    )
    print(
        f"  {'layer':16s} {'dispatch_us':>11s} {'shared_us':>9s} {'window_us':>9s} {'overlap_us':>10s} "
        f"{'saved_us':>8s} {'overlap%':>8s}   worst-device window_us"
    )
    for layer in layers:
        per_dev = []
        for dev in sorted({k[2] for k in sp if k[0] == layer}):
            d, s = sp.get((layer, "dispatch", dev)), sp.get((layer, "shared_expert", dev))
            if not d or not s:
                continue
            dd, ss = (d[1] - d[0]) * npc / 1e3, (s[1] - s[0]) * npc / 1e3
            win = (max(d[1], s[1]) - min(d[0], s[0])) * npc / 1e3
            ov = max(0, min(d[1], s[1]) - max(d[0], s[0])) * npc / 1e3
            per_dev.append(
                {
                    "dev": dev,
                    "dispatch_us": dd,
                    "shared_us": ss,
                    "window_us": win,
                    "overlap_us": ov,
                    "saved_us": dd + ss - win,
                }
            )
        if not per_dev:
            continue
        med = {k: statistics.median(p[k] for p in per_dev) for k in per_dev[0] if k != "dev"}
        worst = max(per_dev, key=lambda p: p["window_us"])
        pct = 100 * med["overlap_us"] / max(1e-9, min(med["dispatch_us"], med["shared_us"]))
        print(
            f"  {layer:16s} {med['dispatch_us']:11.1f} {med['shared_us']:9.1f} {med['window_us']:9.1f} "
            f"{med['overlap_us']:10.1f} {med['saved_us']:8.1f} {pct:7.0f}%   dev{worst['dev']} {worst['window_us']:.1f}"
        )
        rows.append({"layer": layer, "median": med, "overlap_pct_of_shorter": pct, "worst": worst, "devices": per_dev})
    if not rows:
        print("  no MoE layer with both a dispatch and a shared_expert zone under profiled_chunk")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "csv": args.csv,
                    "ghz": 1 / npc,
                    "sub_device_ids": {z: sorted(v) for z, v in sds.items()},
                    "layers": rows,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
