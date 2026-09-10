#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate a tt-perf-report CSV by op code (+ matmul shape): count, device ms, share, mean DRAM % / FLOPs %, dtypes.

    $MM3_PY scripts/perf_report_by_op.py doc/optimize/tracy/<run>/perf_report.csv [--top 30] [--divide N]
"""
import argparse
import collections
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument(
        "--divide", type=float, default=1.0, help="divide counts/times by this (e.g. number of steps in the window)"
    )
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    agg = collections.defaultdict(
        lambda: {"n": 0, "us": 0.0, "dram": [], "flops": [], "dt": set(), "fid": set(), "cores": set(), "blk": set()}
    )
    for r in rows:
        k = r["OP Code"]
        g = agg[k]
        g["n"] += 1
        g["us"] += float(r["Device Time"] or 0)
        if r.get("DRAM %"):
            g["dram"].append(float(r["DRAM %"]))
        if r.get("FLOPs %"):
            g["flops"].append(float(r["FLOPs %"]))
        g["dt"].add(f"{r.get('Input 0 Datatype','')}/{r.get('Input 1 Datatype','')}->{r.get('Output Datatype','')}")
        g["fid"].add(r.get("Math Fidelity", ""))
        g["cores"].add(r.get("Cores", ""))
        g["blk"].add(
            f"k{r.get('Inner Dim Block Size','')}/s{r.get('Output Subblock H','')}x{r.get('Output Subblock W','')}"
        )
    tot = sum(g["us"] for g in agg.values())
    print(f"{'op':58s} {'n':>6s} {'ms':>8s} {'%':>5s} {'DRAM%':>6s} {'FLOP%':>6s}  dtypes fidelity cores blocks")
    for k, g in sorted(agg.items(), key=lambda kv: -kv[1]["us"])[: a.top]:
        dram = sum(g["dram"]) / len(g["dram"]) if g["dram"] else float("nan")
        fl = sum(g["flops"]) / len(g["flops"]) if g["flops"] else float("nan")
        print(
            f"{k[:58]:58s} {g['n']/a.divide:6.0f} {g['us']/1000/a.divide:8.2f} {100*g['us']/tot:5.1f} {dram:6.1f} {fl:6.1f}  "
            f"{','.join(sorted(g['dt']))} {','.join(sorted(g['fid']))} {','.join(sorted(g['cores']))} {','.join(sorted(g['blk']))}"
        )
    print(f"total device ms {tot/1000/a.divide:.2f} over {len(rows)/a.divide:.0f} ops")


if __name__ == "__main__":
    main()
