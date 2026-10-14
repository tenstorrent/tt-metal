# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarize matmul_chunk_sweep.py profiler runs: nanoseconds per computed token versus chunk size.

    python models/demos/gemma4_d_p/scripts/matmul_chunk_sweep_report.py <run_dir> [<run_dir> ...] --out <dir>

Each run_dir is a tracy -o directory holding meta.jsonl. Writes raw.csv (one row per timed call), summary.csv (one row
per variant: median and spread of device kernel time, ns per computed token, FPU utilization, weight bandwidth) and
ns_per_token.png. Repeated runs of the same variant (same tag) are pooled; run-to-run spread is reported separately.

Utilization is ideal FPU cycles over measured cycles: M*K*N / 32^3 matmul tiles at 16 cycles per tile for LoFi and 32
for HiFi2 (tests/ttnn/unit_tests/benchmarks/test_benchmark.py), at 1.35 GHz, over the full 120-core grid and over the
cores the op used.
"""

import argparse
import csv
import glob
import json
import re
import statistics as st
from collections import defaultdict

FREQ_GHZ = 1.35
CYCLES_PER_TILE = {"LoFi": 16, "HiFi2": 32, "HiFi3": 48, "HiFi4": 64}
FULL_GRID = 120
BFP8_BYTES_PER_ELEM = 1088 / 1024  # bfp8 tile: 1024 mantissa bytes + 64 exponent bytes


def read_run(run_dir):
    """{tag: [(op_code, cores, fidelity, kernel_ns), ...]} for the timed calls of run_dir."""
    path = sorted(glob.glob(f"{run_dir}/reports/*/ops_perf_results_*.csv"))[-1]
    calls, cur = defaultdict(list), None
    for r in csv.DictReader(open(path)):
        if r["OP TYPE"] == "signpost":
            m = re.match(r"v-(.*)-(start|stop)$", r["OP CODE"])
            cur = m.group(1) if m and m.group(2) == "start" else None
        elif cur and r["DEVICE KERNEL DURATION [ns]"]:
            calls[cur].append(
                (r["OP CODE"], int(r["CORE COUNT"]), r["MATH FIDELITY"], float(r["DEVICE KERNEL DURATION [ns]"]))
            )
    return calls


def quantile(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    meta, times, per_run = {}, defaultdict(list), defaultdict(list)
    prepare = defaultdict(list)
    raw = []
    for run in args.runs:
        for line in open(f"{run}/meta.jsonl"):
            rec = json.loads(line)
            meta.setdefault(rec["tag"], rec)
        for tag, calls in read_run(run).items():
            for op_code, cores, fid, ns in calls:
                raw.append(dict(run=run, tag=tag, op_code=op_code, cores=cores, math_fidelity=fid, kernel_ns=ns))
            if tag.endswith("-prepare"):
                prepare[tag[: -len("-prepare")]] += [c[3] for c in calls]
                continue
            mm = [c for c in calls if c[0].startswith("Matmul")]
            times[tag] += mm
            if mm:
                per_run[tag].append(st.median(c[3] for c in mm))

    rows = []
    for tag, rec in meta.items():
        row = {k: rec.get(k) for k in ("op", "chunk", "rows", "padded_rows", "k", "n", "attn_fidelity", "status")}
        row.update(pcc=rec.get("pcc"), rel_rmse=rec.get("rel_rmse"), tag=tag)
        cc = rec.get("compute_config") or {}
        row["fp32_dest_acc"] = cc.get("fp32_dest_acc_en", "default")
        row["packer_l1_acc"] = cc.get("packer_l1_acc", "default")
        pc = rec.get("program_config") or "ttnn default (core_grid / auto)"
        row["program_config"] = pc.split("(")[0] if pc else pc
        row["program_config_full"] = pc
        mm = times.get(tag, [])
        if mm:
            ns = [c[3] for c in mm]
            med = st.median(ns)
            cores, fid = mm[0][1], mm[0][2]
            m, k, n = row["padded_rows"], row["k"], row["n"]
            ideal_cycles = m * k * n / 32**3 * CYCLES_PER_TILE.get(fid, float("nan"))
            weight_bytes = k * n * BFP8_BYTES_PER_ELEM
            row.update(
                reps=len(ns),
                cores=cores,
                math_fidelity=fid,
                median_ns=med,
                p25_ns=quantile(ns, 0.25),
                p75_ns=quantile(ns, 0.75),
                min_ns=min(ns),
                max_ns=max(ns),
                spread_pct=100 * (quantile(ns, 0.75) - quantile(ns, 0.25)) / med,
                run_to_run_pct=(100 * (max(per_run[tag]) - min(per_run[tag])) / med) if len(per_run[tag]) > 1 else None,
                ns_per_token=med / row["rows"],
                ns_per_token_padded=med / m,
                util_full_grid_pct=100 * ideal_cycles / FULL_GRID / (med * FREQ_GHZ),
                util_used_cores_pct=100 * ideal_cycles / cores / (med * FREQ_GHZ),
                weight_gbps=weight_bytes / med,
                tflops=2 * m * k * n / med / 1e3,
            )
        if prepare.get(tag):
            row["prepare_median_ns"] = st.median(prepare[tag])
        rows.append(row)
    rows.sort(key=lambda r: (r["op"], r["attn_fidelity"], r["chunk"]))

    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(f"{args.out}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(f"{args.out}/raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0]))
        w.writeheader()
        w.writerows(raw)
    plot(rows, f"{args.out}/ns_per_token.png")
    for r in rows:
        if r.get("median_ns"):
            print(
                f"{r['op']:12s} {r['attn_fidelity']:5s} C={r['chunk']:6d} rows={r['rows']:5d} {r['math_fidelity']:5s} "
                f"{r['cores']:3d}c {r['median_ns'] / 1e3:8.1f} us  {r['ns_per_token']:7.2f} ns/tok  "
                f"util {r['util_full_grid_pct']:5.1f}%  W {r['weight_gbps']:6.1f} GB/s  iqr {r['spread_pct']:4.1f}%  "
                f"pcc {r['pcc']:.5f}  {r['program_config']}"
            )
        else:
            print(f"{r['op']:12s} {r['attn_fidelity']:5s} C={r['chunk']:6d} {r['status']}")


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = defaultdict(list)
    for r in rows:
        if r.get("median_ns"):
            series[(r["op"], r["attn_fidelity"])].append(r)
    ops = sorted({op for op, _ in series})
    fig, axes = plt.subplots(len(ops), 2, figsize=(13, 3.2 * len(ops)), squeeze=False)
    for i, op in enumerate(ops):
        for (o, fid), pts in sorted(series.items()):
            if o != op:
                continue
            pts.sort(key=lambda r: r["chunk"])
            c = [r["chunk"] for r in pts]
            label = f"attn fidelity {fid}" if not op.startswith("mlp") else "production (LoFi)"
            err = [
                [r["ns_per_token"] - r["min_ns"] / r["rows"] for r in pts],
                [r["max_ns"] / r["rows"] - r["ns_per_token"] for r in pts],
            ]
            axes[i][0].errorbar(c, [r["ns_per_token"] for r in pts], yerr=err, marker="o", ms=3, capsize=2, label=label)
            axes[i][1].plot(c, [r["util_full_grid_pct"] for r in pts], marker="o", ms=3, label=label)
        for ax, ylabel in zip(axes[i], ("ns per computed token", "FPU util, 120 cores (%)")):
            ax.set_xscale("log", base=2)
            ax.set_title(op)
            ax.set_xlabel("chunk size C (tokens; C/8 rows per device)")
            ax.set_ylabel(ylabel)
            ax.axvspan(min(r["chunk"] for r in rows) / 1.2, 1024 / 1.1, color="0.9", zorder=0)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        axes[i][0].set_yscale("log")
    fig.suptitle("Gemma4-31B prefill projections, one BH device of CP8/TP4 (shaded: C < 1024, not a valid model chunk)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
