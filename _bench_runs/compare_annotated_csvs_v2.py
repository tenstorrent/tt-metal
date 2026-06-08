#!/usr/bin/env python3
"""Compare per-stage kernel breakdowns across multiple v2-annotated tracy CSVs.

Reads CSVs annotated by annotate_ops_csv_v2.py — multi-inference tracy outputs.
Picks the LAST inference (trace replay) as canonical and reports init separately.

Usage:
    python _bench_runs/compare_annotated_csvs_v2.py \\
        --labels=cam=1,cam=2,cam=3 \\
        cam1_annotated_v2.csv cam2_annotated_v2.csv cam3_annotated_v2.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def per_stage(path: str) -> dict:
    """Sum kernel ms by STAGE for the canonical inference (last/replay).
    Init is labeled 'init_one_time'. Earlier inferences (warmup, capture) are
    suffix-tagged and skipped."""
    rows = list(csv.reader(open(path)))
    header, body = rows[0], rows[1:]
    stage_col_idx = 0
    kd_col_idx = header.index("DEVICE KERNEL DURATION [ns]")
    sums = {}
    for r in body:
        st = r[stage_col_idx]
        if st == "" or "_warmup" in st or "_trace_capture" in st:
            continue
        sums[st] = sums.get(st, 0) + float(r[kd_col_idx] or 0)
    return {k: v / 1e6 for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="v2-annotated CSV files")
    ap.add_argument("--labels", help="Comma-separated labels, e.g. --labels=cam=1,cam=2,cam=3")
    args = ap.parse_args()

    csvs = args.csvs
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(csvs):
            ap.error(f"--labels has {len(labels)} entries but {len(csvs)} CSVs were given")
    else:
        labels = [Path(c).stem for c in csvs]

    results = [(lbl, per_stage(p)) for lbl, p in zip(labels, csvs)]

    stages = [
        "init_one_time",
        "prefix_setup",
        "siglip",
        "vlm_prefill",
        "denoise_step_1",
        "denoise_step_2",
        "denoise_step_3",
        "denoise_step_4",
        "denoise_step_5",
        "project_output",
    ]

    print()
    print("=" * 96)
    print("  PI0.5 STAGE BREAKDOWN — side-by-side (trace replay, init separated)")
    print("=" * 96)
    print()
    header_line = f"  {'STAGE':<20}"
    for lbl, _ in results:
        header_line += f"  {lbl:>14}"
    delta_label = f"Δ({results[0][0]}→{results[-1][0]})" if len(results) >= 2 else ""
    header_line += f"  {delta_label:>18}"
    print(header_line)
    print("  " + "-" * 92)

    canon_totals = [0.0] * len(results)  # per-inference total (excludes init)
    for stage in stages:
        vals = [r[1].get(stage, 0.0) for r in results]
        line = f"  {stage:<20}"
        for ms in vals:
            line += f"  {ms:>11.3f} ms"
        if len(vals) >= 2:
            delta = vals[-1] - vals[0]
            sign = "+" if delta >= 0 else ""
            line += f"  {sign}{delta:>11.3f} ms"
        if stage == "init_one_time":
            line += "  ← runs ONCE at model load"
        print(line)
        if stage == "init_one_time":
            print("  " + "-" * 92)
        for i, v in enumerate(vals):
            if stage != "init_one_time":
                canon_totals[i] += v

    print("  " + "-" * 92)
    line = f"  {'TOTAL/inference':<20}"
    for t in canon_totals:
        line += f"  {t:>11.3f} ms"
    if len(canon_totals) >= 2:
        delta = canon_totals[-1] - canon_totals[0]
        sign = "+" if delta >= 0 else ""
        line += f"  {sign}{delta:>11.3f} ms"
    line += "  ← EXCLUDES init"
    print(line)
    print("  " + "-" * 92)
    print()


if __name__ == "__main__":
    main()
