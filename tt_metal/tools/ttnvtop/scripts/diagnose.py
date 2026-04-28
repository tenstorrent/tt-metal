#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Diagnose why ttnvtop and the device profiler diverge.
# Run after compare.py if the verdict was FAIL, to characterize the gap:
#   - Are missed ops short or long?
#   - Are they many-core or few-core?
#   - Are they per-dispatch frequent or rare?
#   - Do "ttnvtop-only" ops correspond to fabric/setup programs?
#
# Usage:
#   python diagnose.py --profiler ... --ttnvtop ... --aiclk-mhz 1000

import argparse
import csv
import sys
from collections import defaultdict

# Import the parser logic directly. compare.py is in the same dir.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare import parse_profiler, parse_ttnvtop  # noqa: E402


def histogram(values, buckets):
    """Bucket a list of values into named ranges. buckets = [(label, lo, hi), ...]"""
    counts = {label: 0 for label, _, _ in buckets}
    counts["other"] = 0
    for v in values:
        placed = False
        for label, lo, hi in buckets:
            if lo <= v < hi:
                counts[label] += 1
                placed = True
                break
        if not placed:
            counts["other"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiler", required=True)
    ap.add_argument("--ttnvtop", required=True)
    ap.add_argument("--aiclk-mhz", type=int, default=1000)
    args = ap.parse_args()

    print(f"parsing profiler: {args.profiler}", file=sys.stderr)
    prof = parse_profiler(args.profiler)
    print(f"parsing ttnvtop:  {args.ttnvtop}", file=sys.stderr)
    tt, period_us = parse_ttnvtop(args.ttnvtop)

    cycles_per_us = args.aiclk_mhz
    prof_ids = set(prof.keys())
    tt_ids = set(tt.keys())
    caught = prof_ids & tt_ids
    missed = prof_ids - caught
    extra = tt_ids - prof_ids

    print()
    print("=" * 70)
    print("  DISCREPANCY DIAGNOSIS")
    print("=" * 70)
    print(f"  profiler unique:  {len(prof_ids):4}")
    print(f"  ttnvtop unique:   {len(tt_ids):4}")
    print(f"  joined:           {len(caught):4}")
    print(f"  profiler-only:    {len(missed):4}  (missed by ttnvtop)")
    print(f"  ttnvtop-only:     {len(extra):4}  (no profiler kernel-zones)")

    # ─── Missed ops: distribution by core-time ───────────────────────────
    # Each profiler entry's "cycles" is total core-cycles spent (sum over
    # cores × dispatches). Bucket missed ops by that total to see if we miss
    # only short/rare ops or also long ones.
    print()
    print(f"── Profiler-only ops by total core-time ──────────────────────────")
    print(f"  (these are ops the profiler captured but ttnvtop never sampled)")
    print()
    missed_us = [prof[i]["cycles"] / cycles_per_us for i in missed]
    buckets = [
        ("<1 ms", 0, 1_000),
        ("1-10 ms", 1_000, 10_000),
        ("10-100 ms", 10_000, 100_000),
        ("100-1000 ms", 100_000, 1_000_000),
        (">=1000 ms", 1_000_000, 1e15),
    ]
    h = histogram(missed_us, buckets)
    for label, _, _ in buckets:
        print(f"    {label:<14}  {h[label]:4}")
    if h["other"]:
        print(f"    other         {h['other']:4}")

    # Same broken down by dispatch-event count (how many times the op fired).
    print()
    print(f"── Profiler-only ops by event count (dispatch markers) ──────────")
    events = [prof[i]["events"] for i in missed]
    h2 = histogram(
        events,
        [
            ("1 event", 1, 2),
            ("2-10", 2, 11),
            ("11-100", 11, 101),
            ("101-1000", 101, 1001),
            (">=1000", 1001, 1e9),
        ],
    )
    for label, _, _ in [
        ("1 event", 1, 2),
        ("2-10", 2, 11),
        ("11-100", 11, 101),
        ("101-1000", 101, 1001),
        (">=1000", 1001, 1e9),
    ]:
        print(f"    {label:<14}  {h2[label]:4}")

    # ─── ttnvtop-only ops: are they consistently brief? ───────────────────
    print()
    print(f"── ttnvtop-only ops (no profiler kernel-zone match) ─────────────")
    print(f"  Likely setup/fabric programs the profiler doesn't instrument.")
    print(f"  (BRISC-FW boot zones, ERISC routing, dispatcher kernels)")
    print()
    if extra:
        print(f"    {'prog_id':>8}  {'frames':>7}  {'core_frames':>12}  {'first_t_us':>11}")
        for i in sorted(extra, key=lambda i: -tt[i]["core_frames"])[:20]:
            print(f"    {i:>8}  {tt[i]['frames']:>7}  {tt[i]['core_frames']:>12}  {tt[i]['first_t_us']:>11}")

    # ─── Scaling check on caught ops ─────────────────────────────────────
    # For caught ops, plot the ratio prof_us / tt_us. If ttnvtop systematically
    # underestimates, ratio > 1; overestimates, < 1. Big spread → caught ops
    # have unreliable time attribution.
    print()
    print(f"── Caught ops: ttnvtop-vs-profiler ratio distribution ────────────")
    if caught:
        ratios = []
        for i in caught:
            prof_core_us = prof[i]["cycles"] / cycles_per_us
            tt_core_us = tt[i]["core_frames"] * period_us
            if prof_core_us > 0 and tt_core_us > 0:
                ratios.append(tt_core_us / prof_core_us)
        ratios.sort()
        n = len(ratios)
        if n > 0:

            def pct(p):
                return ratios[min(n - 1, int(p * n))]

            print(f"    p10 = {pct(0.10):6.2f}    p50 = {pct(0.50):6.2f}    p90 = {pct(0.90):6.2f}")
            print(f"    min = {ratios[0]:6.2f}    max = {ratios[-1]:6.2f}")
            print(f"    (1.0 = perfect; <1 ttnvtop underestimates; >1 overestimates)")
            print()

            # Worst offenders
            extremes = sorted(
                (
                    (i, tt[i]["core_frames"] * period_us / max(1, prof[i]["cycles"] / cycles_per_us))
                    for i in caught
                    if prof[i]["cycles"] > 0 and tt[i]["core_frames"] > 0
                ),
                key=lambda x: x[1],
            )
            print(f"    Most underestimated ops (ttnvtop saw too few core-frames):")
            print(f"      {'prog_id':>8}  {'prof_us':>10}  {'tt_us':>8}  {'ratio':>6}  {'events':>6}")
            for i, ratio in extremes[:5]:
                pus = prof[i]["cycles"] / cycles_per_us
                tus = tt[i]["core_frames"] * period_us
                print(f"      {i:>8}  {pus:>10.0f}  {tus:>8.0f}  {ratio:>6.2f}  {prof[i]['events']:>6}")
            print()
            print(f"    Most overestimated ops (ttnvtop saw too many core-frames):")
            print(f"      {'prog_id':>8}  {'prof_us':>10}  {'tt_us':>8}  {'ratio':>6}  {'events':>6}")
            for i, ratio in extremes[-5:][::-1]:
                pus = prof[i]["cycles"] / cycles_per_us
                tus = tt[i]["core_frames"] * period_us
                print(f"      {i:>8}  {pus:>10.0f}  {tus:>8.0f}  {ratio:>6.2f}  {prof[i]['events']:>6}")

    # ─── Top time-by-id from each side ────────────────────────────────────
    print()
    print(f"── Top 10 ops by profiler core-time ──────────────────────────────")
    print(f"  {'prog_id':>8}  {'core_us':>10}  {'events':>6}  caught?")
    for i in sorted(prof_ids, key=lambda i: -prof[i]["cycles"])[:10]:
        us = prof[i]["cycles"] / cycles_per_us
        flag = "yes" if i in caught else "NO"
        print(f"  {i:>8}  {us:>10.0f}  {prof[i]['events']:>6}  {flag}")

    print()
    print(f"── Top 10 ops by ttnvtop core-frames ─────────────────────────────")
    print(f"  {'prog_id':>8}  {'core_frames':>12}  {'frames':>7}  in profiler?")
    for i in sorted(tt_ids, key=lambda i: -tt[i]["core_frames"])[:10]:
        flag = "yes" if i in caught else "NO"
        print(f"  {i:>8}  {tt[i]['core_frames']:>12}  {tt[i]['frames']:>7}  {flag}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("  If profiler-only ops are mostly <10 ms with 1-2 events:")
    print("    → ttnvtop misses brief, rare programs at 100 Hz host poll.")
    print("      Expected limitation. v1 (on-chip sampler) would catch these.")
    print()
    print("  If profiler-only ops include >100 ms ones with many events:")
    print("    → ttnvtop is missing programs it should easily see. Bug.")
    print("      Possible causes:")
    print("        - encoding mismatch on subset of dispatch paths (e.g.,")
    print("          dispatch.cpp uses raw runtime_id, mesh uses encoded;")
    print("          our >>10 decode breaks raw)")
    print("        - record.py rate < collector publish rate (samples lost)")
    print("        - multi-chip programs encoded differently per device")
    print()
    print("  If caught-ops ratio is wide (p10 << 1, p90 >> 1):")
    print("    → ttnvtop's per-op time attribution is noisy at sample rate.")
    print("      Spatially-broad programs get tracked well; narrow ones poorly.")
    print()
    print("  If ttnvtop-only IDs all have low core_frames (1-10):")
    print("    → They are indeed setup/fabric programs without TRISC-KERNEL")
    print("      zones in the profiler. Benign.")


if __name__ == "__main__":
    main()
