#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare two LLK perf run directories point by point and count what is not bit-exact.

A same-code control (one build, measured twice) must give identical cycles on a stable
harness. This script answers "how many points moved, and by how much", per run type and
marker. It does not judge regressions: use perf/regression_compare.py for that.

    perf_bitexact.py <run dir A> <run dir B> [--marker TILE_LOOP] [--threshold 0.02]
                     [--min-cycles 30] [--json out.json] [--list 20]

A run directory is perf_data/runs/<PERF_RUN_TAG> (or any directory with the per-module
<module>/<module>.csv reports below it). Per-worker (.gw*, .master*), .post and
.counters CSVs are skipped.

A point is identified by every column that is not a value column. Value columns are
mean(...), std(...), min(...), max(...), median(...) and TEXT_SIZE(...): if any of
them entered the key, a run with extra columns (for example min() with a run count)
would split every point and look clean. TEXT_SIZE is compared separately: a
different TEXT_SIZE means the code is not the same, so the pair is not a same-code
control.
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys

VALUE_PREFIXES = ("mean(", "std(", "min(", "max(", "median(", "TEXT_SIZE(")
SKIP_PARTS = (".gw", ".master", ".post.", ".counters.")


def report_files(root):
    out = []
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        name = os.path.basename(path)
        if any(part in name for part in SKIP_PARTS):
            continue
        out.append(path)
    return sorted(out)


def load(root):
    """{(module, run_type, marker, key...): (cycles, text_size)}"""
    points = {}
    files = report_files(root)
    if not files:
        sys.exit(f"no perf report CSVs under {root}")
    for path in files:
        module = os.path.splitext(os.path.basename(path))[0]
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if "marker" not in cols:
                continue
            keys = [
                c for c in cols if not c.startswith(VALUE_PREFIXES) and c != "marker"
            ]
            means = [c for c in cols if c.startswith("mean(")]
            for row in reader:
                ident = tuple(row[c] for c in keys)
                for m in means:
                    if row[m] in ("", "nan", "NaN"):
                        continue
                    run_type = m[len("mean(") : -1]
                    text = row.get(f"TEXT_SIZE({run_type})", "")
                    points[(module, run_type, row["marker"]) + ident] = (
                        float(row[m]),
                        text,
                    )
    return points


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", help="first run directory")
    ap.add_argument("b", help="second run directory")
    ap.add_argument("--marker", help="only this marker (e.g. TILE_LOOP); default all")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="gate rule: relative part (default 0.02)",
    )
    ap.add_argument(
        "--min-cycles",
        type=float,
        default=30.0,
        help="gate rule: absolute part (default 30)",
    )
    ap.add_argument("--json", help="write the summary here")
    ap.add_argument(
        "--list", type=int, default=10, help="print the N largest moves (default 10)"
    )
    args = ap.parse_args(argv)

    a, b = load(args.a), load(args.b)
    common = [k for k in a.keys() & b.keys() if not args.marker or k[2] == args.marker]
    stats = collections.defaultdict(
        lambda: {"points": 0, "not_bit_exact": 0, "over_gate_rule": 0, "max_pct": 0.0}
    )
    text_diff = 0
    moves = []
    for k in common:
        (va, ta), (vb, tb) = a[k], b[k]
        s = stats[(k[1], k[2])]
        s["points"] += 1
        if ta and tb and ta != tb:
            text_diff += 1
        if va == vb:
            continue
        s["not_bit_exact"] += 1
        pct = abs(vb - va) / va * 100 if va else float("inf")
        s["max_pct"] = max(s["max_pct"], pct)
        if (
            abs(vb - va) > args.min_cycles
            and abs(vb - va) / max(va, 1e-9) > args.threshold
        ):
            s["over_gate_rule"] += 1
        moves.append((pct, k, va, vb))

    total = {
        x: sum(s[x] for s in stats.values())
        for x in ("points", "not_bit_exact", "over_gate_rule")
    }
    print(f"A: {args.a}\nB: {args.b}")
    print(
        f"points only in A: {len(a.keys() - b.keys())}, only in B: {len(b.keys() - a.keys())}, in both: {len(common)}"
    )
    if text_diff:
        print(
            f"WARNING: {text_diff} points have a different TEXT_SIZE: the code differs, this is not a same-code control"
        )
    print(
        f"{'run type':<22} {'marker':<10} {'points':>8} {'not exact':>10} {'over rule':>10} {'max %':>8}  (over rule: either direction)"
    )
    for (rt, mk), s in sorted(stats.items()):
        print(
            f"{rt:<22} {mk:<10} {s['points']:>8} {s['not_bit_exact']:>10} {s['over_gate_rule']:>10} {s['max_pct']:>8.2f}"
        )
    print(
        f"{'total':<33} {total['points']:>8} {total['not_bit_exact']:>10} {total['over_gate_rule']:>10}"
    )
    for pct, k, va, vb in sorted(moves, reverse=True)[: args.list]:
        print(
            f"  {pct:7.2f}%  {va:>12.0f} -> {vb:>12.0f}  {k[0]} {k[1]} {k[2]} {'|'.join(k[3:])}"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "total": total,
                    "text_size_differs": text_diff,
                    "by_run_type_marker": {
                        f"{rt}/{mk}": s for (rt, mk), s in sorted(stats.items())
                    },
                },
                fh,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
