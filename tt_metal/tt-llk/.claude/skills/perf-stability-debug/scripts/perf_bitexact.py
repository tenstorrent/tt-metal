#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

VALUE_PREFIXES = ("mean(", "std(", "min(", "max(", "median(", "TEXT_SIZE(")
SKIP_PARTS = (".gw", ".master", ".post.", ".counters.")
COUNTER_METRIC = re.compile(
    r"^(L1_TO_L1|UNPACK_ISOLATE|MATH_ISOLATE|PACK_ISOLATE|L1_CONGESTION)_"
)
ARCH = re.compile(r"wormhole|blackhole|quasar")


def report_files(root):
    out = []
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        name = os.path.basename(path)
        if any(part in name for part in SKIP_PARTS):
            continue
        out.append(path)
    return sorted(out)


def load(root):
    points = {}
    files = report_files(root)
    if not files:
        sys.exit(f"no perf report CSVs under {root}")
    for path in files:
        module = os.path.splitext(os.path.basename(path))[0]
        arch = ARCH.search(os.path.relpath(path, root))
        if arch:
            module = f"{arch.group(0)}/{module}"
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if "marker" not in cols:
                continue
            keys = [
                c
                for c in cols
                if not c.startswith(VALUE_PREFIXES)
                and c != "marker"
                and not COUNTER_METRIC.match(c)
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
    ap = argparse.ArgumentParser(
        description="Count LLK perf points that are not bit-exact between two run directories."
    )
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
