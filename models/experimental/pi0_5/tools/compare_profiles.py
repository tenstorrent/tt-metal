# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare two tt-metal op profiling reports (ops_perf_results_*.csv).

The OLDEST report is the REFERENCE, the NEWEST is the LATEST. The script reports
which ops got faster and which got slower, using DEVICE KERNEL DURATION [ns].

It compares along two axes:
  1. Per op-code aggregate (total + mean device-kernel time, call count) -- robust to
     small differences in the number of dispatched ops between runs.
  2. Positional (per-instance) alignment when the two runs have the same op sequence --
     shows the delta for each individual op call.

Usage:
    python compare_profiles.py REFERENCE.csv LATEST.csv [--metric COLNAME] [--top N] [--csv OUT.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field

METRIC_DEFAULT = "DEVICE KERNEL DURATION [ns]"
OP_COL = "OP CODE"
CORE_COL = "CORE COUNT"


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(path, metric):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            sys.exit(f"metric column {metric!r} not found in {path}\navailable: {reader.fieldnames}")
        rows = []
        for r in reader:
            dur = _to_float(r.get(metric))
            if dur is None:
                continue
            rows.append(
                {
                    "op": (r.get(OP_COL) or "").strip(),
                    "dur": dur,
                    "cores": _to_float(r.get(CORE_COL)),
                }
            )
        return rows


@dataclass
class Agg:
    count: int = 0
    total: float = 0.0
    durs: list = field(default_factory=list)
    cores: list = field(default_factory=list)

    def add(self, dur, cores):
        self.count += 1
        self.total += dur
        self.durs.append(dur)
        if cores is not None:
            self.cores.append(cores)

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    @property
    def mean_cores(self):
        return sum(self.cores) / len(self.cores) if self.cores else 0.0


def aggregate(rows):
    agg = defaultdict(Agg)
    for r in rows:
        agg[r["op"]].add(r["dur"], r["cores"])
    return agg


def _pct(new, old):
    if old == 0:
        return float("inf") if new else 0.0
    return (new - old) / old * 100.0


def _fmt_us(ns):
    return f"{ns / 1000.0:9.3f}us"


def _fmt_delta_pct(p):
    if p == float("inf"):
        return "   NEW  "
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:6.1f}%"


def print_op_table(ref_agg, lat_agg, top, out_csv):
    ops = sorted(set(ref_agg) | set(lat_agg))
    rows = []
    for op in ops:
        r = ref_agg.get(op, Agg())
        l = lat_agg.get(op, Agg())
        rows.append(
            {
                "op": op,
                "ref_count": r.count,
                "lat_count": l.count,
                "ref_total": r.total,
                "lat_total": l.total,
                "ref_mean": r.mean,
                "lat_mean": l.mean,
                "ref_cores": r.mean_cores,
                "lat_cores": l.mean_cores,
                "total_delta": l.total - r.total,
                "total_pct": _pct(l.total, r.total),
                "mean_pct": _pct(l.mean, r.mean),
            }
        )

    ref_grand = sum(r["ref_total"] for r in rows)
    lat_grand = sum(r["lat_total"] for r in rows)

    print("=" * 118)
    print("PER-OP-CODE AGGREGATE  (device kernel time; REFERENCE = oldest, LATEST = newest)")
    print("=" * 118)
    hdr = (
        f"{'OP CODE':<34}{'cnt r/l':>10}{'mean ref':>13}{'mean lat':>13}"
        f"{'mean Δ%':>10}{'total ref':>13}{'total lat':>13}{'total Δ%':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in sorted(rows, key=lambda x: x["total_delta"], reverse=True):
        print(
            f"{row['op']:<34}"
            f"{str(row['ref_count']) + '/' + str(row['lat_count']):>10}"
            f"{_fmt_us(row['ref_mean']):>13}"
            f"{_fmt_us(row['lat_mean']):>13}"
            f"{_fmt_delta_pct(row['mean_pct']):>10}"
            f"{_fmt_us(row['ref_total']):>13}"
            f"{_fmt_us(row['lat_total']):>13}"
            f"{_fmt_delta_pct(row['total_pct']):>10}"
        )
    print("-" * len(hdr))
    print(
        f"{'TOTAL':<34}{'':>10}{'':>13}{'':>13}{'':>10}"
        f"{_fmt_us(ref_grand):>13}{_fmt_us(lat_grand):>13}{_fmt_delta_pct(_pct(lat_grand, ref_grand)):>10}"
    )

    changed = [r for r in rows if r["ref_count"] and r["lat_count"]]
    slower = sorted(changed, key=lambda x: x["total_delta"], reverse=True)[:top]
    faster = sorted(changed, key=lambda x: x["total_delta"])[:top]

    print("\n" + "=" * 60)
    print(f"TOP {top} SLOWER (by total device-time increase)")
    print("=" * 60)
    for r in slower:
        if r["total_delta"] <= 0:
            continue
        print(
            f"  {r['op']:<32} {_fmt_delta_pct(r['total_pct'])}  (+{_fmt_us(r['total_delta']).strip()})  mean {_fmt_delta_pct(r['mean_pct']).strip()}"
        )

    print("\n" + "=" * 60)
    print(f"TOP {top} FASTER (by total device-time decrease)")
    print("=" * 60)
    for r in faster:
        if r["total_delta"] >= 0:
            continue
        print(
            f"  {r['op']:<32} {_fmt_delta_pct(r['total_pct'])}  ({_fmt_us(r['total_delta']).strip()})  mean {_fmt_delta_pct(r['mean_pct']).strip()}"
        )

    print(
        f"\nOverall device kernel time: {_fmt_us(ref_grand).strip()} -> {_fmt_us(lat_grand).strip()}"
        f"  ({_fmt_delta_pct(_pct(lat_grand, ref_grand)).strip()})"
    )

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "op",
                    "ref_count",
                    "lat_count",
                    "ref_mean_ns",
                    "lat_mean_ns",
                    "mean_pct",
                    "ref_total_ns",
                    "lat_total_ns",
                    "total_delta_ns",
                    "total_pct",
                    "ref_mean_cores",
                    "lat_mean_cores",
                ]
            )
            for r in sorted(rows, key=lambda x: x["total_delta"], reverse=True):
                w.writerow(
                    [
                        r["op"],
                        r["ref_count"],
                        r["lat_count"],
                        f"{r['ref_mean']:.1f}",
                        f"{r['lat_mean']:.1f}",
                        f"{r['mean_pct']:.2f}",
                        f"{r['ref_total']:.1f}",
                        f"{r['lat_total']:.1f}",
                        f"{r['total_delta']:.1f}",
                        f"{r['total_pct']:.2f}",
                        f"{r['ref_cores']:.1f}",
                        f"{r['lat_cores']:.1f}",
                    ]
                )
        print(f"\nWrote per-op CSV -> {out_csv}")


def print_positional(ref_rows, lat_rows, top):
    if len(ref_rows) != len(lat_rows):
        print(
            f"\n[positional diff skipped] op counts differ: reference={len(ref_rows)} rows, "
            f"latest={len(lat_rows)} rows. Aggregate table above is the reliable comparison."
        )
        return
    mismatched = [i for i in range(len(ref_rows)) if ref_rows[i]["op"] != lat_rows[i]["op"]]
    if mismatched:
        print(
            f"\n[positional diff skipped] op sequence differs at {len(mismatched)} positions "
            f"(first at index {mismatched[0]})."
        )
        return
    print("\n" + "=" * 90)
    print("POSITIONAL PER-INSTANCE DELTAS (same op sequence)")
    print("=" * 90)
    diffs = []
    for i, (r, l) in enumerate(zip(ref_rows, lat_rows)):
        diffs.append((i, r["op"], r["dur"], l["dur"], l["dur"] - r["dur"]))
    for i, op, rd, ld, d in sorted(diffs, key=lambda x: x[4], reverse=True)[:top]:
        print(f"  #{i:<4} {op:<34} {_fmt_us(rd)} -> {_fmt_us(ld)}  ({_fmt_delta_pct(_pct(ld, rd))})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="oldest report CSV (baseline)")
    ap.add_argument("latest", help="newest report CSV")
    ap.add_argument("--metric", default=METRIC_DEFAULT, help=f"metric column (default: {METRIC_DEFAULT!r})")
    ap.add_argument("--top", type=int, default=10, help="how many entries in the faster/slower lists")
    ap.add_argument("--csv", dest="out_csv", default=None, help="optional path to write per-op comparison CSV")
    args = ap.parse_args()

    ref_rows = load_rows(args.reference, args.metric)
    lat_rows = load_rows(args.latest, args.metric)

    print(f"REFERENCE (oldest): {args.reference}  ({len(ref_rows)} ops)")
    print(f"LATEST    (newest): {args.latest}  ({len(lat_rows)} ops)")
    print(f"metric: {args.metric}\n")

    print_op_table(aggregate(ref_rows), aggregate(lat_rows), args.top, args.out_csv)
    print_positional(ref_rows, lat_rows, args.top)


if __name__ == "__main__":
    main()
