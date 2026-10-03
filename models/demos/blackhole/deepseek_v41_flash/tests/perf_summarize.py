"""Summarise a Tracy/device-profiler ops CSV (ops_perf_results_*.csv) into a per-op device-time table.

python tests/perf_summarize.py <csv> [--device 0] [--after-signpost NAME] [--top 40]

Reports, for one device: per op (OP CODE): count, total / mean DEVICE KERNEL DURATION, and the idle gap between
consecutive ops (host/launch/dependency stalls). Column names vary a little between versions, so they are matched by
substring.
"""
import argparse
import csv
import sys
from collections import defaultdict


def find(cols, *needles):
    for c in cols:
        lc = c.lower()
        if all(n in lc for n in needles):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--device", default=None, help="DEVICE ID to keep (default: the lowest id found)")
    ap.add_argument("--after-signpost", default=None)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        sys.exit("empty csv")
    cols = list(rows[0].keys())
    c_op = find(cols, "op", "code")
    c_dur = find(cols, "device", "kernel", "duration") or find(cols, "kernel", "duration")
    c_dev = find(cols, "device", "id")
    c_start = find(cols, "device", "kernel", "start") or find(cols, "global", "call")
    print(f"columns used: op={c_op} dur={c_dur} dev={c_dev} start={c_start}", file=sys.stderr)
    if a.after_signpost:
        i = next((k for k, r in enumerate(rows) if a.after_signpost in (r.get(c_op) or "")), None)
        rows = rows[i + 1 :] if i is not None else rows
    if c_dev:
        dev = a.device or str(min(int(float(r[c_dev])) for r in rows if r.get(c_dev)))
        rows = [r for r in rows if str(int(float(r[c_dev]))) == str(dev)] if rows[0].get(c_dev) else rows
    agg = defaultdict(lambda: [0, 0.0])
    total = 0.0
    seq = []
    for r in rows:
        try:
            d = float(r[c_dur])
        except (TypeError, ValueError):
            continue
        agg[r[c_op]][0] += 1
        agg[r[c_op]][1] += d
        total += d
        seq.append((r[c_op], d, float(r[c_start]) if c_start and r.get(c_start) else None))
    print(f"{'OP':60s} {'count':>6s} {'total us':>10s} {'mean us':>9s} {'%':>6s}")
    for op, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"{op[:60]:60s} {n:6d} {t / 1e3:10.1f} {t / n / 1e3:9.2f} {100 * t / total:6.1f}")
    print(f"{'TOTAL device kernel time':60s} {len(seq):6d} {total / 1e3:10.1f}")
    starts = [s for _, _, s in seq if s is not None]
    if len(starts) > 1:
        wall = (max(starts) - min(starts)) + seq[-1][1]
        print(f"wall span between first and last op: {wall / 1e3:.1f} us  => gaps/idle ~ {(wall - total) / 1e3:.1f} us")


if __name__ == "__main__":
    main()
