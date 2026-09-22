#!/usr/bin/env python3
"""A/B two tt-perf-report renders and attribute the delta per op. Stdlib only.

    diff_ops.py BEFORE.txt AFTER.txt [--layers 50] [--iters 2]

`--layers` scales a per-layer capture to the whole model (50 sliding / 10 global).
`--iters` is how many times the benchmark ran the layer inside one capture; the
published captures use 2. If the whole-model delta does not reconcile with the e2e
delta, say so -- an attribution that does not close is a finding, not a rounding error.
"""
import argparse
import collections
import re
import sys

ROW = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s%\s+(SLOW|DRAM|COMP|\s*)\s*"
    r"([A-Za-z]+DeviceOperation)\s*([\d x]*?)\s+(\d+)\s+([\d.]+)\s*([mμ])s"
)


def load(path):
    tot = collections.Counter()
    for line in open(path):
        m = ROW.match(line)
        if not m:
            continue
        us = float(m.group(7)) * (1000 if m.group(8) == "m" else 1)
        dims = m.group(5).strip()
        key = m.group(4).replace("DeviceOperation", "") + (f" {dims}" if dims else "")
        tot[key] += us
    if not tot:
        sys.exit(f"no op rows parsed from {path} -- wrong file, or the renderer changed")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument(
        "--min-us", type=float, default=2.0, help="hide ops whose delta is under this; device spread on one op is ~17%%"
    )
    a = ap.parse_args()
    b, f = load(a.before), load(a.after)
    scale = a.layers / a.iters / 1000.0  # us -> ms, whole model

    print(f"{'op':<32}{'before us':>11}{'after us':>10}{'delta':>9}{'%':>8}{'model ms':>10}")
    print("-" * 80)
    for k in sorted(set(b) | set(f), key=lambda k: -abs(f[k] - b[k])):
        d = f[k] - b[k]
        if abs(d) < a.min_us:
            continue
        pct = 100 * d / b[k] if b[k] else float("nan")
        print(f"{k:<32}{b[k]:>11.1f}{f[k]:>10.1f}{d:>+9.1f}{pct:>+7.1f}%{d*scale:>+10.2f}")
    tb, tf = sum(b.values()), sum(f.values())
    print("-" * 80)
    print(f"{'TOTAL':<32}{tb:>11.1f}{tf:>10.1f}{tf-tb:>+9.1f}{100*(tf-tb)/tb:>+7.1f}%{(tf-tb)*scale:>+10.2f}")
    if a.layers > 1:
        print(f"\nwhole-model delta: {(tf-tb)*scale:+.2f} ms  " f"(x{a.layers} layers, /{a.iters} iters)")
        print("Compare against the e2e delta in `a`. Reconcile it and state the residual.")


if __name__ == "__main__":
    main()
