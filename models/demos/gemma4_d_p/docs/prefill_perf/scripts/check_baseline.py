#!/usr/bin/env python3
"""Fit run logs and compare against BASELINE.json. One command after a rebase.

    check_baseline.py --config all_fixes runs/*.log

Exits non-zero if any cell moved more than --tol (default 1%). Within-session sigma
here is 0.03% on `a`; cross-session drift is 0.2-0.4%. 1% is comfortably outside both,
so a failure means something really changed -- rebase, board, or a fix that stopped
engaging. Check the witness lines in the log before blaming the model.
"""
import argparse
import json
import os
import re
import sys

RE = re.compile(r"\[traced_perf\] chunk (\d+)/(\d+) \[(\d+), (\d+)\) device=([0-9.]+)ms")


def fit(ts):
    n = len(ts)
    sx = sum(range(n))
    sy = sum(ts)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * t for i, t in enumerate(ts))
    d = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / d
    a = (sy - slope * sx) / n
    return a, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--config", default="all_fixes")
    ap.add_argument("--tol", type=float, default=1.0, help="percent")
    ap.add_argument("--isl", type=int, default=262144)
    ap.add_argument("--baseline", default=os.path.join(os.path.dirname(__file__), "BASELINE.json"))
    args = ap.parse_args()

    ref = json.load(open(args.baseline))["configs"].get(args.config)
    if ref is None:
        sys.exit(f"no config '{args.config}' in {args.baseline}")

    checked = skipped = 0
    print(f"{'chunk':>6} {'metric':>8} {'expected':>10} {'measured':>10} {'drift':>9}  verdict")
    print("-" * 62)
    bad = 0
    for path in args.logs:
        rows = {}
        for line in open(path):
            m = RE.search(line)
            if m:
                rows[int(m.group(1))] = float(m.group(5))
        if not rows:
            print(f"  {os.path.basename(path)}: no [traced_perf] lines -- run failed?")
            bad += 1
            continue
        if "witness_fail=1" in open(path).read():
            print(f"  {os.path.basename(path)}: WITNESS FAIL -- a flag never engaged; result void")
            bad += 1
            continue
        # Chunk width, most reliable source first. The pytest node id is echoed into
        # every log and always carries it; headers and filenames vary by era.
        chunk = None
        body = open(path).read()
        m = re.search(r"chunk(\d+)-text-8x4", body) or re.search(r"^# chunk\s+(\d+)", body, re.M)
        if m:
            chunk = m.group(1)
        else:
            m = re.search(r"chunk(\d+)", os.path.basename(path)) or re.search(r"[_/]c(\d+)[_.]", os.path.basename(path))
            chunk = m.group(1) if m else None
        if chunk is None:
            print(f"  {os.path.basename(path)}: cannot determine chunk width")
            bad += 1
            continue
        exp = ref.get(str(chunk))
        if exp is None:
            print(f"  chunk {chunk}: not in baseline, skipping")
            skipped += 1
            continue
        checked += 1
        ts = [rows[k] for k in sorted(rows)]
        a, slope = fit(ts)
        n = args.isl // int(chunk)
        total = (n * a + slope * n * (n - 1) / 2) / 1000
        for name, got, want in (("a", a, exp["a"]), ("slope", slope, exp["slope"]), ("total_s", total, exp["total_s"])):
            drift = 100 * (got - want) / want
            ok = abs(drift) <= args.tol
            bad += 0 if ok else 1
            print(f"{chunk:>6} {name:>8} {want:>10.3f} {got:>10.3f} {drift:>+8.2f}%  " f"{'ok' if ok else 'DRIFT'}")
    print("-" * 62)
    if checked == 0:
        print(f"NOTHING CHECKED ({skipped} skipped). Not a pass -- fix the inputs.")
        return 2
    if bad:
        print(f"{bad} cell(s) outside +/-{args.tol}%  ({checked} config/width pairs checked)")
        return 1
    print(f"all within tolerance ({checked} config/width pairs checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
