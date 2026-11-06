#!/usr/bin/env python3
"""dyn_report.py [--ref ARM] [--all] ARM=CSV[,CSV...] ...: the dynamic slot allocator's A/B, per test row.

Each CSV is a per-test bandwidth file (<label>_tests.csv, or <label>_13129088_vc1_tests.csv for the VC1 split; a Galaxy
job's results/<label>_tests.csv). Several CSVs per arm (passes) are averaged per row. Cycles/packet = 1.35e9 / packets
per second; rows are keyed by test name and sender device (a VC1 split's two ranks report separately).

Prints, per row, each arm's cycles/packet and its change against the first arm (main's tier pick, the base), then per
arm: rows more than 1% slower than the base, and with --ref ARM (the study's row) rows more than 2% slower than the
reference (SLOT_ALLOCATOR.md's target: within ~1-2% of the study's best, no row worse than main's tiers). --all prints
every row, otherwise only rows where some arm differs from the base by more than 1%.

Mixed rows (user, 2026-11-03): a test whose flows share one link (names with "Mix", or PcsVC02*: VC0 + VC1 / VC2 on a
link) is scored by the link total, 1 / sum(1 / cycles per flow), shown as "<test> LINK TOTAL". Its per-flow rows are
printed for reading (marked "flow") and are not counted: a per-flow drop is not a regression unless the total drops.
"""
import re

MIX = re.compile(r"Mix|^PcsVC02")
import collections
import csv
import sys


def rows(paths):
    acc = collections.defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            pps = float(r["packets_per_second"])
            if pps > 0:
                acc[(r["test_name"], r.get("device", ""), r.get("direction", ""))].append(1.35e9 / pps)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def is_flow(key):
    return bool(MIX.search(key[0])) and key[1] != "LINK"


def with_link_totals(rows_):
    """Adds one LINK TOTAL row per mixed test: the flows' combined cycles per packet on the shared link."""
    by_test = collections.defaultdict(list)
    for k, v in rows_.items():
        if MIX.search(k[0]):
            by_test[k[0]].append(v)
    out = dict(rows_)
    for test, flows in by_test.items():
        if len(flows) >= 2:
            out[(test, "LINK", "TOTAL")] = 1.0 / sum(1.0 / c for c in flows)
    return out


def main():
    args = sys.argv[1:]
    ref, show_all = None, False
    if "--all" in args:
        args.remove("--all")
        show_all = True
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i : i + 2]
    arms = []
    for a in args:
        name, _, paths = a.partition("=")
        if not paths:
            sys.exit(__doc__)
        arms.append((name, rows(paths.split(","))))
    if len(arms) < 2:
        sys.exit(__doc__)
    arms = [(n, with_link_totals(d)) for n, d in arms]
    names = [n for n, _ in arms]
    if ref is not None and ref not in names:
        sys.exit(f"--ref {ref}: not an arm ({', '.join(names)})")
    base_name, base = arms[0]
    keys = sorted(set().union(*(d.keys() for _, d in arms)))
    print("| row | " + " | ".join(names) + " |")
    print("| --- |" + " ---: |" * len(names))
    worse = collections.defaultdict(list)
    worse_ref = collections.defaultdict(list)
    for k in keys:
        b = base.get(k)
        cells, differs = [], False
        for n, d in arms:
            v = d.get(k)
            if v is None:
                cells.append("-")
                continue
            if b is None or n == base_name:
                cells.append(f"{v:.1f}")
                continue
            ch = (v - b) / b * 100
            differs |= abs(ch) > 1
            cells.append(f"{v:.1f} ({ch:+.1f}%)")
            if ch > 1 and not is_flow(k):
                worse[n].append((ch, k))
            if ref is not None and n != ref and ref in names and not is_flow(k):
                r = dict(arms)[ref].get(k)
                if r is not None and (v - r) / r * 100 > 2:
                    worse_ref[n].append(((v - r) / r * 100, k))
        if show_all or differs:
            if k[1] == "LINK":
                label = f"{k[0]} LINK TOTAL"
            else:
                label = k[0] + (f" dev{k[1]}" if k[1] else "") + (f" {k[2]}" if k[2] else "") + (" (flow)" if is_flow(k) else "")
            print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    for n in names[1:]:
        w = sorted(worse[n], reverse=True)
        print(f"{n}: {len(w)} rows > 1% slower than {base_name} (mixed tests by link total)" + (f"; worst {w[0][0]:+.1f}% {w[0][1][0]}" if w else ""))
        if ref is not None and n != ref:
            wr = sorted(worse_ref[n], reverse=True)
            print(f"{n}: {len(wr)} rows > 2% slower than {ref}" + (f"; worst {wr[0][0]:+.1f}% {wr[0][1][0]}" if wr else ""))


if __name__ == "__main__":
    main()
