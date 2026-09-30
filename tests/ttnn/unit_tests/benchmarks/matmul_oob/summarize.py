# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Summarize run_suite.py results, or compare two selectors.

  python summarize.py results.csv                         # one run: status, config mix, utilization per tier
  python summarize.py base.csv new.csv                    # compare: speedup of new over base, per tier
  python summarize.py both.csv --base-mode oob --new-mode v2   # compare two modes stored in one CSV
"""

import argparse
import csv
import math
from collections import Counter, defaultdict

REGRESSION = 0.95  # speedup below this counts as a regression


def ok(r):
    return r is not None and r["status"] == "ok" and r["device_ns"]


def load(path, mode=None):
    """case -> row; with several rows per case (sweep candidates) keeps the fastest ok one."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if mode is None or r["mode"] == mode]
    best = {}
    for r in rows:
        cur = best.get(r["case"])
        if cur is None or (ok(r) and (not ok(cur) or float(r["device_ns"]) < float(cur["device_ns"]))):
            best[r["case"]] = r
    return best


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def short_config(r):
    """Config type plus the block parameters that matter most, compactly."""
    cfg = r.get("config", "")
    if "(" not in cfg:
        return cfg
    kind = cfg.split("(", 1)[0].replace("MatmulMultiCore", "").replace("ProgramConfig", "") or "MultiCore"
    fields = dict(kv.split("=", 1) for kv in cfg.split("(", 1)[1].rstrip(")").split(",") if "=" in kv)
    keys = ["in0_block_w", "out_block_h", "out_block_w", "per_core_M", "per_core_N", "out_subblock_h", "out_subblock_w"]
    vals = [fields[k] for k in keys if k in fields]
    extra = "mcast_in0" if fields.get("mcast_in0") == "1" else ""
    return f"{kind}[{'/'.join(vals)}]{extra}" if vals else kind


def summarize(rows):
    by_tier = defaultdict(list)
    for r in rows.values():
        by_tier[r["tier"]].append(r)

    print(f"{len(rows)} rows, statuses: {dict(Counter(r['status'] for r in rows.values()))}\n")
    print("roofline % = ideal time (larger of peak-math and DRAM-once bounds) / measured time\n")
    print(f"{'tier':8s} {'n':>4s} {'ok':>4s} {'roof p10':>9s} {'p50':>6s} {'p90':>6s}  config types")
    for tier, rs in by_tier.items():
        utils = [float(r["roofline_pct"]) for r in rs if ok(r)]
        kinds = Counter(
            r["config_type"].replace("MatmulMultiCore", "").replace("ProgramConfig", "") for r in rs if ok(r)
        )
        kinds = ", ".join(f"{k or 'MultiCore'}:{v}" for k, v in kinds.most_common())
        print(
            f"{tier:8s} {len(rs):4d} {len(utils):4d} {pct(utils, .1):9.1f} {pct(utils, .5):6.1f} {pct(utils, .9):6.1f}  {kinds}"
        )

    bad = [r for r in rows.values() if r["status"] not in ("ok", "infeasible")]
    infeasible = [r["case"] for r in rows.values() if r["status"] == "infeasible"]
    if infeasible:
        print(f"\nSkipped as infeasible (L1 too small): {', '.join(infeasible)}")
    if bad:
        print("\nNot ok:")
        for r in bad:
            print(f"  {r['case']:48s} {r['status']:11s} {r['error'][:140]}")

    worst = sorted((r for r in rows.values() if ok(r)), key=lambda r: float(r["roofline_pct"]))[:25]
    print("\nLowest roofline %:")
    for r in worst:
        print(
            f"  {r['case']:48s} {float(r['roofline_pct']):6.1f}% {r['bound']:4s} {float(r['device_ns']) / 1e3:10.1f}us  "
            f"{r['cores']:>3s}c {r['programs_per_call']}p  {short_config(r)}"
        )


def compare(base, new, base_label, new_label):
    common = [c for c in base if c in new]
    print(f"{len(common)} cases in both ({len(base)} {base_label}, {len(new)} {new_label})\n")

    status_changes = [(c, base[c]["status"], new[c]["status"]) for c in common if base[c]["status"] != new[c]["status"]]
    speedups = {}
    for c in common:
        if ok(base[c]) and ok(new[c]):
            speedups[c] = float(base[c]["device_ns"]) / float(new[c]["device_ns"])

    by_tier = defaultdict(list)
    for c, s in speedups.items():
        by_tier[base[c]["tier"]].append(s)
    print(f"speedup = {base_label} time / {new_label} time\n")
    print(f"{'tier':8s} {'n':>4s} {'geomean':>8s} {'min':>6s} {'p50':>6s} {'max':>7s} {'<0.95':>6s} {'>1.05':>6s}")
    for tier, ss in list(by_tier.items()) + [("ALL", list(speedups.values()))]:
        print(
            f"{tier:8s} {len(ss):4d} {geomean(ss):8.3f} {min(ss):6.2f} {pct(ss, .5):6.2f} {max(ss):7.2f} "
            f"{sum(s < REGRESSION for s in ss):6d} {sum(s > 1 / REGRESSION for s in ss):6d}"
        )

    if status_changes:
        print("\nStatus changes:")
        for c, b, n in status_changes:
            print(f"  {c:48s} {b:>11s} -> {n:11s} {new[c]['error'][:120]}")

    def show(title, items):
        print(f"\n{title}:")
        for c, s in items:
            print(
                f"  {c:48s} {s:6.2f}x  {float(base[c]['device_ns']) / 1e3:9.1f} -> {float(new[c]['device_ns']) / 1e3:9.1f}us  "
                f"{short_config(base[c])} -> {short_config(new[c])}"
            )

    ranked = sorted(speedups.items(), key=lambda kv: kv[1])
    show("Largest regressions", [kv for kv in ranked[:20] if kv[1] < 1])
    show("Largest improvements", [kv for kv in reversed(ranked[-20:]) if kv[1] > 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--base-mode", default=None)
    parser.add_argument("--new-mode", default=None)
    args = parser.parse_args()

    if len(args.csvs) == 1 and not args.new_mode:
        summarize(load(args.csvs[0], args.base_mode))
        return
    base_path = args.csvs[0]
    new_path = args.csvs[1] if len(args.csvs) > 1 else args.csvs[0]
    base_label = args.base_mode or "base"
    new_label = args.new_mode or "new"
    compare(load(base_path, args.base_mode), load(new_path, args.new_mode), base_label, new_label)


if __name__ == "__main__":
    main()
