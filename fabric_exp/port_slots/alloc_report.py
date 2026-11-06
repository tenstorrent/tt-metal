#!/usr/bin/env python3
"""alloc_report.py [--results DIR ...] [--log JOBLOG]: the staged allocator's device A/B on the P150 (alloc_runs.sh;
SLOT_ALLOCATOR.md section 12.14). Per suite and test row: cycles/packet at 1.35 GHz per pass and arm, from each run's
per-test CSV (rows keyed by test, sender device and direction, as slotalloc/dyn_report.py), and STAGED against TIER
(the mean of the two passes, and per pass). Mixed tests (names with "Mix", or PcsVC02*) are scored by the link total,
1 / sum(1 / cycles per flow); their per-flow rows are shown as "(flow)" and not counted. Per suite: rows counted, rows
more than 1% faster (wins) and slower (losses), the best and the worst row, and whether the suite meets the proposed
acceptance (no counted row more than 1% slower than TIER). With --log, the job's slot-row check lines first.
--glx: the Galaxy job's results (alloc_glx.sbatch: <kind>_<pass>_<arm>_tests.csv, kinds vc1 vc13k vc1x44 vc2k vc23k)."""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../consolidation/slotalloc"))
import dyn_report as dr  # noqa: E402

GLX_SUITES = ["vc1", "vc13k", "vc1x44", "vc2k", "vc23k"]
SUITES = ["2d_4352", "1d_4352", "ring_4352", "vc1_4352", "vc2_4352", "ne_4352",
          "2d_3072", "1d_3072", "ring_3072", "vc1_3072", "vc2_3072", "ne_3072"]


def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def label(k):
    if k[1] == "LINK":
        return f"{k[0]} LINK TOTAL"
    return k[0] + (f" dev{k[1]}" if k[1] else "") + (f" {k[2]}" if k[2] else "") + (" (flow)" if dr.is_flow(k) else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", default=["/home/aagarwal/tt-metal/fabric_exp/results"])
    ap.add_argument("--log", default=None)
    ap.add_argument("--prefix", default="pa")
    ap.add_argument("--glx", action="store_true")
    a = ap.parse_args()
    suites = GLX_SUITES if a.glx else SUITES
    def name(p, arm, s):
        return f"{s}_{p}_{arm}_tests.csv" if a.glx else f"{a.prefix}{p}_{arm}_{s}_tests.csv"
    if a.log:
        print("Slot rows per run (each checked against the mock vet):")
        for line in open(a.log, errors="replace"):
            if line.startswith("#####") or line.startswith("=== ") or "rows as vetted" in line or "rows as captured" in line:
                print("   " + line.rstrip())
        print()
    summary = []
    for s in suites:
        d = {}
        for p in "ab":
            for arm in ("tier", "staged"):
                f = find(a.results, name(p, arm, s))
                if f:
                    d[(p, arm)] = dr.with_link_totals(dr.rows([f]))
        if not d:
            continue
        keys = sorted(set.intersection(*(set(v) for v in d.values())))
        print(f"## {s} (runs: {', '.join(f'{p}/{arm}' for p, arm in sorted(d))})\n")
        print("| row | tier a | tier b | staged a | staged b | staged vs tier | a | b |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        counted = []
        for k in keys:
            cell = lambda p, arm: f"{d[(p, arm)][k]:.1f}" if (p, arm) in d else "-"  # noqa: E731
            tv = [d[(p, "tier")][k] for p in "ab" if (p, "tier") in d]
            sv = [d[(p, "staged")][k] for p in "ab" if (p, "staged") in d]
            ch = 100 * (st.mean(sv) / st.mean(tv) - 1) if tv and sv else None
            per = [f"{100 * (d[(p, 'staged')][k] / d[(p, 'tier')][k] - 1):+.1f}%"
                   if (p, "staged") in d and (p, "tier") in d else "-" for p in "ab"]
            print(f"| {label(k)} | {cell('a', 'tier')} | {cell('b', 'tier')} | {cell('a', 'staged')} | "
                  f"{cell('b', 'staged')} | {'-' if ch is None else f'{ch:+.1f}%'} | {per[0]} | {per[1]} |")
            if ch is not None and not dr.is_flow(k):
                counted.append((ch, label(k)))
        print()
        if counted:
            counted.sort()
            wins = sum(1 for c, _ in counted if c < -1)
            losses = sum(1 for c, _ in counted if c > 1)
            summary.append((s, len(counted), wins, losses, counted[0], counted[-1],
                            st.median(c for c, _ in counted), losses == 0))
    print("## Per suite (STAGED against TIER, mean of the passes; mixed tests by link total)\n")
    print("| suite | rows | wins (< -1%) | losses (> +1%) | median | best | worst | acceptance |")
    print("| --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for s, n, w, l, best, worst, med, ok in summary:
        print(f"| {s} | {n} | {w} | {l} | {med:+.1f}% | {best[0]:+.1f}% {best[1]} | {worst[0]:+.1f}% {worst[1]} | "
              f"{'met' if ok else 'NOT met'} |")


if __name__ == "__main__":
    main()
