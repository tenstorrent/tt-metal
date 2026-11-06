#!/usr/bin/env python3
"""trid_report.py [--results DIR ...]: the per-channel transaction-ID A/B on the P150 (trid_runs.sh; TRID_ALLOCATION.md
section 7). Per suite and test row: cycles/packet at 1.35 GHz per pass and arm (each run's per-test CSV, rows keyed as
slotalloc/dyn_report.py does), RULE against UNIFORM (mean of the two passes, and per pass), and what differs between
the arms on that suite:
  codegen   2D mesh (16 slots, groups of 4) and 1D (32, groups of 4): the rule's 4 / 8 IDs give the same slot-to-ID map
            as the uniform 16 (4 or 8 groups, an ID each either way), and a flushed atomic takes the counter path (no
            deadlock avoidance), so only the generated code differs
  behaviour rings (32 slots, first-level acks: 16 IDs shared by 2 slots against 8 shared by 4) and the torus axes
            (16 slots, first-level acks: an ID per slot against 8 shared by 2): fewer slots share an ID, so a proof
            waits on fewer other writes; a flushed atomic walks the channel's 16 IDs against 8
Per suite: rows, wins and losses beyond 1%, median, best, worst."""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../consolidation/slotalloc"))
import dyn_report as dr  # noqa: E402

SUITES = [("t42ns", "behaviour"), ("ring", "behaviour"), ("c2d", "codegen"), ("c1lin", "codegen"), ("c1ne", "codegen")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", default=["/home/aagarwal/tt-metal/fabric_exp/results"])
    ap.add_argument("--prefix", default="tr")
    a = ap.parse_args()
    summary = []
    for s, kind in SUITES:
        d = {}
        for p in "ab":
            for arm in ("uni", "rule"):
                for r in a.results:
                    f = os.path.join(r, f"{a.prefix}{p}_{arm}_{s}_tests.csv")
                    if os.path.exists(f):
                        d[(p, arm)] = dr.with_link_totals(dr.rows([f]))
                        break
        if not d:
            continue
        keys = sorted(set.intersection(*(set(v) for v in d.values())))
        print(f"## {s} ({kind}; runs: {', '.join(f'{p}/{arm}' for p, arm in sorted(d))})\n")
        print("| row | uniform a | uniform b | rule a | rule b | rule vs uniform | a | b |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        changes = []
        for k in keys:
            cell = lambda p, arm: f"{d[(p, arm)][k]:.1f}" if (p, arm) in d else "-"  # noqa: E731
            uv = [d[(p, "uni")][k] for p in "ab" if (p, "uni") in d]
            rv = [d[(p, "rule")][k] for p in "ab" if (p, "rule") in d]
            ch = 100 * (st.mean(rv) / st.mean(uv) - 1) if uv and rv else None
            per = [f"{100 * (d[(p, 'rule')][k] / d[(p, 'uni')][k] - 1):+.1f}%"
                   if (p, "rule") in d and (p, "uni") in d else "-" for p in "ab"]
            label = k[0] + (f" dev{k[1]}" if k[1] and k[1] != "LINK" else "") + (" LINK TOTAL" if k[1] == "LINK" else "")
            print(f"| {label} | {cell('a', 'uni')} | {cell('b', 'uni')} | {cell('a', 'rule')} | {cell('b', 'rule')} | "
                  f"{'-' if ch is None else f'{ch:+.1f}%'} | {per[0]} | {per[1]} |")
            if ch is not None and not dr.is_flow(k):
                changes.append((ch, label))
        print()
        if changes:
            changes.sort()
            summary.append((s, kind, len(changes), sum(c < -1 for c, _ in changes), sum(c > 1 for c, _ in changes),
                            st.median(c for c, _ in changes), changes[0], changes[-1]))
    print("## Per suite (RULE against UNIFORM, mean of the passes)\n")
    print("| suite | differs in | rows | wins (< -1%) | losses (> +1%) | median | best | worst |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for s, kind, n, w, l, med, best, worst in summary:
        print(f"| {s} | {kind} | {n} | {w} | {l} | {med:+.1f}% | {best[0]:+.1f}% {best[1]} | {worst[0]:+.1f}% {worst[1]} |")


if __name__ == "__main__":
    main()
