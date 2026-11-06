#!/usr/bin/env python3
"""dio_report.py RESULTS_DIR: the deliver_other inline A/B (dio_ab.sh). Per test sender (main, loop) and row (test,
size, noc send type): cycles/packet at 1.35 GHz from each run's per-test CSV (pvs_report.py's rows_of), NOINLINE and
INLINE per pass, INLINE - NOINLINE per pass (cycles and %), and the mean of the two passes."""
import argparse
import importlib.util
import os
import statistics as st

spec = importlib.util.spec_from_file_location(
    "pvs_report", "/home/aagarwal/tt-metal/fabric_exp/port_slots/pvs_report.py")
pvs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pvs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    a = ap.parse_args()
    for s in ("main", "loop"):
        d = {}
        for p in "ab":
            for arm in ("noinline", "inline"):
                f = os.path.join(a.results, f"dio{p}_{s}_{arm}_tests.csv")
                if os.path.exists(f):
                    d[(p, arm)] = pvs.rows_of(f)
        if not d:
            continue
        keys = sorted(set.intersection(*(set(v) for v in d.values())), key=lambda k: (k[0], k[2], k[1]))
        print(f"## sender {s} (runs: {', '.join(f'{p}/{arm}' for p, arm in sorted(d))})\n")
        print("| row | size | ntype | noinline a | noinline b | inline a | inline b | inline - noinline a "
              "| inline - noinline b | mean |")
        print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for k in keys:
            cells = [f"{d[(p, arm)][k]:.1f}" if (p, arm) in d else "-" for arm in ("noinline", "inline") for p in "ab"]
            deltas, rel = [], []
            for p in "ab":
                if (p, "noinline") in d and (p, "inline") in d:
                    x, y = d[(p, "inline")][k], d[(p, "noinline")][k]
                    deltas.append(f"{x - y:+.1f} ({100 * (x / y - 1):+.1f}%)")
                    rel.append(x - y)
                else:
                    deltas.append("-")
            mean = f"{st.mean(rel):+.2f}" if rel else "-"
            print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells + deltas) + f" | {mean} |")
        print()


if __name__ == "__main__":
    main()
