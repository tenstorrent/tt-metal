#!/usr/bin/env python3
"""gap_loop_report.py RESULTS_DIR [--ref DIR]: the write-gap quick test (gap_loop_ab.sh). Per row (test, size, noc send
type) cycles/packet at 1.35 GHz from each run's per-test CSV (pvs_report.py's rows_of): ASBUILT (the port 6dad5fec638 as
built, main's test sender) and LOOP (the study's one-segment stream-benchmark loop in the sender) per pass, LOOP against
ASBUILT per pass, and both against the reference job's STUDY and PORTDEC arms (lb2 176226 on a07u28, means of its two
passes; another host, so a cross-host comparison)."""
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
    ap.add_argument("--ref", default="/data/aagarwal/lb2/jobs/176226/results")
    a = ap.parse_args()
    d = {}
    for p in "ab":
        for arm in ("asbuilt", "loop"):
            f = os.path.join(a.results, f"gl{p}_{arm}_tests.csv")
            if os.path.exists(f):
                d[(p, arm)] = pvs.rows_of(f)
    ref = {}
    for arm in ("study", "portdec"):
        v = [pvs.rows_of(f) for p in "ab" for f in [os.path.join(a.ref, f"pvs{p}_{arm}_ns_tests.csv")]
             if os.path.exists(f)]
        if v:
            ref[arm] = {k: st.mean(x[k] for x in v) for k in set.intersection(*(set(x) for x in v))}
    if not d:
        print("no runs found")
        return
    keys = sorted(set.intersection(*(set(v) for v in d.values())), key=lambda k: (k[0], k[2], k[1]))
    print(f"runs: {', '.join(f'{p}/{arm}' for p, arm in sorted(d))}; reference arms: {', '.join(ref) or 'none'}\n")
    head = ("| row | size | ntype | asbuilt a | asbuilt b | loop a | loop b | loop - asbuilt a | loop - asbuilt b "
            "| loop vs asbuilt | study (176226) | portdec (176226) | asbuilt vs study | loop vs study |")
    print(head)
    print("| --- | ---: | --- | " + " | ".join("---:" for _ in head.split("|")[4:-1]) + " |")

    def mean(arm, k):
        v = [d[(p, arm)][k] for p in "ab" if (p, arm) in d]
        return st.mean(v) if v else None

    def rel(x, y):
        return f"{100 * (x / y - 1):+.1f}%" if x is not None and y is not None else "-"

    for k in keys:
        cells = [f"{d[(p, arm)][k]:.1f}" if (p, arm) in d else "-" for arm in ("asbuilt", "loop") for p in "ab"]
        deltas = []
        for p in "ab":
            if (p, "asbuilt") in d and (p, "loop") in d:
                x, y = d[(p, "loop")][k], d[(p, "asbuilt")][k]
                deltas.append(f"{x - y:+.1f} ({100 * (x / y - 1):+.1f}%)")
            else:
                deltas.append("-")
        av, lv = mean("asbuilt", k), mean("loop", k)
        sv, pv = ref.get("study", {}).get(k), ref.get("portdec", {}).get(k)
        print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells + deltas) + f" | {rel(lv, av)} | "
              f"{f'{sv:.1f}' if sv else '-'} | {f'{pv:.1f}' if pv else '-'} | {rel(av, sv)} | {rel(lv, sv)} |")


if __name__ == "__main__":
    main()
