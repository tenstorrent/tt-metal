#!/usr/bin/env python3
"""bfc_report.py: the bubble-flow-control injection rule's device results (RING_JAM_FIX.md §10.3).

  bfc_report.py p150 [RESULTS_DIR]   bfc_p150_runs.sh: validation (bfcv_*) and perf (bfcp<PASS>_<base|bfc>_<row>)
  bfc_report.py glx JOB_RESULTS_DIR  frs_glx_bfc.sbatch: labels <PASS>_<base|bfc>_<suite>

Per row (file, test, size, ntype): cycles/packet (1.35e9 / the summary CSV's avg_packets_per_s) as "pass a / pass b"
(one number when they agree to 0.1) per tree, the branch's mean against the base's (%), and "*" where a tree's two
passes differ by more than 1%. Then every validation / check run's result line."""
import collections
import csv
import glob
import os
import re
import sys


def rates(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(r["test_name"], int(r["packet_size"]), r.get("ntype", ""))] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def result_line(log):
    if not os.path.exists(log):
        return "no log"
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())
    m = re.findall(r"Tests: [0-9]+ executed, [0-9]+ failed", t)
    return m[-1] if m else "NO RESULT"


def fmt(v):
    a, b = v.get("a"), v.get("b")
    if a is None or b is None:
        return f"{a or b:.1f} (one pass)" if (a or b) else "-"
    star = "*" if abs(a - b) / min(a, b) > 0.01 else ""
    return f"{a:.1f}" + ("" if abs(a - b) < 0.1 else f" / {b:.1f}") + star


def table(runs):
    """runs: {(row, pass, tree): rates}"""
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (row, p, tree), r in runs.items():
        for k, v in r.items():
            rows[(row,) + k][tree][p] = v
    print("| row | test | size | ntype | base (a / b) | bfc (a / b) | bfc vs base |")
    print("| --- | --- | ---: | --- | --- | --- | ---: |")
    for k in sorted(rows):
        t = rows[k]
        mb = [x for x in t["base"].values()]
        mn = [x for x in t["bfc"].values()]
        pct = f"{(sum(mn) / len(mn)) / (sum(mb) / len(mb)) * 100 - 100:+.1f}%" if mb and mn else "-"
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {fmt(t['base'])} | {fmt(t['bfc'])} | {pct} |")


def main():
    mode = sys.argv[1]
    if mode == "p150":
        d = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        for log in sorted(glob.glob(f"{d}/bfcv_*.log")):
            print(f"validation {os.path.basename(log)[:-4]}: {result_line(log)}")
        runs = {}
        for f in sorted(glob.glob(f"{d}/bfcp[ab]_*_*.csv")):
            m = re.match(r"bfcp([ab])_(base|bfc)_(\w+)\.csv$", os.path.basename(f))
            if m:
                runs[(m.group(3), m.group(1), m.group(2))] = rates(f)
        table(runs)
    else:
        d = sys.argv[2]
        runs = {}
        for log in sorted(glob.glob(f"{d}/*_*_*.log")):
            b = os.path.basename(log)[:-4]
            if b.endswith("_hangcap") or b.startswith("glx_reset"):
                continue
            p, tree, suite = b.split("_", 2)
            if suite in ("check",) or suite.startswith("frg_val"):
                print(f"{b}: {result_line(log)}")
                continue
            c = log[:-4] + ".csv"
            if os.path.exists(c):
                runs[(suite, p, tree)] = rates(c)
        table(runs)


if __name__ == "__main__":
    main()
