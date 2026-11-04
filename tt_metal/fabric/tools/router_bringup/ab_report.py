#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Router bring-up tools (temporary: drop before merge; see README.md).

ab_report.py OUT [--clock-mhz MHZ] [--noise 1.0] [--csv FILE]: cycles/packet per row of an ab_run.sh directory, arm A
against arm B, each pass shown.

cycles/packet = clock / the summary CSV's avg_packets_per_s (the slowest-sender rate the test reports), i.e. device
cycles per packet of the row's traffic. The clock is the one test_tt_fabric used to turn cycles into time:
  --clock-mhz MHZ   given (Blackhole AICLK 1350; check the Wormhole device's AICLK, e.g. from the test log or tt-smi);
  otherwise         recovered per run from its per-sender CSV (packets_per_second x cycles / (num_packets x
                    total_traffic_count)); the report prints the clock it used. Without a per-sender CSV and without
                    --clock-mhz it stops and asks for the clock.
Per row (yaml, test, packet size, ntype, ftype, topology): A and B as "pass a / pass b (mean)", B against A in %, and
"*" on an arm whose passes differ by more than --noise % (default 1): read such rows as noise-limited.
"""
import argparse
import collections
import csv
import glob
import os
import re
import statistics
import sys


def arm_names(out):
    txt = open(os.path.join(out, "ab_run.txt")).read() if os.path.exists(os.path.join(out, "ab_run.txt")) else ""
    a = re.search(r"^tree_a=\S+ \((\S+)\)", txt, re.M)
    b = re.search(r"^tree_b=\S+ \((\S+)\)", txt, re.M)
    return (a.group(1) if a else "main"), (b.group(1) if b else "new")


def recovered_clock_hz(tests_csv):
    """The clock the test used, from its per-sender rows (median over senders), or None."""
    if not os.path.exists(tests_csv):
        return None
    v = []
    for r in csv.DictReader(open(tests_csv)):
        try:
            n = int(r["num_packets"]) * int(r["total_traffic_count"])
            if n and float(r["cycles"]) > 0:
                v.append(float(r["packets_per_second"]) * float(r["cycles"]) / n)
        except (KeyError, ValueError):
            continue
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--clock-mhz", type=float, default=None)
    ap.add_argument("--noise", type=float, default=1.0, help="pass-to-pass difference (%%) that flags an arm")
    ap.add_argument("--csv", default=None, help="also write the table as CSV")
    a = ap.parse_args()
    name_a, name_b = arm_names(a.out)
    runs = collections.defaultdict(dict)  # (stem, arm) -> {pass: {row: cycles/packet}}
    clocks = {}
    for f in sorted(glob.glob(os.path.join(a.out, "*.csv"))):
        b = os.path.basename(f)[:-4]
        if b.endswith("_tests"):
            continue
        m = re.match(r"([ab])_(.+?)_(.+)$", b)
        if not m or m.group(2) not in (name_a, name_b):
            continue
        p, arm, stem = m.groups()
        clk = a.clock_mhz * 1e6 if a.clock_mhz else recovered_clock_hz(f[:-4] + "_tests.csv")
        if not clk:
            sys.exit(f"{b}: no per-sender CSV to recover the clock from: give --clock-mhz (the device AICLK)")
        clocks[b] = clk
        rows = {}
        for r in csv.DictReader(open(f)):
            key = (r["test_name"], int(r["packet_size"]), r.get("ntype", ""), r.get("ftype", ""), r.get("topology", ""))
            rows[key] = clk / float(r["avg_packets_per_s"])
        runs[(stem, arm)][p] = rows
    if not runs:
        sys.exit(f"no ab_run.sh results in {a.out}")
    print(
        f"clock: {'given ' + str(a.clock_mhz) + ' MHz' if a.clock_mhz else 'recovered per run: ' + ', '.join(sorted({f'{c / 1e6:.1f}' for c in clocks.values()})) + ' MHz'}"
    )
    out_rows = []
    for stem in sorted({s for s, _ in runs}):
        keys = sorted({k for arm in (name_a, name_b) for pr in runs.get((stem, arm), {}).values() for k in pr})
        print(f"\n## {stem}")
        print(f"| test | size | ntype | ftype | {name_a} | {name_b} | {name_b} vs {name_a} |")
        print("| --- | ---: | --- | --- | ---: | ---: | ---: |")
        for k in keys:
            cells, means = [], []
            for arm in (name_a, name_b):
                v = [runs[(stem, arm)][p][k] for p in sorted(runs.get((stem, arm), {})) if k in runs[(stem, arm)][p]]
                if not v:
                    cells.append("-")
                    means.append(None)
                    continue
                m = sum(v) / len(v)
                flag = "*" if len(v) > 1 and (max(v) - min(v)) / min(v) * 100 > a.noise else ""
                cells.append(" / ".join(f"{x:.1f}" for x in v) + flag + (f" ({m:.1f})" if len(v) > 1 else ""))
                means.append(m)
            d = f"{100 * (means[1] - means[0]) / means[0]:+.1f}%" if None not in means else "-"
            nt = k[2].replace("NOC_UNICAST_", "").lower()
            print(f"| {k[0]} | {k[1]} | {nt} | {k[3]} | {cells[0]} | {cells[1]} | {d} |")
            out_rows.append([stem, *k, cells[0], cells[1], d])
    print(f"\ncycles/packet; '*': the arm's passes differ by more than {a.noise}%; + means {name_b} is slower.")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["yaml", "test", "size", "ntype", "ftype", "topology", name_a, name_b, "delta"])
            w.writerows(out_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
