#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run-time medians across runs: reads the "Phase times:" and "Bus-bar part times:" lines that fl_report prints
and prints, per stage and per bus-bar part, n, the median and the spread (min-max) in seconds, plus the run total
(the sum of the stage times). Each such line pair is one run, so one run.log of a long run gives one run per leg.

usage: run_times.py <run.log or run dir> ...   (a dir: every run.log under it)"""
import os
import re
import statistics
import sys

ITEM = re.compile(r"\s*(.+?) (-?\d+) s\s*$")
KINDS = (("Phase times:", "stage"), ("Bus-bar part times:", "part"))


def parse_line(body):
    """'stage 1 gate 2 s, stage 2 pre-run checks 6 s' -> [('stage 1 gate', 2), ('stage 2 pre-run checks', 6)]"""
    out = []
    for it in body.split(","):
        m = ITEM.match(it)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def runs_of(lines):
    """One dict {'stage': [...], 'part': [...]} per "Phase times:" line; a part line belongs to the phase line before it."""
    runs = []
    for ln in lines:
        for tag, kind in KINDS:
            i = ln.find(tag)
            if i < 0:
                continue
            if kind == "stage":
                runs.append({"stage": parse_line(ln[i + len(tag):]), "part": []})
            elif runs:
                runs[-1]["part"] = parse_line(ln[i + len(tag):])
    return runs


def files_of(paths):
    for p in paths:
        if os.path.isdir(p):
            for d, _, fs in sorted(os.walk(p)):
                if "run.log" in fs:
                    yield os.path.join(d, "run.log")
        else:
            yield p


def table(runs):
    """[(kind, name, n, median, min, max)] in first-seen order; 'run total' is the sum of a run's stage times."""
    vals, order = {}, []
    for r in runs:
        for kind in ("stage", "part"):
            for name, s in r[kind]:
                k = (kind, name)
                if k not in vals:
                    vals[k] = []
                    order.append(k)
                vals[k].append(s)
        if r["stage"]:
            k = ("total", "run total")
            if k not in vals:
                vals[k] = []
            vals[k].append(sum(s for _, s in r["stage"]))
    if ("total", "run total") in vals:
        order.append(("total", "run total"))
    return [(kind, name, len(vals[(kind, name)]), statistics.median(vals[(kind, name)]),
             min(vals[(kind, name)]), max(vals[(kind, name)])) for kind, name in order]


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    runs = []
    for f in files_of(argv):
        with open(f, errors="replace") as fh:
            runs += runs_of(fh)
    if not runs:
        print("run_times: no 'Phase times:' line in " + " ".join(argv), file=sys.stderr)
        return 1
    print(f"{len(runs)} run(s)")
    print(f"{'kind':<6} {'name':<34} {'n':>3} {'median s':>9} {'min-max s':>12}")
    for kind, name, n, med, lo, hi in table(runs):
        print(f"{kind:<6} {name:<34} {n:>3} {med:>9g} {f'{lo}-{hi}':>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
