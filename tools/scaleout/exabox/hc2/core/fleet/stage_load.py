#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""stage_load.py: the TFLOP/s line of run.sh's C5 stage.

Usage: stage_load.py C5 <stage dir>
Reads only what the stage already wrote (no device, no timer inside a burst) and prints one line:
  LOAD C5: <x> TFLOP/s per chip, op <op>; <how it was counted>
It also writes <stage dir>/load.json ({"stage", "tflops_per_chip", "op", ...}, as try*.json's burst record).
  C5   all-gather only, no matmul: 0 TFLOP/s; each pair's link GB/s (fabric_repro.py rows), the slowest pair named by
       BDF and tray 1-4.
A stage that hung or wrote nothing prints what is known; it never fails (exit 0).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunk_sched import tray_of  # noqa: E402


def jsonl(path):
    out = []
    try:
        for ln in open(path, errors="replace"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def lk(bdf):
    return f"{bdf[5:] if bdf.count(':') == 2 else bdf} (tray {tray_of(bdf) or '?'})"


def c5(d):
    pairs = {}
    try:
        for ln in list(open(os.path.join(d, "pairs.tsv")))[1:]:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 6:
                pairs[f"p{f[0]}"] = (f[4], f[5])
    except OSError:
        pass
    per = []
    for lab, (ba, bb) in sorted(pairs.items(), key=lambda kv: int(kv[0][1:])):
        g = [r["GBps_total"] for r in jsonl(os.path.join(d, f"{lab}.rows"))
             if r.get("event") == "iter" and isinstance(r.get("GBps_total"), (int, float))]
        if g:
            per.append(dict(pair=lab, bdf_a=ba, tray_a=tray_of(ba), bdf_b=bb, tray_b=tray_of(bb),
                            gbps=round(sum(g) / len(g), 1)))
    rec = dict(op="all_gather", tflops_per_chip=0, pairs=per)
    s = "0 TFLOP/s per chip, op all_gather (pairwise link check, no matmul); "
    if not per:
        return rec, s + f"link GB/s not read ({len(pairs)} pairs listed, none with a row)"
    lo, hi = min(per, key=lambda p: p["gbps"]), max(per, key=lambda p: p["gbps"])
    return rec, s + (f"{len(per)} pairs, link GB/s {lo['gbps']:g}-{hi['gbps']:g}, slowest "
                     f"{lk(lo['bdf_a'])} <-> {lk(lo['bdf_b'])}")


STAGES = {"C5": c5}


def main(stage, d):
    try:
        rec, s = STAGES[stage](d)
    except Exception as e:   # a malformed row: say so, never fail the run
        rec, s = {}, f"TFLOP/s not reported ({type(e).__name__}: {str(e)[:100]})"
    rec["stage"] = stage
    try:
        tmp = os.path.join(d, f".load.json.tmp{os.getpid()}")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(d, "load.json"))
    except OSError:
        pass
    return f"LOAD {stage}: {s}"


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2]))
