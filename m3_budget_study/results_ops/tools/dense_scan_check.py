#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1-E readout: dense-layer cost at h=16384 vs KV slot capacity (64k vs 1M), budget harness and runner.

  dense_scan_check.py [--logs logs] [--pipeline pipeline] [--out dense_scan_check.txt]

A (budget_sweep, layers 0-2 on one (2,4) stage): logs/p1e_bs_cap<C>.log RESULT lines; the `point` median wall of
one forward of the 3 dense layers, per h, and DRAM in use (`mem`) after the last point.
B (runner, one layer per stage): pipeline/p1e_rn_cap*/runner.log CHUNK_COMPUTE medians per stage and stream
(stages 0-2 = dense layers 0-2, stage 3 = sparse layer 3), streams from the producer logs.
Pavlo's whole-capacity-scan fit: ~100 ms per 1M-token slot per dense layer per chunk, i.e. 1M vs 64k predicts
+~94 ms per dense layer (+~281 ms for layers 0-2); no scan predicts ~0.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_overheads import COMPUTE, chunk_labels, read_env  # noqa: E402

RES = Path(__file__).resolve().parent.parent
PAVLO_MS_PER_M_SLOT = 100.0


def budget(logs):
    out = {}
    for log in sorted(glob.glob(os.path.join(logs, "p1e_bs_cap*.log"))):
        cap = int(re.search(r"cap(\d+)", log).group(1))
        pts, mem, status = {}, None, None
        for line in open(log, errors="replace"):
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                if r.get("kind") == "point":
                    pts[(r["h"], r["n"])] = (r["wall_ms_median"], r["wall_ms_min"])
                elif r.get("kind") == "mem":
                    mem = r
            elif line.startswith("STATUS="):
                status = line.split()[0]
        out[cap] = (pts, mem, status)
    return out


def runner(pipe):
    out = {}
    for d in sorted(glob.glob(os.path.join(pipe, "p1e_rn_cap*"))):
        if not os.path.exists(os.path.join(d, "runner.log")):
            continue
        cap = int(read_env(d).get("PREFILL_MAX_SEQ_LEN", "0"))
        cp = {}
        for line in open(os.path.join(d, "runner.log"), errors="replace"):
            if m := COMPUTE.search(line):
                cp[(int(m[1]), int(m[2]))] = float(m[3])
        order = chunk_labels(d)
        res = {}
        for label in dict.fromkeys(order):
            if "warm" in label:
                continue
            cs = {c for c, lab in enumerate(order) if lab == label}
            ranks = sorted({r for r, _ in cp})
            res[label] = {
                r: statistics.median([v for (rr, c), v in cp.items() if rr == r and c in cs] or [0]) for r in ranks
            }
        out[cap] = (os.path.basename(d), res)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=str(RES / "logs"))
    ap.add_argument("--pipeline", default=str(RES / "pipeline"))
    ap.add_argument("--out", default=str(RES / "dense_scan_check.txt"))
    args = ap.parse_args()

    L = ["P1-E: dense attention cost vs KV slot capacity, W=4096, (2,4) SP=2 TP=4", ""]
    b = budget(args.logs)
    L.append("A. budget_sweep, dense layers 0-2 in one forward (median wall ms, min in brackets)")
    if not b:
        L.append("  (no logs/p1e_bs_cap*.log yet)")
    caps = sorted(b)
    for cap in caps:
        pts, mem, status = b[cap]
        cells = "  ".join(f"h={h} n={n}: {m:.2f} [{mn:.2f}]" for (h, n), (m, mn) in sorted(pts.items()))
        dram = ""
        if mem:
            a = mem.get("total_bytes_allocated_per_bank")
            dram = f"  DRAM allocated/bank {a / 2**30:.2f} GiB" if a else ""
        L.append(f"  capacity {cap:>8}: {cells}{dram}  {status or ''}")
    if len(caps) >= 2:
        lo, hi = caps[0], caps[-1]
        for h in sorted(set(b[lo][0]) & set(b[hi][0])):
            d = b[hi][0][h][0] - b[lo][0][h][0]
            hn = f"h={h[0]} n={h[1]}"
            pred = PAVLO_MS_PER_M_SLOT * (hi - lo) / 2**20 * 3
            L.append(
                f"  {hn}: {hi} vs {lo}: {d:+.2f} ms ({d / b[lo][0][h][0]:+.1%}) for 3 dense layers; "
                f"whole-capacity scan predicts {pred:+.0f} ms"
            )
    L.append("")
    r = runner(args.pipeline)
    L.append("B. runner, one layer per stage (median CHUNK_COMPUTE ms per stage; s0-s2 dense, s3 sparse)")
    if not r:
        L.append("  (no pipeline/p1e_rn_cap*/runner.log yet)")
    for cap, (name, res) in sorted(r.items()):
        for label, per in res.items():
            L.append(
                f"  capacity {cap:>8} {label:10} "
                + "  ".join(f"s{s} {v:7.2f}" for s, v in per.items())
                + f"   ({name})"
            )
    rc = sorted(r)
    if len(rc) >= 2:
        lo, hi = rc[0], rc[-1]
        pred = PAVLO_MS_PER_M_SLOT * (hi - lo) / 2**20
        for label in r[hi][1]:
            a, z = r[lo][1].get(label, {}), r[hi][1][label]
            diffs = "  ".join(f"s{s} {z[s] - a[s]:+.2f} ({(z[s] - a[s]) / a[s]:+.1%})" for s in z if s in a and a[s])
            L.append(f"  {label}: {hi} vs {lo}: {diffs}   (scan predicts {pred:+.0f} ms per dense stage)")
    txt = "\n".join(L) + "\n"
    print(txt, end="")
    Path(args.out).write_text(txt)


if __name__ == "__main__":
    main()
