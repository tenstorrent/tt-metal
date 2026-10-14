# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Summarise dump_moe_swiglu captures (r<rank>_00_moe_swiglu.sqlite / r<rank>_07_moe_swiglu_again.sqlite) per device.

  python3 classify_moe.py <hang_dir> [--grid]

Per device: distributions of reader block (XSTAGED = gb+1), writer block (WDSPLIT // 11), GO, PHASE_FREE, mailbox magic
(expert), count; the cores with a non-zero x_ready/x_free (an x-chain mid-handshake); whether the idx/counts
scratch pages agree across cores. --grid prints XSTAGED/GO per logical core (rows = y).
"""

import collections
import glob
import os
import re
import sqlite3
import sys

HG = 11  # moe_fused_swiglu HGROUPS: WDSPLIT = (writer gb + 1) * HG
CORES = 88  # moe_fused_swiglu cores per device


def load(path):
    c = sqlite3.connect(path)
    rows = c.execute(
        'select Dev, Loc, Semaphores, "Mailbox w0..15", "CB pushed/popped", "idx scratch[0:16]", '
        '"counts scratch[0:128]" from dump_moe_swiglu'
    ).fetchall()
    diag = c.execute("select Device, Location, Message from diagnostics").fetchall()
    out = collections.defaultdict(list)
    for dev, loc, sems, mb, cbs, idx, cnt in rows:
        m = re.search(r"\((\d+),(\d+)\)", loc)
        s = dict(kv.split("=") for kv in sems.split())
        s = {k: int(v) for k, v in s.items()}
        mbw = mb.split()
        out[int(dev)].append(
            dict(loc=loc, x=int(m.group(1)), y=int(m.group(2)), s=s, mb=mbw, cbs=cbs, idx=idx, cnt=cnt)
        )
    return out, diag


def hist(vals):
    return " ".join(f"{k}:{v}" for k, v in sorted(collections.Counter(vals).items(), key=lambda kv: str(kv[0])))


def summarise(path, grid=False):
    data, diag = load(path)
    print(f"== {path}: {sum(len(v) for v in data.values())} moe cores on {len(data)} devices; {len(diag)} diagnostics")
    for dev in sorted(data):
        cs = data[dev]
        print(f"-- dev {dev}: {len(cs)} cores")
        print("   reader gb+1 (XSTAGED):", hist(c["s"].get("XSTAGED") for c in cs))
        print("   writer WDSPLIT//11   :", hist(c["s"].get("WDSPLIT", 0) // HG for c in cs))
        print("   GO                   :", hist(c["s"].get("GO") for c in cs))
        print("   PHASE_FREE           :", hist(c["s"].get("PHASE_FREE") for c in cs))
        print("   magic (expert)       :", hist(c["mb"][3] if len(c["mb"]) > 3 else "?" for c in cs))
        print("   mailbox count/m_t/mb :", hist(" ".join(c["mb"][:3]) for c in cs))
        print(
            "   idx variants         :",
            len(set(c["idx"] for c in cs)),
            " counts-scratch variants:",
            len(set(c["cnt"] for c in cs)),
        )
        odd = [c for c in cs if c["s"].get("x_ready") or c["s"].get("x_free")]
        for c in odd:
            print(
                f"   x-chain nonzero: {c['loc']} x_ready={c['s'].get('x_ready')} x_free={c['s'].get('x_free')} "
                f"XSTAGED={c['s'].get('XSTAGED')} GO={c['s'].get('GO')}"
            )
        if len(cs) != CORES:
            print(f"   cores not in table: {CORES - len(cs)} (DONE / other program / skipped)")
        if grid:
            ys = sorted({c["y"] for c in cs})
            xs = sorted({c["x"] for c in cs})
            by = {(c["x"], c["y"]): c for c in cs}
            print("   grid XSTAGED/GO/WDSPLIT//11  x=" + " ".join(f"{x:>11}" for x in xs))
            for y in ys:
                cells = []
                for x in xs:
                    c = by.get((x, y))
                    cells.append(f"{c['s']['XSTAGED']}/{c['s']['GO']}/{c['s']['WDSPLIT']//HG}" if c else "-")
                print(f"   y={y}: " + " ".join(f"{t:>11}" for t in cells))
    if diag:
        print("   diagnostics (first 5):", diag[:5])


if __name__ == "__main__":
    d = sys.argv[1]
    grid = "--grid" in sys.argv
    paths = sorted(glob.glob(os.path.join(d, "r*_0[07]_moe_swiglu*.sqlite"))) if os.path.isdir(d) else [d]
    for p in paths:
        summarise(p, grid)
