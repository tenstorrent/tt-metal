# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only reducer for the `cskip_finalize` bench's per-TRISC zone numbers.

`tests/.../perf_zone_report.py` assumes every RISC carries a `*-KERNEL` span; this
single-core, compute-only bench has no reader/writer kernel, so it trips that
assumption. Same arithmetic, no span bookkeeping.

    python3 .../cskip_finalize/zone_reduce.py <log.csv> [<log.csv> ...]
"""

from __future__ import annotations

import csv
import re
import sys

CLK_MHZ_DEFAULT = 1350.0
# TRISC_0 = unpack, TRISC_1 = math, TRISC_2 = pack.
RISC_ROLE = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}


def reduce_log(path, zone_filter="cp_finalize"):
    rows = list(csv.reader(open(path)))
    clk = CLK_MHZ_DEFAULT
    for tok in rows[0]:
        if "CHIP_FREQ" in tok:
            clk = float(tok.split(":")[1])
    rows = rows[2:]
    ids = sorted({int(r[7]) for r in rows if len(r) > 7 and r[7].strip().isdigit()})
    rid = ids[-1] if ids else None  # the LAST dispatch in the log == this variant's run
    rows = [r for r in rows if len(r) > 7 and r[7].strip().isdigit() and int(r[7]) == rid]

    per = {}
    for r in rows:
        if len(r) < 12:
            continue
        risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
        if zone_filter not in zone:
            continue
        per.setdefault(risc, []).append((typ, t))

    out = {}
    for risc, evs in per.items():
        stack, total, count = [], 0.0, 0
        for typ, t in evs:
            if typ == "ZONE_START":
                stack.append(t)
            elif typ == "ZONE_END" and stack:
                total += (t - stack.pop()) * 1000.0 / clk
                count += 1
        out[RISC_ROLE.get(risc, risc)] = (total, count)
    return out, rid


def main(paths):
    print(f"{'log':44s} {'unpack_ns':>10s} {'math_ns':>10s} {'pack_ns':>10s} {'exec':>5s}")
    for p in paths:
        out, rid = reduce_log(p)
        name = re.sub(r".*/", "", p)
        u = out.get("unpack", (0.0, 0))
        m = out.get("math", (0.0, 0))
        k = out.get("pack", (0.0, 0))
        print(f"{name:44s} {u[0]:10.0f} {m[0]:10.0f} {k[0]:10.0f} {max(u[1], m[1], k[1]):5d}")


if __name__ == "__main__":
    main(sys.argv[1:])
