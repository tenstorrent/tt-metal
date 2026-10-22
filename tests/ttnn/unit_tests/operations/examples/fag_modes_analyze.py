#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tables from the FAG_MODES lines of test_fabric_all_gather_modes.py logs: device us per variant, per size.

usage: fag_modes_analyze.py run.log [more.log ...]
"""
import re
import sys
from collections import defaultdict

pat = re.compile(
    r"FAG_MODES (\S+) links=(\d) (\S+)\s+(\d+)x(\d+) tiles=(\d+) device_us=([\d.]+) min=([\d.]+) max=([\d.]+) "
    r"busiest_hop_shards=([\d.]+)"
)
rows = defaultdict(dict)  # (fabric, links, H, W, tiles) -> variant -> us
busy = {}
for path in sys.argv[1:]:
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            fab, links, v, H, W, tiles, us = (
                m.group(1),
                int(m.group(2)),
                m.group(3),
                int(m.group(4)),
                int(m.group(5)),
                int(m.group(6)),
                float(m.group(7)),
            )
            rows[(fab, links, H, W, tiles)][v] = us
            busy[v] = float(m.group(10))
variants = [v for v in ("base", "b", "d", "db", "m", "mb", "base@auto", "b@auto") if any(v in r for r in rows.values())]
for fab in sorted({k[0] for k in rows}):
    for links in (1, 2):
        keys = sorted([k for k in rows if k[0] == fab and k[1] == links], key=lambda k: (k[4], k[2]))
        if not keys:
            continue
        print(f"\n{fab}, links={links}  (device us; * = fastest of the uniform-placement variants)")
        print(f"{'shape':>12} {'tiles':>6} " + " ".join(f"{v:>9}" for v in variants) + "   winner")
        for k in keys:
            r = rows[k]
            uni = {v: r[v] for v in r if "@" not in v}
            best = min(uni, key=uni.get) if uni else None
            cells = " ".join(
                f"{(('%.1f' % r[v]) + ('*' if v == best else ' ')) if v in r else '-':>9}" for v in variants
            )
            print(f"{k[2]:>5}x{k[3]:<6} {k[4]:>6} {cells}   {best}")
print("\nbusiest hop (shards) per variant:", {v: busy[v] for v in variants})
