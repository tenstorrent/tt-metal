#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tables from the FAG_MODES lines of test_fabric_all_gather_modes.py logs: device us per variant, per size, and the
busiest hop's rate per link direction (busiest_hop_shards x shard bytes / links / time; bf16 TILE shards, 2 KiB a tile).

usage: fag_modes_analyze.py run.log [more.log ...]   (a case measured in several logs keeps the last log's number)
"""

import re
import sys
from collections import defaultdict

TILE_BYTES = 2048
_ORDER = ("base", "b", "d", "db", "m", "mb", "ms", "mh")
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
seen = {v for r in rows.values() for v in r}
known = [v for v in _ORDER if v in seen] + [f"{v}@auto" for v in _ORDER if f"{v}@auto" in seen]
variants = known + sorted(seen - set(known))
for fab in sorted({k[0] for k in rows}):
    for links in sorted({k[1] for k in rows if k[0] == fab}):
        keys = sorted([k for k in rows if k[0] == fab and k[1] == links], key=lambda k: (k[4], k[2]))
        print(f"\n{fab}, links={links}  (device us; * = fastest)")
        print(f"{'shape':>12} {'tiles':>6} " + " ".join(f"{v:>9}" for v in variants) + "   winner")
        for k in keys:
            r = rows[k]
            best = min(r, key=r.get)
            cells = " ".join(
                f"{(('%.1f' % r[v]) + ('*' if v == best else ' ')) if v in r else '-':>9}" for v in variants
            )
            print(f"{k[2]:>5}x{k[3]:<6} {k[4]:>6} {cells}   {best}")
        print(f"\n{fab}, links={links}  (GB/s per link direction on the busiest hop)")
        print(f"{'shape':>12} {'tiles':>6} " + " ".join(f"{v:>9}" for v in variants))
        for k in keys:
            r = rows[k]
            gbps = lambda v: busy[v] * k[4] * TILE_BYTES / links / r[v] / 1e3  # noqa: E731
            print(
                f"{k[2]:>5}x{k[3]:<6} {k[4]:>6} "
                + " ".join(f"{gbps(v):9.1f}" if v in r else f"{'-':>9}" for v in variants)
            )
print("\nbusiest hop (shards) per variant:", {v: busy[v] for v in variants})
