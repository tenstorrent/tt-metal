# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize only uncontaminated profiler sum slots.

LLK CB waits also write sum slot 0 on UNPACK and slot 1 on PACK. Those
thread/slot pairs cannot be interpreted as this experiment's stage totals.
MATH has both slots available, PACK has slot 0, UNPACK has slot 1.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict


def analyze(path):
    records = defaultdict(dict)
    with open(path) as f:
        header = f.readline().strip()
        for raw in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in raw.items()}
            thread = r["RISC processor type"]
            if not thread.startswith("TRISC_"):
                continue
            key = (r["run host ID"], r["core_x"], r["core_y"], thread)
            if r["zone name"] == "TRISC-KERNEL":
                records[key][r["type"]] = int(r["time[cycles since reset]"])
            if r["type"] == "ZONE_TOTAL" and r["zone name"].startswith("UTIL_"):
                records[key][r["zone name"]] = int(r["data"])
    groups = defaultdict(list)
    for (run, x, y, thread), r in records.items():
        duration = r["ZONE_END"] - r["ZONE_START"]
        assert 0 < duration < 2**32, (run, x, y, thread, duration)
        for name, cycles in r.items():
            if not name.startswith("UTIL_"):
                continue
            slot = 0 if name in ("UTIL_QK", "UTIL_SUB_EXP") else 1
            if (thread == "TRISC_0" and slot == 0) or (thread == "TRISC_2" and slot == 1):
                continue
            assert 0 < cycles <= duration, (run, x, y, thread, name, cycles, duration)
            groups[(run, thread, name)].append(100 * cycles / duration)
    for (run, thread, name), pct in sorted(groups.items()):
        print(
            json.dumps(
                dict(
                    path=path,
                    header=header,
                    run=run,
                    thread=thread,
                    stage=name,
                    cores=len(pct),
                    mean_pct=statistics.mean(pct),
                    min_pct=min(pct),
                    max_pct=max(pct),
                    semantics="inclusive per-thread time; not exclusive hardware or additive critical-path time",
                )
            )
        )


for path in sys.argv[1:]:
    analyze(path)
