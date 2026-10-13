# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Splits the overlapped RE + combine program into the routed expert's span and combine's tail.

Reads the raw device profiler log of one test_hybrid_routed_expert_combine_overlap_perf case. The case
captures three traces in order: the routed expert alone, combine alone, then the overlap. Per chip and per
replay, a program spans the first kernel start to the last kernel end on any Tensix core. In the overlap,
the routed expert owns logical rows 2-9, which are NOC0 rows 4-11 on Blackhole (rows are never harvested),
so its span ends at the last kernel end on those rows; everything after is combine's tail. Each replay is
reported on its slowest chip, and the table takes the median replay.

usage: re_combine_tail.py LABEL=profile_log_device.csv [LABEL=...]
"""

import csv
import statistics
import sys
from collections import defaultdict

_RE_NOC0_ROWS = range(4, 12)


def _spans(path):
    with open(path) as f:
        freq_mhz = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(rows)]
        col = {h: i for i, h in enumerate(header)}
        # (trace, replay, chip) -> [start, end, re_end, re_cores]
        acc = {}
        re_cores = defaultdict(set)
        for r in rows:
            if len(r) < len(header) or not r[col["trace id"]]:
                continue
            zone = r[col["zone name"]]
            if not zone.endswith("-KERNEL") or zone.startswith(("ERISC", "DRISC")):
                continue
            key = (int(r[col["trace id"]]), int(r[col["trace id counter"]]), int(r[col["PCIe slot"]]))
            t = int(r[col["time[cycles since reset]"]])
            a = acc.setdefault(key, [t, t, None])
            kind = r[col["type"]]
            y = int(r[col["core_y"]])
            if kind == "ZONE_START":
                a[0] = min(a[0], t)
            elif kind == "ZONE_END":
                a[1] = max(a[1], t)
                if y in _RE_NOC0_ROWS:
                    a[2] = t if a[2] is None else max(a[2], t)
                    re_cores[key].add((int(r[col["core_x"]]), y))
    us = 1.0 / freq_mhz
    out = defaultdict(lambda: defaultdict(dict))
    for (trace, replay, chip), (start, end, re_end) in acc.items():
        out[trace][replay][chip] = (
            (end - start) * us,
            None if re_end is None else (re_end - start) * us,
            len(re_cores[(trace, replay, chip)]),
        )
    return out


def _slowest(per_replay, skip_first=True):
    """Per replay, the slowest chip's (total, re_end, re_cores); skips the untimed warm-up replay."""
    replays = sorted(per_replay)
    if skip_first and len(replays) > 1:
        replays = replays[1:]
    return [max(per_replay[r].values(), key=lambda v: v[0]) for r in replays]


def _median_replay(samples):
    totals = sorted(samples, key=lambda v: v[0])
    return totals[len(totals) // 2]


def main(argv):
    print(
        "| Cell | RE alone | RE overlapped | RE slowdown | Combine alone | Tail | Tail % | Hidden % "
        "| Overlap total | RE cores |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arg in argv:
        label, path = arg.split("=", 1)
        spans = _spans(path)
        traces = sorted(spans)
        if len(traces) != 3:
            print(f"| {label} | expected 3 traces, found {traces} |")
            continue
        re_tr, cmb_tr, ovl_tr = traces
        re_alone = statistics.median(v[0] for v in _slowest(spans[re_tr]))
        cmb_alone = statistics.median(v[0] for v in _slowest(spans[cmb_tr]))
        total, re_ovl, cores = _median_replay(_slowest(spans[ovl_tr]))
        tail = total - re_ovl
        print(
            f"| {label} | {re_alone:.1f} us | {re_ovl:.1f} us | {re_ovl - re_alone:.1f} us | {cmb_alone:.1f} us "
            f"| {tail:.1f} us | {tail / cmb_alone:.1%} | {1 - tail / cmb_alone:.1%} | {total:.1f} us | {cores} |"
        )
        for name, tr in (("re", re_tr), ("combine", cmb_tr), ("overlap", ovl_tr)):
            per = [f"{v[0]:.1f}/{v[1] if v[1] is None else round(v[1], 1)}" for v in _slowest(spans[tr], False)]
            print(f"  {label} {name} trace {tr} replays (total/re_end, slowest chip): {' '.join(per)}")


if __name__ == "__main__":
    main(sys.argv[1:])
