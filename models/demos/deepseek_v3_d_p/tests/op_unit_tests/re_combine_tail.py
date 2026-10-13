# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Splits the overlapped RE + combine program into the routed expert's span and combine's tail.

Reads the raw device profiler log of one test_hybrid_routed_expert_combine_overlap_perf case. The case
captures three traces in order: the routed expert alone, combine alone, then the overlap. Per chip and per
replay, a program spans the first kernel start to the last kernel end on any Tensix core. In the overlap,
the routed expert owns logical rows 2-9, which are NOC0 rows 4-11 on Blackhole (rows are never harvested),
so its span ends at the last kernel end on those rows. Each span is taken on its own slowest chip, as the
mesh waits for that chip: the hot expert's chip sets the routed expert's span, and combine's last receiver can be
another chip. The tail is the overlap's span past the routed expert's. The table takes the median of the timed
replays.

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


def _timed(per_replay):
    """Replays in order, without the untimed warm-up replay."""
    replays = sorted(per_replay)
    return replays[1:] if len(replays) > 1 else replays


def _span(per_replay, field):
    """Median over the timed replays of the slowest chip's `field` (0: whole program, 1: routed expert)."""
    return statistics.median(max(v[field] for v in per_replay[r].values()) for r in _timed(per_replay))


def main(argv):
    print(
        "| Cell | RE alone | RE overlapped | RE slowdown | Combine alone | Tail | Tail % | Hidden % "
        "| Overlap total | RE cores |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    details = []
    for arg in argv:
        label, path = arg.split("=", 1)
        spans = _spans(path)
        traces = sorted(spans)
        if len(traces) != 3:
            print(f"| {label} | expected 3 traces, found {traces} |")
            continue
        re_tr, cmb_tr, ovl_tr = traces
        re_alone = _span(spans[re_tr], 0)
        cmb_alone = _span(spans[cmb_tr], 0)
        re_ovl = _span(spans[ovl_tr], 1)
        total = _span(spans[ovl_tr], 0)
        tail = total - re_ovl
        cores = min(v[2] for r in spans[ovl_tr] for v in spans[ovl_tr][r].values())
        print(
            f"| {label} | {re_alone:.1f} us | {re_ovl:.1f} us | {re_ovl - re_alone:.1f} us | {cmb_alone:.1f} us "
            f"| {tail:.1f} us | {tail / cmb_alone:.1%} | {1 - tail / cmb_alone:.1%} | {total:.1f} us | {cores} |"
        )
        for name, tr in (("re", re_tr), ("combine", cmb_tr), ("overlap", ovl_tr)):
            for r in _timed(spans[tr]):
                chips = " ".join(
                    f"{c}:{v[0]:.0f}" + ("" if name != "overlap" else f"/{v[1]:.0f}")
                    for c, v in sorted(spans[tr][r].items())
                )
                details.append(f"{label} {name} replay {r} chip:total[/re_end] {chips}")
    print("\n".join(details))


if __name__ == "__main__":
    main(sys.argv[1:])
