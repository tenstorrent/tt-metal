# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-block timeline of one traced TPBoundary replay, from profile_log_device.csv of a run with LEGO_ZONES=1.

    python tp_boundary_zones.py <csv> [--trace N] [--replay K] [--device D] [--match REGEX]

For each zone (a block label: mm1_in0, rs_writer, ...) prints the first start, the earliest and latest end over
the cores that ran it, and the core count, in us relative to the replay's first kernel start on that device.
Defaults: the last trace, its middle replay, the slowest device."""
import argparse, csv, re, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--trace", type=int)
ap.add_argument("--replay", type=int)
ap.add_argument("--device", type=int)
ap.add_argument("--match", default=".")
a = ap.parse_args()

with open(a.csv) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    ix = {c.strip(): i for i, c in enumerate(next(rd))}
    rows = []
    for r in rd:
        tid = r[ix["trace id"]].strip()
        if not tid:
            continue
        rows.append(
            (
                int(tid),
                int(r[ix["trace id counter"]]),
                int(r[ix["PCIe slot"]]),
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
            )
        )

trace = a.trace if a.trace is not None else max(r[0] for r in rows)
rows = [r for r in rows if r[0] == trace]
replays = sorted({r[1] for r in rows})
replay = a.replay if a.replay is not None else replays[len(replays) // 2]
rows = [r for r in rows if r[1] == replay]
span = defaultdict(lambda: [float("inf"), 0])
for _, _, dev, _, _, z, ty, t in rows:
    if z.endswith("-KERNEL"):
        s = span[dev]
        s[0] = min(s[0], t) if ty == "ZONE_START" else s[0]
        s[1] = max(s[1], t) if ty == "ZONE_END" else s[1]
dev = a.device if a.device is not None else max(span, key=lambda d: span[d][1] - span[d][0])
t0 = span[dev][0]
print(f"trace {trace} replay {replay} device {dev}: span {(span[dev][1] - t0) / freq:.1f} us")
print("  ".join(f"dev{d}={(b - s) / freq:.1f}" for d, (s, b) in sorted(span.items())))

# pair starts and ends per (core, risc, zone)
open_, zs = {}, defaultdict(lambda: dict(start=[], end=[], cores=set()))
for _, _, d, core, risc, z, ty, t in sorted((r for r in rows if r[2] == dev), key=lambda r: r[7]):
    if z.endswith("-KERNEL") or not re.search(a.match, z):
        continue
    k = (core, risc, z)
    if ty == "ZONE_START":
        open_[k] = t
    elif k in open_:
        e = zs[(z, risc)]
        e["start"].append(open_.pop(k))
        e["end"].append(t)
        e["cores"].add(core)
us = lambda t: (t - t0) / freq  # noqa: E731
print(f"{'zone':<22}{'risc':<8}{'cores':>6}{'start':>9}{'end min':>9}{'end med':>9}{'end max':>9}{'dur max':>9}")
for (z, risc), e in sorted(zs.items(), key=lambda kv: min(kv[1]["start"])):
    durs = [b - s for s, b in zip(e["start"], e["end"])]
    print(
        f"{z:<22}{risc:<8}{len(e['cores']):>6}{us(min(e['start'])):>9.1f}{us(min(e['end'])):>9.1f}"
        f"{us(statistics.median(e['end'])):>9.1f}{us(max(e['end'])):>9.1f}{max(durs) / freq:>9.1f}"
    )
