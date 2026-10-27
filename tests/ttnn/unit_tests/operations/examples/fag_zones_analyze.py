#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-port timeline of fabric_all_gather send modes from test_fabric_all_gather_zones.py (run under --profile).

usage: fag_zones_analyze.py run.log profile_log_device.csv
Per case (FAG_ZONES line) and call, per chip: program span = first to last zone of the program on that chip. Per port
(sender BRISC, reader NCRISC on that port core), medians over calls of:
  fence      S_fence: send readies + wait for every destination's ready
  start      program start -> first send (S_e0 start)
  S_e<k>     send of entry k (dur), and inside it slot = waiting for a free fabric slot, cb = waiting for the reader
  tail       S_tail: after sending, wait for everything from upstream to land
  R_e<k>     reader entry k (dur), arr = waiting for upstream arrivals (relays), cb = waiting for CB space
"""
import csv
import re
import statistics
import sys
from collections import defaultdict

log, path = sys.argv[1], sys.argv[2]
cases, plans = [], defaultdict(dict)
for line in open(log, errors="ignore"):
    m = re.search(r"FAG_ZONES links=(\d) (\S+) (\S+) tiles=(\d+) rids=(\S+)", line)
    if m:
        cases.append(
            (int(m.group(1)), m.group(2), m.group(3), int(m.group(4)), [int(x) for x in m.group(5).split(",")])
        )
    m = re.search(r"FAG_PLAN links=(\d) (\S+) (\S+) dev=(\d+) p=(\d+) ports=(\S+) fwd=(.*) bwd=(.*)", line)
    if m:
        ports = {}
        for t in m.group(6).split(";"):
            k, xy = t.split("@")
            ports[tuple(int(v) for v in xy.split("."))] = k
        plans[(int(m.group(1)), m.group(2), m.group(3))][int(m.group(4))] = (
            int(m.group(5)),
            ports,
            m.group(7),
            m.group(8),
        )

with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    hdr = [h.strip() for h in next(rows)]
    ix = {h: i for i, h in enumerate(hdr)}
    wanted = {r for c in cases for r in c[4]}
    ev = defaultdict(list)  # (rid, chip, trace counter) -> rows
    for r in rows:
        if len(r) < len(hdr):
            continue
        try:
            rid = int(r[ix["run host ID"]]) >> 10
        except ValueError:
            continue
        if rid not in wanted:
            continue
        ev[(rid, int(r[ix["PCIe slot"]]), r[ix["trace id counter"]])].append(
            (
                int(r[ix["time[cycles since reset]"]]),
                r[ix["type"]],
                r[ix["zone name"]],
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]],
                r[ix["data"]],
            )
        )


def call_stats(evs, ports):
    """One call on one chip -> span, {port: {metric: us}}"""
    # the plan's virtual coordinates come from one chip's translation; a chip whose harvesting differs places a port
    # elsewhere: match the leftover port cores to the leftover named ports in x, y order
    seen = sorted({e[3] for e in evs if e[2].startswith("S_")})  # port cores (copy cores run R_ zones only)
    unknown = [c for c in seen if c not in ports]
    if unknown:
        missing = sorted((c, k) for c, k in ports.items() if c not in seen)
        ports = dict(ports)
        for c, (_, k) in zip(unknown, missing):
            ports[c] = k
    kern = [e[0] for e in evs if e[2].endswith("-KERNEL")]  # kernel markers: FW zones include waiting for launch
    t0, t1 = min(kern), max(kern)
    open_, out = {}, defaultdict(dict)
    for t, typ, z, core, risc, data in sorted(evs):
        if core not in ports:
            continue
        pk = ports[core]
        if typ == "ZONE_START":
            open_[(core, risc, z)] = t
        elif typ == "ZONE_END" and (core, risc, z) in open_:
            a = open_.pop((core, risc, z))
            if z.startswith(("S_", "R_")):
                out[pk][z] = (t - a) / freq
                if z == "S_e0":
                    out[pk]["start"] = (a - t0) / freq
                if z.startswith("S_e"):
                    out[pk]["send_end"] = (t - t0) / freq
        elif typ == "TS_DATA":
            v = int(data)
            out[pk][f"{z}{v >> 32}"] = (v & 0xFFFFFFFF) / freq
    sends = [e[0] for e in evs if e[2] == "S_e0" and e[1] == "ZONE_START"]
    out["_chip"]["work"] = (t1 - min(sends)) / freq if sends else float("nan")  # first send -> kernel end
    return (t1 - t0) / freq, out


med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
for links, v, shape, tiles, rids in cases:
    pl = plans.get((links, v, shape))
    if not pl:
        continue
    per = defaultdict(list)  # (chip) -> [(span, stats)]
    for (rid, chip, cnt), evs in ev.items():
        if rid in rids and chip in pl:
            per[chip].append(call_stats(evs, pl[chip][1]))
    if not per:
        continue
    spans = [max(s for s, _ in per[c][i : i + 1]) for c in per for i in range(len(per[c]))]
    print(
        f"\n=== links={links} {v} {shape} tiles={tiles}  calls/chip={min(len(x) for x in per.values())}  "
        f"span per chip (median us): "
        + " ".join(f"p{pl[c][0]}={med([s for s, _ in per[c]]):.2f}" for c in sorted(per, key=lambda c: pl[c][0]))
        + "  first send->end: "
        + " ".join(
            f"p{pl[c][0]}={med([st['_chip']['work'] for _, st in per[c]]):.2f}"
            for c in sorted(per, key=lambda c: pl[c][0])
        )
    )
    print(
        f"  {'chip':>4} {'port':>5} {'fence':>6} {'start':>6} {'S_e0':>18} {'S_e1':>18} {'S_e2':>18} {'sendEnd':>7} {'tail':>6}"
        f" {'R_e0':>20} {'R_e1':>20} {'R_e2':>20}"
    )
    for c in sorted(per, key=lambda c: pl[c][0]):
        for pk in sorted({k for _, st in per[c] for k in st} - {"_chip"}):
            g = lambda m: med([st[pk][m] for _, st in per[c] if m in st.get(pk, {})])  # noqa: E731

            def se(k):
                d = g(f"S_e{k}")
                return "-" if d != d else f"{d:6.2f}(s{g(f'S_slotwait{k}'):4.2f} c{g(f'S_cbwait{k}'):4.2f})"

            def re_(k):
                d = g(f"R_e{k}")
                return "-" if d != d else f"{d:5.2f}(a{g(f'R_arrwait{k}'):4.2f} c{g(f'R_cbwait{k}'):4.2f})"

            print(
                f"  p{pl[c][0]:>3} {pk:>5} {g('S_fence'):6.2f} {g('start'):6.2f} {se(0):>18} {se(1):>18} {se(2):>18} "
                f"{g('send_end'):7.2f} {g('S_tail'):6.2f} {re_(0):>20} {re_(1):>20} {re_(2):>20}"
            )
        print(f"        routes fwd={pl[c][2]} bwd={pl[c][3]}")
