#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a tt-metal NoC trace contains, before any simulation: event types and bytes, the destinations (DRAM ports
show up by their coordinates), and per-core inbound / outbound bytes per time window, so the traffic classes of a
kernel can be named and their link loads estimated by hand.

    noc_trace_summary.py noc_trace_dev0_ID1234.json [--bin 4] [--top 10] [--core x,y ...] [--clock-mhz 1350]

The trace is the flat JSON array the device profiler writes to generated/profiler/.logs/ when
TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 (one file per device and program). Each event has proc, sx, sy, noc, type,
num_bytes, timestamp and either dx, dy (unicast) or mcast_* (multicast); zone markers have no type. Known gaps: a
rectangle multicast may appear as one unicast event per destination core, semaphore increments / sets are not
traced, READ_WITH_STATE events carry 0 bytes (the bytes are on their READ_SET_STATE).
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--bin", type=float, default=4.0, help="window in us")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--core", action="append", default=[], help="x,y: print this core's inbound / outbound per window")
    ap.add_argument("--clock-mhz", type=float, default=1350.0)
    a = ap.parse_args()
    clk = a.clock_mhz

    ev = json.load(open(a.trace))
    typed = [e for e in ev if e.get("type")]
    t0 = min(e["timestamp"] for e in ev)
    t1 = max(e["timestamp"] for e in ev)
    print(
        f"{len(ev)} events ({len(typed)} with a type) over {(t1 - t0) / clk:.1f} us; sources: "
        f"{len({(e['sx'], e['sy']) for e in ev})} cores"
    )
    by_type = collections.Counter(e["type"] for e in typed)
    bytes_by_type = collections.Counter()
    for e in typed:
        bytes_by_type[e["type"]] += e.get("num_bytes", 0)
    print("\n== events by type (count, MB):")
    for t, n in by_type.most_common():
        print(f"  {t:<28} {n:7d}  {bytes_by_type[t] / 1e6:8.2f} MB")
    print("\n== by processor and NoC:")
    for (p, n), c in collections.Counter((e["proc"], e.get("noc")) for e in typed).most_common():
        print(f"  {p:<8} {str(n):<6} {c:7d}")

    def dst(e):
        if "dx" in e:
            return (e["dx"], e["dy"])
        if "mcast_start_x" in e:
            return (f"mc{e['mcast_start_x']}-{e['mcast_end_x']}", f"{e['mcast_start_y']}-{e['mcast_end_y']}")
        return None

    reads = [e for e in typed if e["type"].startswith("READ") and dst(e)]
    writes = [e for e in typed if e["type"].startswith("WRITE") and dst(e)]
    print(f"\n== READ destinations (top {a.top}; DRAM ports appear by coordinate):")
    for d, n in collections.Counter(dst(e) for e in reads).most_common(a.top):
        print(f"  {str(d):<14} {n:6d} reads")
    print(f"\n== WRITE destinations by bytes (top {a.top}):")
    wb = collections.Counter()
    ws = collections.defaultdict(set)
    for e in writes:
        wb[dst(e)] += e.get("num_bytes", 0)
        ws[dst(e)].add((e["sx"], e["sy"]))
    for d, b in wb.most_common(a.top):
        print(f"  {str(d):<14} {b / 1e6:8.2f} MB from {len(ws[d])} sources")

    # per-core inbound (writes landing) and outbound (writes + reads issued) bytes per window
    nbin = int((t1 - t0) / clk / a.bin) + 1
    inb = collections.defaultdict(lambda: [0] * nbin)
    outb = collections.defaultdict(lambda: [0] * nbin)
    for e in typed:
        w = int((e["timestamp"] - t0) / clk / a.bin)
        b = e.get("num_bytes", 0)
        outb[(e["sx"], e["sy"])][w] += b
        d = dst(e)
        if e["type"].startswith("WRITE") and isinstance(d, tuple) and isinstance(d[0], int):
            inb[d][w] += b
    gbs = lambda v: f"{v / a.bin / 1e3:4.0f}"
    hdr = "  core      | " + " ".join(f"{int(i * a.bin):4d}" for i in range(nbin))
    print(f"\n== inbound write GB/s per {a.bin:g} us window, top {a.top} cores by total:")
    print(hdr)
    for c, v in sorted(inb.items(), key=lambda kv: -sum(kv[1]))[: a.top]:
        print(f"  {str(c):<9} | " + " ".join(gbs(x) for x in v))
    print(f"\n== outbound (reads issued + writes) GB/s per window, top {a.top} cores by total:")
    print(hdr)
    for c, v in sorted(outb.items(), key=lambda kv: -sum(kv[1]))[: a.top]:
        print(f"  {str(c):<9} | " + " ".join(gbs(x) for x in v))
    for cs in a.core:
        c = tuple(int(x) for x in cs.split(","))
        print(f"\n== core {c}: inbound | outbound GB/s per window")
        print(hdr)
        print(f"  in        | " + " ".join(gbs(x) for x in inb[c]))
        print(f"  out       | " + " ".join(gbs(x) for x in outb[c]))
        srcs = collections.Counter()
        for e in writes:
            if dst(e) == c:
                srcs[(e["sx"], e["sy"])] += e.get("num_bytes", 0)
        print("  writers:  " + ", ".join(f"{s}:{b / 1e6:.1f}MB" for s, b in srcs.most_common(12)))


if __name__ == "__main__":
    main()
