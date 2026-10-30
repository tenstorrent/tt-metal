#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Edit a tt-metal NoC trace to price a change before building it: scale or drop the bytes of a traffic class,
or move its sources / destinations, then replay the edited trace in tt-npe and compare the link demand.

    noc_trace_whatif.py in.json out.json [--type WRITE_] [--src x,y ...] [--dst x,y ...] [--bytes 4096,16384]
                        [--from-us T0] [--to-us T1] [--scale-bytes 0.5] [--drop] [--remap-dst "x,y>x2,y2" ...]
                        [--remap-src "x,y>x2,y2" ...] [--clock-mhz 1350]

Every event matching ALL the given filters (type prefix, source set, destination set, byte sizes, time window) is
edited; the rest is copied. Examples:
  halve a hand-off:        --type WRITE_ --dst 3,2 --dst 4,2 --bytes 4096,8192,16384 --scale-bytes 0.5
  move a producer:         --src 5,2 --remap-src "5,2>1,2"
  reads from another port: --type READ --dst 0,3 --remap-dst "0,3>0,10"
  no o writes at all:      --type WRITE_ --bytes 2048 --drop
Multicast events (mcast_* fields) match --dst on their start coordinate.
"""
import argparse
import json


def parse_xy(s):
    x, y = s.split(",")
    return int(x), int(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_in")
    ap.add_argument("trace_out")
    ap.add_argument("--type", default="", help="event type prefix (WRITE_, READ, ...)")
    ap.add_argument("--src", action="append", default=[], help="x,y (repeatable)")
    ap.add_argument("--dst", action="append", default=[], help="x,y (repeatable)")
    ap.add_argument("--bytes", default="", help="comma list of num_bytes values to match")
    ap.add_argument("--from-us", type=float, default=None)
    ap.add_argument("--to-us", type=float, default=None)
    ap.add_argument("--scale-bytes", type=float, default=None)
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--remap-dst", action="append", default=[], help='"x,y>x2,y2" (repeatable)')
    ap.add_argument("--remap-src", action="append", default=[], help='"x,y>x2,y2" (repeatable)')
    ap.add_argument("--clock-mhz", type=float, default=1350.0)
    a = ap.parse_args()

    srcs = {parse_xy(s) for s in a.src}
    dsts = {parse_xy(s) for s in a.dst}
    sizes = {int(b) for b in a.bytes.split(",") if b}
    rd = {parse_xy(m.split(">")[0]): parse_xy(m.split(">")[1]) for m in a.remap_dst}
    rs = {parse_xy(m.split(">")[0]): parse_xy(m.split(">")[1]) for m in a.remap_src}

    ev = json.load(open(a.trace_in))
    t0 = min(e["timestamp"] for e in ev)

    def dst(e):
        if "dx" in e:
            return (e["dx"], e["dy"])
        if "mcast_start_x" in e:
            return (e["mcast_start_x"], e["mcast_start_y"])
        return None

    def matches(e):
        if not e.get("type") or (a.type and not e["type"].startswith(a.type)):
            return False
        if srcs and (e["sx"], e["sy"]) not in srcs:
            return False
        if dsts and dst(e) not in dsts:
            return False
        if sizes and e.get("num_bytes") not in sizes:
            return False
        t = (e["timestamp"] - t0) / a.clock_mhz
        if a.from_us is not None and t < a.from_us:
            return False
        if a.to_us is not None and t >= a.to_us:
            return False
        return True

    out, n_edit, n_drop = [], 0, 0
    for e in ev:
        if not matches(e):
            out.append(e)
            continue
        if a.drop:
            n_drop += 1
            continue
        if a.scale_bytes is not None and "num_bytes" in e:
            e["num_bytes"] = int(e["num_bytes"] * a.scale_bytes)
        if rd and "dx" in e and (e["dx"], e["dy"]) in rd:
            e["dx"], e["dy"] = rd[(e["dx"], e["dy"])]
        if rs and (e["sx"], e["sy"]) in rs:
            e["sx"], e["sy"] = rs[(e["sx"], e["sy"])]
        out.append(e)
        n_edit += 1
    json.dump(out, open(a.trace_out, "w"))
    print(f"{len(ev)} events: {n_edit} edited, {n_drop} dropped -> {a.trace_out}")


if __name__ == "__main__":
    main()
