#!/usr/bin/env python3
"""Bits 24/25: cycles/packet (1.35 GHz / packets per s) for the 4352 B set, against the same-session baseline
(13129088 for stock RX, 15226240 W32 for blind push). Last row per test and ntype = trimmed replay."""
import csv, os

R = "/home/aagarwal/tt-metal/fabric_exp/results"
STOCK = [("13129088", "base 13129088"), ("29906304", "+24"), ("46683520", "+25"), ("63460736", "+24+25")]
BLIND = [("15226240w32", "blind base"), ("32003456w32", "blind +24"), ("65557888w32", "blind +24+25")]
TESTS = [
    ("Mesh1HopEW", "1-hop E/W"),
    ("Mesh1HopNS", "1-hop N/S"),
    ("Mesh1HopEWBidir", "bidir"),
    ("Mesh3HopEW", "3-hop"),
    ("MeshMulticast", "multicast"),
]


def load(label):
    f = os.path.join(R, label + ".csv")
    out = {}
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            out[(r["test_name"], "scatter" if "SCATTER" in r["ntype"] else "write")] = 1.35e9 / float(
                r["avg_packets_per_s"]
            )
    return out


for trim, title in [("_trim", "trimmed"), ("", "untrimmed")]:
    for cfgs, forwarded in [(STOCK, True), (BLIND, False)]:
        print(f"\n4352 B {title}, cycles/packet write / scatter (change vs the first column):\n")
        print("| | " + " | ".join(n for _, n in cfgs) + " |")
        print("| --- |" + " ---: |" * len(cfgs))
        base = load(f"b_{cfgs[0][0]}_4352{trim}")
        for test, tname in TESTS:
            if not forwarded and test in ("Mesh3HopEW", "MeshMulticast"):
                continue
            cells = []
            for cfg, _ in cfgs:
                d = load(f"b_{cfg}_4352{trim}")
                if (test, "write") not in d:
                    cells.append("-")
                    continue
                parts = []
                for kind in ("write", "scatter"):
                    v, b = d[(test, kind)], base.get((test, kind))
                    parts.append(
                        f"{v:.1f}" + (f" ({v - b:+.1f})" if b and cfg != cfgs[0][0] and abs(v - b) >= 0.3 else "")
                    )
                cells.append(" / ".join(parts))
            print(f"| {tname} | " + " | ".join(cells) + " |")
