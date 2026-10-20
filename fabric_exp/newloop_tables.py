#!/usr/bin/env python3
"""§29-style 4352 B GB/s tables from the n_* runs (new benchmark loop), with the change against the g_* runs
(old loop, §29). Last row per test and ntype is the trimmed replay pass."""
import csv, os

R = "/home/aagarwal/tt-metal/fabric_exp/results"
CONFIGS = [
    ("stockka8000", "stock, ka 8000"),
    ("stock", "stock"),
    ("546176", "546176"),
    ("4740480", "4740480"),
    ("13129088", "13129088"),
    ("13129088w32", "13129088 W32/fwd8"),
    ("15226240w32", "15226240 W32/fwd8 (blind)"),
]
TESTS = [
    ("Mesh1HopEW", "1-hop E/W"),
    ("Mesh1HopNS", "1-hop N/S"),
    ("Mesh1HopEWBidir", "bidirectional"),
    ("Mesh3HopEW", "3-hop"),
    ("MeshMulticast", "multicast"),
]


def load(label):
    f = os.path.join(R, label + ".csv")
    out = {}
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            out[(r["test_name"], "scatter" if "SCATTER" in r["ntype"] else "write")] = float(
                r["avg_bandwidth_gigabytes_per_s"]
            )
    return out


for trim, title in [("_trim", "trimmed"), ("", "untrimmed")]:
    print(f"\n4352 B, {title}, GB/s write / scatter (change vs §29 in %):\n")
    print("| | " + " | ".join(name for _, name in CONFIGS) + " |")
    print("| --- |" + " ---: |" * len(CONFIGS))
    for test, tname in TESTS:
        cells = []
        for cfg, _ in CONFIGS:
            new, old = load(f"n_{cfg}_4352{trim}"), load(f"g_{cfg}_4352{trim}")
            if (test, "write") not in new:
                cells.append("-")
                continue
            parts = []
            for kind in ("write", "scatter"):
                n = new[(test, kind)]
                o = old.get((test, kind))
                d = f" ({100 * (n - o) / o:+.1f})" if o and abs(100 * (n - o) / o) >= 0.5 else ""
                parts.append(f"{n:.1f}{d}")
            cells.append(" / ".join(parts))
        print(f"| {tname} | " + " | ".join(cells) + " |")
