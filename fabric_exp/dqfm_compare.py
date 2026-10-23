#!/usr/bin/env python3
"""§64: flushed-signal benchmark (streams/flush_mix.yaml) on three router versions: before (dqfm_before), §62
flush-only staging (dqfm_s62), §63 deferred atomic (dqfm_s63). Cycles/packet per row; for each chunk shape, the
flush cost = flushed minus unflushed, per packet and per signal (x chunk length)."""
import csv, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = [("before", "dqfm_before"), ("§62", "dqfm_s62"), ("§63", "dqfm_s63")]
CHUNK = {"Fused": 1, "Chunk8Fused": 8, "Chunk16Fused": 16, "Chunk8Atomic": 8}
d = {}
for n, p in V:
    d[n] = {
        r["test_name"]: 1.35e9 / float(r["avg_packets_per_s"])
        for r in csv.DictReader(open(os.path.join(R, p + ".csv")))
    }
print(f"{'row':26s}" + "".join(f"{n:>9s}" for n, _ in V))
for t in d["before"]:
    print(f"{t:26s}" + "".join(f"{d[n][t]:9.1f}" for n, _ in V))
print("\nflush cost (flushed - unflushed): per packet / per flushed signal")
for direction in ("EW", "NS"):
    for shape, k in CHUNK.items():
        cells = []
        for n, _ in V:
            on, off = d[n][f"{direction}_{shape}_Flush"], d[n][f"{direction}_{shape}_NoFlush"]
            cells.append(f"{n} {on - off:6.1f} / {(on - off) * k:6.1f}")
        print(f"   {direction}_{shape:14s} " + "   ".join(cells))
