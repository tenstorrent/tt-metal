"""python3 -I fusedec_moe_corr.py <run dir> -- per-chip moe_compute time vs the number of distinct active experts of the chip (sequential mapping, routes_round.json {layer: rows}):
linear fit time = a + b * n, per layer, with the identity device-id -> linear mesh id pairing (device id = row * 8 + col)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusedec_roundtab as RT

run = sys.argv[1]
O, pm, _ = RT.load(run)
T = RT.trace_table(run, O, "M", pm["M"])
R = json.load(open(os.path.join(run, "routes_round.json")))
chip = lambda e: ((e % 48) // 12) * 8 + e // 48
for oid, row, ph, per in T["ops"]:
    if row and "moe_compute" in row["op"] and ph.startswith("layer"):
        L = ph[5:]
        if L not in R:
            continue
        cnt = {}
        for r in R[L]:
            pass
        ids = {e for r in R[L] for e in r}
        n = np.zeros(32)
        for e in ids:
            n[chip(e)] += 1
        devs = sorted(per)
        t = np.array([per[d][0] for d in devs]) * 1e-3  # us
        nn = np.array([n[d % 32] for d in devs])
        a, b = (np.polyfit(nn, t, 1)[1], np.polyfit(nn, t, 1)[0]) if nn.std() > 0 else (t.mean(), 0)
        c = np.corrcoef(nn, t)[0, 1] if nn.std() > 0 and t.std() > 0 else float("nan")
        print(
            f"layer {L}: n per chip mean {nn.mean():.2f} max {nn.max():.0f}; time us mean {t.mean():.0f} min {t.min():.0f} max {t.max():.0f}; fit {a:.0f} + {b:.1f} us/expert; corr {c:.2f}"
        )
        print("   n:", nn.astype(int).tolist())
        print("   t:", [int(x) for x in t])
