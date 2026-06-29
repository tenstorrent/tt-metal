#!/usr/bin/env python3
"""Markdown report of the BH joint sweep results so far (joint_sweep.py bh_joint.json).
Sorted ASCENDING by oracle util. Shows winning (S,Pk) + blocking, and straggler analysis.
   python tools/mm_sweep/report_bh.py [bh_joint.json] [out.md]
"""
import json, sys, math
from collections import Counter

import os

PATH = sys.argv[1] if len(sys.argv) > 1 else "tools/mm_sweep/bh_joint.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tools/mm_sweep/bh_results_early.md"
PEAK_BW = float(os.environ.get("MM_PEAK_BW_GBPS", 500)) * 1e9  # DRAM BW for the BW-util column
d = json.load(open(PATH))


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


rows = []
for key, r in d.items():
    b, a = r.get("best"), r.get("auto")
    if not b:
        continue
    Mt, Nt, Kt = r["MtNtKt"]
    M, K, N = r["shape"]
    small, big = min(Mt, Nt), max(Mt, Nt)
    # minimum DRAM traffic (each operand moved once, bf16): in0 + in1 + out
    min_bytes = (M * K + K * N + M * N) * 2
    gbps = min_bytes / (b["us"] * 1e-6) / 1e9  # effective BW assuming minimal traffic
    rows.append(
        {
            "shape": key,
            "M": M,
            "K": K,
            "N": N,
            "out": Mt * Nt,
            "Kt": Kt,
            "cores": r["grid"][0] * r["grid"][1],
            "skew": big / small if small else 1.0,
            "bS": b["S"],
            "bPk": b["Pk"],
            "blk": f"{b['mb']}/{b['kb']}/{b['nb']}",
            "sb": f"{b['sbh']}x{b['sbw']}",
            "util": b["util"],
            "us": b["us"],
            "gbps": gbps,
            "bwpct": 100 * gbps * 1e9 / PEAK_BW,
            "auto": a["util"] if a else None,
            "bva": r.get("best_vs_auto"),
            "heur": tuple(r["heuristic_SPk"]),
            "cache": b.get("cache_pcc", 1.0) < 0.99,
            "ncfg": r.get("n_configs"),
        }
    )

rows.sort(key=lambda x: x["util"])  # ascending by util
peak = next((d[k]["peak_tflops"] for k in d), 0)
L = []
L.append(f"# BH minimal_matmul joint sweep — early results")
L.append(
    f"\n**{len(rows)} shapes** (sweep in progress) · grid 11×10 · peak {peak:.0f} TFLOP/s · "
    f"`*`=oracle config on fused-K cache-bug path (timing valid)\n"
)

util_all = [x["util"] for x in rows]
maxpar = [x for x in rows if x["bS"] * x["bPk"] == d[x["shape"]]["grid"][1]]
L.append(
    "> Results are post-fix: fused split-K program-cache bug (df8fcb2) **and** the "
    "`rows_per_group==1` off-by-one — so `S·Pk = grid.y` max-parallelism partitions "
    "((10,1),(5,2),(2,5),(1,10)) are now included (they previously crashed)."
)
L.append(f"- oracle util: geomean **{gm(util_all):.1f}%**, min **{min(util_all):.1f}%**, max **{max(util_all):.1f}%**")
L.append(f"- best_vs_auto: geomean **{gm([x['bva'] for x in rows if x['bva']]):.2f}x**")
L.append(
    f"- **max-parallelism (`S·Pk=grid.y`) is the oracle best on {len(maxpar)}/{len(rows)} shapes** "
    f"(newly unlocked): {', '.join(x['shape'] for x in maxpar)}"
)


# ---- straggler tiers ----
def tier(u):
    return "🔴 <5%" if u < 5 else "🟠 5-20%" if u < 20 else "🟡 20-40%" if u < 40 else "🟢 ≥40%"


tc = Counter(tier(x["util"]) for x in rows)
L.append("\n## Util distribution")
for t in ["🔴 <5%", "🟠 5-20%", "🟡 20-40%", "🟢 ≥40%"]:
    if tc.get(t):
        L.append(f"- {t}: {tc[t]} shapes")

# ---- main table, ascending ----
L.append(f"\n## All shapes — ascending by oracle util")
L.append(
    f"BW% = effective DRAM BW (min traffic / best µs) vs {PEAK_BW/1e9:.0f} GB/s. "
    f"**Low math-util + low BW% = real headroom; low math-util + high BW% = memory-wall-bound (structural).**"
)
L.append("| util% | BW% | GB/s | shape (M×K×N) | out·Kt | skew | best S/Pk | block | subblk | µs | best/auto | notes |")
L.append("|--:|--:|--:|---|--:|--:|:-:|:-:|:-:|--:|--:|---|")
for x in rows:
    notes = []
    if x["util"] < 20 and x["bwpct"] < 55:
        notes.append("**HEADROOM**")
    if x["util"] < 20 and x["bwpct"] >= 75:
        notes.append("mem-wall")
    if x["cache"]:
        notes.append("Kfused*")
    if x["heur"] != (x["bS"], x["bPk"]):
        notes.append("heur≠")
    if x["bva"] and x["bva"] < 0.97:
        notes.append("AUTO>oracle")
    bv = f"{x['bva']:.2f}x" if x["bva"] else "—"
    L.append(
        f"| **{x['util']:.1f}** | {x['bwpct']:.0f} | {x['gbps']:.0f} | {x['M']}×{x['K']}×{x['N']} | "
        f"{x['out']}·{x['Kt']} | {x['skew']:.0f} | ({x['bS']},{x['bPk']}) | {x['blk']} | {x['sb']} | "
        f"{x['us']:.0f} | {bv} | {' '.join(notes)} |"
    )

# ---- headroom shortlist: low math-util but NOT memory-bound ----
hr = sorted([x for x in rows if x["util"] < 20 and x["bwpct"] < 55], key=lambda x: x["bwpct"])
L.append(f"\n## 🎯 Headroom shortlist — low math-util AND low BW% ({len(hr)} shapes)")
L.append(
    "Not saturating memory at best config → the bottleneck is overhead / re-reads / poor overlap, "
    "not the BW wall. These are where tuning (not algorithm change) can still win."
)
L.append("| math% | BW% | GB/s | shape | best S/Pk | µs | skew |")
L.append("|--:|--:|--:|---|:-:|--:|--:|")
for x in hr:
    L.append(
        f"| {x['util']:.1f} | {x['bwpct']:.0f} | {x['gbps']:.0f} | {x['shape']} | "
        f"({x['bS']},{x['bPk']}) | {x['us']:.0f} | {x['skew']:.0f} |"
    )

# ---- winning (S,Pk) ----
L.append("\n## Winning (S,Pk)")
spk = Counter((x["bS"], x["bPk"]) for x in rows)
L.append("| S,Pk | #wins | example shapes |")
L.append("|:-:|--:|---|")
for k, n in spk.most_common():
    ex = ", ".join(x["shape"] for x in rows if (x["bS"], x["bPk"]) == k)[:70]
    L.append(f"| {k} | {n} | {ex} |")

# ---- winning blocking ----
L.append("\n## Winning blocking")
kb = Counter(x["blk"].split("/")[1] for x in rows)
mb = Counter(x["blk"].split("/")[0] for x in rows)
nb = Counter(x["blk"].split("/")[2] for x in rows)
sb = Counter(x["sb"] for x in rows)
L.append(f"- **K_block** winners: " + ", ".join(f"kb={k}×{n}" for k, n in kb.most_common()))
L.append(f"- **M_block** winners: " + ", ".join(f"{k}×{n}" for k, n in mb.most_common()))
L.append(f"- **N_block** winners: " + ", ".join(f"{k}×{n}" for k, n in nb.most_common()))
L.append(f"- **subblock** winners: " + ", ".join(f"{k}×{n}" for k, n in sb.most_common()))
L.append(
    f"- most common full blocking: "
    + ", ".join(f"`{k}`×{n}" for k, n in Counter(x["blk"] for x in rows).most_common(6))
)

# ---- stragglers ----
strag = [x for x in rows if x["util"] < 5]
L.append(f"\n## 🔴 Bigtime stragglers (util < 5%) — {len(strag)} shapes")
L.append("These are output-tiny / extreme-skew shapes where even the oracle can't fill the machine.")
L.append("| util% | shape | out tiles | skew | best S/Pk | why |")
L.append("|--:|---|--:|--:|:-:|---|")
for x in strag:
    why = []
    if x["out"] < x["cores"]:
        why.append(f"out {x['out']}<110 cores (starved)")
    if x["skew"] >= 24:
        why.append(f"skew {x['skew']:.0f}")
    if x["M"] // 32 <= 2:
        why.append(f"M={x['M']//32}tile")
    L.append(
        f"| {x['util']:.1f} | {x['shape']} | {x['out']} | {x['skew']:.0f} | ({x['bS']},{x['bPk']}) | {'; '.join(why)} |"
    )

open(OUT, "w").write("\n".join(L))
print(f"wrote {OUT}  ({len(rows)} shapes)")
print("\n".join(L[:14]))
