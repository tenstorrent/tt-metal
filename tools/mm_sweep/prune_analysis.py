#!/usr/bin/env python3
"""Mine bh_joint.json's per-config 'all' data to find a SAFE reduced search space.
Q: which (S,Pk) ever win? how much does (S,Pk)!=(1,1) help, esp. for grid-full shapes? do blocks cluster?
   python tools/mm_sweep/prune_analysis.py [bh_joint.json]
"""
import json, sys
from collections import Counter, defaultdict

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tools/mm_sweep/bh_joint.json"))


def best_of(recs):
    return max(recs, key=lambda r: r["util"]) if recs else None


print(f"{len(d)} shapes\n")
win_spk = Counter()
spk_value = []  # (shape, out, cores, skew, best_util, best_spk, util_at_11, regret_if_11)
near_win_spk = Counter()  # (S,Pk) within 2% of best for that shape -> "could be pruned to these"
for key, r in d.items():
    allc = r.get("all") or []
    if not allc:
        continue
    Mt, Nt, Kt = r["MtNtKt"]
    cores = r["grid"][0] * r["grid"][1]
    out = Mt * Nt
    small, big = min(Mt, Nt), max(Mt, Nt)
    b = best_of(allc)
    win_spk[(b["S"], b["Pk"])] += 1
    # best achievable restricted to (1,1)
    c11 = best_of([c for c in allc if c["S"] == 1 and c["Pk"] == 1])
    u11 = c11["util"] if c11 else 0.0
    regret = b["util"] / u11 if u11 else float("inf")
    spk_value.append((key, out, cores, big / small if small else 1, b["util"], (b["S"], b["Pk"]), u11, regret))
    thr = b["util"] * 0.98
    for c in allc:
        if c["util"] >= thr:
            near_win_spk[(c["S"], c["Pk"])] += 1

print("== winning (S,Pk) across shapes ==")
for spk, n in win_spk.most_common():
    print(f"  {spk}: {n}")

print("\n== how much (S,Pk)!=(1,1) helps, by grid occupancy ==")
print(f"{'shape':<18}{'out':>7}{'skew':>6}{'best%':>7}{'bestSPk':>9}{'(1,1)%':>8}{'gain':>7}")
for key, out, cores, skew, bu, bspk, u11, regret in sorted(
    spk_value, key=lambda x: -x[7] if x[7] != float("inf") else 0
):
    full = "FULL" if out >= cores else "starv"
    rg = f"{regret:.2f}x" if regret != float("inf") else "inf"
    print(f"{key:<18}{out:>7}{skew:>6.0f}{bu:>7.1f}{str(bspk):>9}{u11:>8.1f}{rg:>7}  {full}")

# grid-full shapes: does (1,1) ever lose meaningfully?
full_regrets = [x[7] for x in spk_value if x[1] >= x[2] and x[7] != float("inf")]
starv_regrets = [x[7] for x in spk_value if x[1] < x[2] and x[7] != float("inf")]
if full_regrets:
    print(
        f"\nGRID-FULL (out>=cores) shapes: {len(full_regrets)}; "
        f"max gain from slicing/Kpar = {max(full_regrets):.2f}x, "
        f">2% better than (1,1): {sum(1 for x in full_regrets if x>1.02)}"
    )
if starv_regrets:
    print(
        f"OUTPUT-STARVED (out<cores) shapes: {len(starv_regrets)}; "
        f"max gain = {max(starv_regrets):.2f}x, median {sorted(starv_regrets)[len(starv_regrets)//2]:.2f}x"
    )

print("\n== (S,Pk) appearing within 2% of best (prunable target set) ==")
for spk, n in near_win_spk.most_common():
    print(f"  {spk}: {n}")
