#!/usr/bin/env python3
"""Back-test candidate search-space reductions against the full-oracle 'all' data already swept.
For each reduction: regret = full_best_util / reduced_best_util per shape (1.00 = no loss), and the
fraction of configs kept (== runtime fraction). Picks the prune that's ~free on the done shapes.
   python tools/mm_sweep/prune_backtest.py [bh_joint.json]
"""
import json, sys, statistics

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tools/mm_sweep/bh_joint.json"))
SPK5 = {(1, 1), (2, 1), (5, 1), (1, 2), (1, 5)}  # the 5 (S,Pk) ever within 2% of best


def keep(c, spk_set=None, kb_set=None, mb_max=None, nb_max=None):
    if spk_set is not None and (c["S"], c["Pk"]) not in spk_set:
        return False
    if kb_set is not None and c["kb"] not in kb_set:
        return False
    if mb_max is not None and c["mb"] > mb_max:
        return False
    if nb_max is not None and c["nb"] > nb_max:
        return False
    return True


REDUCTIONS = {
    "R0 full (sanity)": dict(),
    "R1 SPk5": dict(spk_set=SPK5),
    "R2 SPk5 + kb<=8": dict(spk_set=SPK5, kb_set={4, 8}),
    "R3 SPk5 + kb<=8 + mb<=32,nb<=24": dict(spk_set=SPK5, kb_set={4, 8}, mb_max=32, nb_max=24),
    "R4 SPk5 + mb<=32,nb<=24": dict(spk_set=SPK5, mb_max=32, nb_max=24),
}

for name, kw in REDUCTIONS.items():
    regrets, kept_frac, worst = [], [], []
    for key, r in d.items():
        allc = r.get("all") or []
        if not allc:
            continue
        full_best = max(c["util"] for c in allc)
        red = [c for c in allc if keep(c, **kw)]
        if not red:
            regrets.append(float("inf"))
            continue
        rb = max(c["util"] for c in red)
        rg = full_best / rb if rb else float("inf")
        regrets.append(rg)
        kept_frac.append(len(red) / len(allc))
        worst.append((rg, key, full_best, rb))
    fin = [x for x in regrets if x != float("inf")]
    worst.sort(reverse=True)
    kf = statistics.mean(kept_frac) if kept_frac else 0
    print(f"\n{name}: kept {kf*100:.0f}% of configs (~{kf*100:.0f}% runtime)")
    print(
        f"   regret geomean {statistics.geometric_mean(fin):.3f}  max {max(fin):.3f}  "
        f">2% loss on {sum(1 for x in fin if x>1.02)} shapes; inf(empty) on {sum(1 for x in regrets if x==float('inf'))}"
    )
    for rg, key, fb, rb in worst[:3]:
        if rg > 1.02:
            print(f"     worst: {key} full {fb:.1f}% -> reduced {rb:.1f}% ({rg:.2f}x)")
