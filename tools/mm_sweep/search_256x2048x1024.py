#!/usr/bin/env python3
# Bounded Sm>1 config search for 256x2048x1024 on the FINAL kernel (coalesce + corrected forward-signal-first),
# to finish the pinned M-split follow-up. Fresh cache (no seeding). Candidate set: current table pick +
# best-per-(Pk,Ns,Sm) factorization (cost-ranked kb/nsb rep) + cost-top Sm>1 + a few top Sm=1 for reference.
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import regime_a_bench as rb
import picker_v2 as pv

M, K, N = 256, 2048, 1024
CUR = (1, 4, 2, 2, 4)  # current table pick
CACHE = f"{os.path.dirname(os.path.abspath(__file__))}/search_256x2048x1024_cache.json"


def cost(c):
    try:
        v = pv.cost(M, K, N, c, pv.bestP)
        return v if v is not None else 1e18
    except Exception:
        return 1e18


feas = [tuple(c) for c in rb.enumerate_feasible(M, K, N)]
ranked = sorted(feas, key=cost)
# best (Pk,Ns,Sm) factorization rep (cost-min kb/nsb), for Sm>1 and Sm=1
per_fact = {}
for c in ranked:
    Ns, Pk, Sm, kb, nsb = c
    per_fact.setdefault((Pk, Ns, Sm), c)
sm_facts = sorted([c for f, c in per_fact.items() if f[2] > 1], key=cost)
sm1_facts = sorted([c for f, c in per_fact.items() if f[2] == 1], key=cost)
sm_top = [c for c in ranked if c[2] > 1][:12]
cands, seen = [], set()
for c in [CUR] + sm_facts + sm_top + sm1_facts[:8]:
    c = tuple(c)
    if c not in seen and rb.planner_feasible(M, K, N, c)[0]:
        seen.add(c)
        cands.append(c)
    if len(cands) >= 46:
        break

if not os.path.exists(CACHE):
    json.dump({}, open(CACHE, "w"))
cache = rb.load_cache(CACHE)
print(f"256x2048x1024 bounded search on final kernel: {len(cands)} candidates (current={CUR})", flush=True)
res = []
for i, c in enumerate(cands):
    r = rb.run_cfg(M, K, N, c, cache)
    if rb._ok(r):
        res.append({"cfg": list(c), "us": r["us_med"], "pct512": r["pct512"], "pcc": r["pcc"]})
        print(f"  [{i+1}/{len(cands)}] {c} {r['us_med']:.2f}us {r['pct512']:.0f}% pcc={r['pcc']:.4f}", flush=True)
res.sort(key=lambda x: x["us"])
cur_r = rb.run_cfg(M, K, N, CUR, cache)
cur_us = cur_r["us_med"] if rb._ok(cur_r) else None
best = res[0] if res else None
gain = ((cur_us / best["us"] - 1) * 100) if (best and cur_us) else None
print(f"\nCURRENT {CUR} = {cur_us and round(cur_us,2)}us", flush=True)
print(
    f"BEST    {best and best['cfg']} = {best and round(best['us'],2)}us  (gain vs current = {gain and round(gain,1)}%)",
    flush=True,
)
print("TOP 6:", flush=True)
for r in res[:6]:
    print(f"  {r['cfg']} {r['us']:.2f}us {r['pct512']:.0f}% pcc={r['pcc']:.4f}", flush=True)
json.dump(
    {"current": list(CUR), "current_us": cur_us, "best": best, "gain_pct": gain, "results": res},
    open(f"{os.path.dirname(os.path.abspath(__file__))}/search_256x2048x1024.json", "w"),
    indent=2,
)
print("SEARCH DONE", flush=True)
