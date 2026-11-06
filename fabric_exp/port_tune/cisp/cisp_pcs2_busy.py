#!/usr/bin/env python3
"""cisp_pcs2_busy.py RESULTS_DIR: NX 4096 B write L2 rows, stall-event vs quiet router-iterations: busy share
(E0 sender::send, E1 receiver handle/forward) by decile of the iteration window and the top not-busy pass lines."""
import sys, os, re, collections, statistics
import numpy as np
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune/cisp")
sys.argv = ["x", sys.argv[1]]
import cisp_pcs2_report as P
import cisp_router_map as rmap
D = sys.argv[1]
peer, c2chip = rmap.topology()
meta, rec, t_abs, tg_info, bucket_of, line_of = P.load("on")
win = P.windows(os.path.join(D, "pcs2_on.log"))
on_it = rmap.iterations(os.path.join(D, "pcs2_on.log")); off_it = rmap.iterations(os.path.join(D, "pcs2_off.log"))
tidx = {(t["chip"], t["chan"], t["erisc"]): i for i, t in tg_info.items() if t["matched"]}
def busy(j, e):
    key = line_of.get(j)
    if not key or key[0] == "fw": return False
    fns = [f for f, _ in P._chains[key]]
    if e == 0: return any(f.endswith("sender::send") for f in fns)
    return any(f.split("::")[-1] in ("handle", "forward_all", "receive", "deliver", "forward") for f in fns)
def inner(j):
    key = line_of.get(j)
    if not key: return "?"
    if key[0] == "fw": return f"fw 0x{key[1] & ~0xff:x}"
    f, l = P._chains[key][0]; return f"{f} {l}"
stats = {e: {"ev": collections.Counter(), "ot": collections.Counter()} for e in (0, 1)}
dec = {e: {"ev": np.zeros((10, 2)), "ot": np.zeros((10, 2))} for e in (0, 1)}
lines = {e: {"ev": collections.Counter(), "ot": collections.Counter()} for e in (0, 1)}
rows_only = re.compile(r"size_4096_ntype_unicast_write_num_links_2$")
nev = 0
for test, k, a, b in win:
    if test not in off_it or k >= len(on_it.get(test, [])): continue
    for rt, cyc in on_it[test][k].items():
        base = statistics.median(it[rt] for it in off_it[test] if rt in it)
        (r, c), ch = rt; chip = c2chip[(r, c)]
        ev = cyc - base > 300e3
        if not ev and not rows_only.search(test): continue  # compare against the same row's quiet routers
        if ev and not rows_only.search(test): continue
        nev += ev
        g = "ev" if ev else "ot"
        for e in (0, 1):
            i = tidx.get((chip, ch, e))
            if i is None: continue
            sel = np.nonzero((rec["i"] == i) & (t_abs >= a) & (t_abs <= b))[0]
            for j in sel.tolist():
                bz = busy(j, e)
                stats[e][g]["busy" if bz else "not"] += 1
                d = min(9, int(10 * (t_abs[j] - a) / max(1e-9, b - a)))
                dec[e][g][d] += (1, bz)
                if not bz and P.ORDER[bucket_of[j]] == "passes": lines[e][g][inner(j)] += 1
print(f"NX 4096 B write L2 rows: {nev} event router-iterations")
for e in (0, 1):
    for g in ("ev", "ot"):
        n = sum(stats[e][g].values()); print(f"ERISC{e} {g}: samples {n}, busy (E0 sender::send / E1 handle,forward) {stats[e][g]['busy'] / max(1, n):.1%}")
        print("   busy share by decile of the window: " + " ".join(f"{x[1] / max(1, x[0]):.2f}" for x in dec[e][g]) + "   (samples " + " ".join(str(int(x[0])) for x in dec[e][g]) + ")")
    for g in ("ev", "ot"):
        n = sum(lines[e][g].values())
        print(f"   ERISC{e} {g}: top not-busy pass lines (share of not-busy pass samples, n {n})")
        for l, c in lines[e][g].most_common(8): print(f"      {100 * c / max(1, n):6.2f}%  {l}")
