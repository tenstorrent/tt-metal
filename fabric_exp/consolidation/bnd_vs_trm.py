#!/usr/bin/env python3
"""bnd_vs_trm.py [SUITE ...]: the lead's bound map (bnd_delay_runs.sh, v4bdd) against v4trm's (prof_delay_runs.sh,
v4trmd, a09u32 jobs 172-174), per row: base rate, c32 on each side and the side read (prof_report.py's rule), and
whether the side changed. Both at 32-cycle spins; v4trm's from results/prof_<suite>_<arm>.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bnd_report as br

R = br.R


def trm(s, a):
    if s in ("vc1",):
        return br.rows(f"{R}/profvc1_{a}_default_p1_o3_13129088_vc1_tests.csv")
    return br.rows(f"{R}/prof_{s}_{a}.csv")


def side(d, k):
    b = d["base"][k]
    t, r = d["tx32"][k] - b, d["rx32"][k] - b
    return b, t, r, br.side(max(0.0, 44 - t), max(0.0, 44 - r))


suites = sys.argv[1:] or ["4352", "1d", "mix", "ring", "torusy", "4352_1e", "1d_1e", "rse", "mux", "vc1"]
for s in suites:
    L = {a: br.load(s, a) for a in ("base", "tx32", "rx32")}
    T = {a: trm(s, a) for a in ("base", "tx32", "rx32")}
    if not all(L.values()) or not all(T.values()):
        continue
    print(f"\n## {s}")
    print(f"{'row':48}{'trm base':>9}{'tx':>6}{'rx':>6} {'side':5}{'lead base':>10}{'tx':>6}{'rx':>6} {'side':5} change")
    n = c = 0
    for k in L["base"]:
        if not all(k in d for d in list(L.values()) + list(T.values())):
            continue
        tb, tt, tr, ts = side(T, k)
        lb, lt, lr, ls = side(L, k)
        n += 1
        ch = "" if ts == ls else f"{ts} -> {ls}"
        c += ts != ls
        lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
        print(f"{lab[:48]:48}{tb:9.1f}{tt:+6.1f}{tr:+6.1f} {ts:5}{lb:10.1f}{lt:+6.1f}{lr:+6.1f} {ls:5} {ch}")
    print(f"  {c} of {n} rows changed side")
