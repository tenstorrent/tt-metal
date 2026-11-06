#!/usr/bin/env python3
"""sab_report.py [--prefix sab] [--md]: the single-ERISC A/B table (single_ab_runs.sh; local, never committed).

Per row of the suites 4352_1e 1d_1e rse mux: cycles/packet (1.35 GHz, census_report.py's rows()) of main, A (ba486eb70ba
as built) and B (inline passes), each pass a / b and the mean; B - A in cycles and percent; A / main and B / main.
Rows where B is slower than A by more than 0.5% and more than 1 cycle are flagged 'B SLOWER'. Rows whose two passes of an
arm differ by more than 1% are flagged 'noisy'. Then the validation runs' results (pass a)."""
import os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "consolidation"))
from census_report import R, rows  # noqa: E402

P = "sab"
if "--prefix" in sys.argv:
    P = sys.argv[sys.argv.index("--prefix") + 1]
ARMS = ["main", "A", "B"]
SUITES = ["4352_1e", "1d_1e", "rse", "mux"]
VALS = ["mesh_1e", "val1d_1e"]


def mean(x):
    return statistics.mean(x) if x else None


def f(x, w=7, d=1):
    return f"{x:{w}.{d}f}" if x is not None else " " * (w - 1) + "-"


def main():
    flagged = []
    print(f"{'suite':8} {'row':34} {'ntype':14} {'B':>5} | {'main a':>7} {'b':>7} {'mean':>7} | {'A a':>7} {'b':>7} "
          f"{'mean':>7} | {'B a':>7} {'b':>7} {'mean':>7} | {'B-A':>6} {'%':>6} | {'A/main':>6} {'B/main':>6}")
    for s in SUITES:
        d = {a: [rows(f"{R}/cen{P}{p}_{a}_{s}.csv") for p in "ab"] for a in ARMS}
        keys = sorted(set().union(*[set(x) for a in ARMS for x in d[a] if x]))
        for k in keys:
            v = {a: [x.get(k) if x else None for x in d[a]] for a in ARMS}
            m = {a: mean([y for y in v[a] if y is not None]) for a in ARMS}
            noisy = [a for a in ARMS if None not in v[a] and abs(v[a][0] - v[a][1]) > 0.01 * min(v[a])]
            dba = m["B"] - m["A"] if m["A"] and m["B"] else None
            pct = 100 * dba / m["A"] if dba is not None else None
            am = m["A"] / m["main"] if m["A"] and m["main"] else None
            bm = m["B"] / m["main"] if m["B"] and m["main"] else None
            flag = []
            if dba is not None and dba > 1 and pct > 0.5:
                flag.append("B SLOWER")
                flagged.append((s, k, dba, pct))
            if noisy:
                flag.append("noisy " + ",".join(noisy))
            name, ntype, size, n = k
            row = f"{name}{'#' + str(n) if n else ''}"
            print(f"{s:8} {row:34} {ntype:14} {size:>5} | {f(v['main'][0])} {f(v['main'][1])} {f(m['main'])} | "
                  f"{f(v['A'][0])} {f(v['A'][1])} {f(m['A'])} | {f(v['B'][0])} {f(v['B'][1])} {f(m['B'])} | "
                  f"{f(dba, 6)} {f(pct, 6, 2)} | {f(am, 6, 3)} {f(bm, 6, 3)}  {' '.join(flag)}")
    print("\nB slower than A by > 0.5% and > 1 cycle:" + ("" if flagged else " none"))
    for s, k, dba, pct in flagged:
        print(f"   {s} {k[0]} {k[1]} {k[2]}: +{dba:.1f} cycles ({pct:+.2f}%)")
    print("\nvalidation (pass a, one ERISC):")
    for a in ARMS:
        for s in VALS:
            l = f"{R}/cen{P}a_{a}_{s}.log"
            if not os.path.exists(l):
                print(f"   {a:5} {s:9} no log")
                continue
            t = re.sub(r"\x1b\[[0-9;]*m", "", open(l, errors="replace").read())
            res = re.findall(r"Tests: \d+ executed, \d+ failed", t)
            grp = re.findall(r"Test groups: \d+ total, \d+ ran, \d+ skipped", t)
            print(f"   {a:5} {s:9} {res[-1] if res else 'NO RESULT'}  ({grp[-1] if grp else 'no group line'})")
    print("\nleast ACTIVE_ETH config-buffer headroom per run (JIT telemetry, logged at exit):")
    for a in ARMS:
        hs = []
        for p in "ab":
            for s in VALS + SUITES:
                l = f"{R}/cen{P}{p}_{a}_{s}.log"
                if os.path.exists(l):
                    t = re.sub(r"\x1b\[[0-9;]*m", "", open(l, errors="replace").read())
                    h = re.findall(r"ACTIVE_ETH config buffer, least headroom +(\d+) B left \((\d+) of (\d+) B used\)", t)
                    if h:
                        hs.append((int(h[-1][0]), s, p, h[-1]))
        if hs:
            lo = min(hs)
            print(f"   {a:5} least {lo[0]} B left ({lo[3][1]} of {lo[3][2]} B used) in {lo[2]}/{lo[1]}; "
                  + ", ".join(f"{s}{p}:{h}" for h, s, p, _ in sorted(hs, key=lambda x: (x[1], x[2]))))
        else:
            print(f"   {a:5} not logged")


if __name__ == "__main__":
    main()
