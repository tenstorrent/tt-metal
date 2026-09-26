#!/usr/bin/env python3
"""Summarize results_sp2/sim_layouts/*.txt: best split per (scenario, pass, W, policy) by useful tok/s.

  sim_summary.py results_sp2/sim_layouts > results_sp2/sim_layouts/summary.txt
"""
import glob, os, re, sys

LAYOUTS = {
    "L-a": "8-stage SP=4 pipeline over 4 galaxies",
    "L-b": "4 x independent 4-stage SP=2 galaxies",
    "L-c": "16-stage SP=2 pipeline over 4 galaxies",
    "L-d": "4 x independent 2-stage SP=4 galaxies",
}


def parse(path):
    head, split, rows = {}, None, []
    for line in open(path):
        if line.startswith("stages="):
            head = dict(re.findall(r"(\w+)=([\w.,%()]+)", line))
        elif line.startswith("split="):
            split = line.split("=", 1)[1].strip()
        else:
            p = line.split()
            if len(p) > 8 and p[0].isdigit() and p[1] in ("fcfs", "cost", "bucket"):
                rows.append(
                    dict(
                        split=split,
                        W=int(p[0]),
                        pol=p[1],
                        tok=float(p[2]),
                        p50=float(p[3]),
                        p99=float(p[4]),
                        fill=float(p[5]),
                        hot50=float(p[6]),
                        hot99=float(p[7]),
                        util=" ".join(p[8:]),
                    )
                )
    return head, rows


def main(d):
    print(
        f"{'scen':5} {'pass':8} {'W':>5} {'policy':6} {'tok/s':>7} {'fwd p50':>8} {'fwd p99':>8} {'hot p50':>8} "
        f"{'hot p99':>8} {'<=1.5s':>6}  best split  (stage utilisation)"
    )
    best_overall = {}
    for f in sorted(glob.glob(os.path.join(d, "L-*.txt"))):
        scen, tag = os.path.basename(f)[:-4].split("_", 1)
        head, rows = parse(f)
        for W in sorted({r["W"] for r in rows}):
            for pol in ("fcfs", "cost"):
                cand = [r for r in rows if r["W"] == W and r["pol"] == pol]
                if not cand:
                    continue
                b = max(cand, key=lambda r: r["tok"])
                ok = "yes" if b["hot99"] <= 1500 else "no"
                print(
                    f"{scen:5} {tag:8} {W:5d} {pol:6} {b['tok']:7.0f} {b['p50']:8.1f} {b['p99']:8.1f} {b['hot50']:8.0f} "
                    f"{b['hot99']:8.0f} {ok:>6}  {b['split']}  ({b['util']})"
                )
                key = (scen, tag)
                if b["hot99"] <= 1500 and (key not in best_overall or b["tok"] > best_overall[key]["tok"]):
                    best_overall[key] = dict(b, W=W, pol=pol)
    print("\nbest configuration meeting hot p99 <= 1.5 s, per scenario and hop pass:")
    for (scen, tag), b in sorted(best_overall.items()):
        print(
            f"  {scen} ({LAYOUTS.get(scen, '')}), {tag}: {b['tok']:.0f} tok/s  W={b['W']} {b['pol']}  "
            f"hot p50/p99 {b['hot50']:.0f}/{b['hot99']:.0f} ms  split {b['split']}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
