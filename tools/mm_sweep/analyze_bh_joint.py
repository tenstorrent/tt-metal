#!/usr/bin/env python3
"""
Analyze the BH joint (S,Pk,blocking) sweep output (joint_sweep.py's bh_joint.json dict-format).
Safe to run while the sweep is still going (reads whatever shapes are present). Answers:
  Q1 - oracle vs current heuristic: best_vs_auto distribution; where AUTO BEATS the oracle
       (best_vs_auto<1 => the pruned block set missed AUTO's auto-block choice -> generator gap).
  Q2 - heuristic_SPk vs oracle (best.S,best.Pk) divergence -> input for re-fitting BH thresholds.
  Q3 - structurally low oracle util (even the best config is bad) -> flag for kernel work.
  Q4 - how many oracle 'best' configs rely on the fused-K cache-bug path (cache_pcc<0.99).

  python tools/mm_sweep/analyze_bh_joint.py [bh_joint.json] [--md OUT.md]
"""
import json, sys, math, statistics
from collections import Counter

PATH = next((a for a in sys.argv[1:] if not a.startswith("--")), "tools/mm_sweep/bh_joint.json")
MD = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
LOW_UTIL = 40.0  # oracle-best util below this is flagged structural


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main():
    d = json.load(open(PATH))
    rows, lines = [], []
    for key, r in d.items():
        b, a = r.get("best"), r.get("auto")
        Mt, Nt, Kt = r["MtNtKt"]
        small, big = min(Mt, Nt), max(Mt, Nt)
        rows.append(
            {
                "shape": key,
                "out": Mt * Nt,
                "Kt": Kt,
                "skew": big / small if small else 1.0,
                "grid": r["grid"],
                "cores": r["grid"][0] * r["grid"][1],
                "heur": tuple(r["heuristic_SPk"]),
                "bSPk": (b["S"], b["Pk"]) if b else None,
                "butil": b["util"] if b else None,
                "autil": a["util"] if a else None,
                "bva": r.get("best_vs_auto"),
                "bcache": (b.get("cache_pcc", 1.0) if b else 1.0),
                "ncfg": r.get("n_configs"),
                "ncache": r.get("n_cache_bug", 0),
            }
        )

    rows.sort(key=lambda x: (x["butil"] is None, x["butil"] or 0))
    lines.append(f"# BH joint sweep analysis — {len(rows)} shapes  ({PATH})\n")

    # Q1/Q4 per-shape table
    lines.append("## Per-shape (sorted by oracle util)\n")
    lines.append("| shape | out·Kt skew | heurSPk | bestSPk | best% | auto% | best/auto | flags |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for x in rows:
        flag = []
        if x["butil"] is not None and x["butil"] < LOW_UTIL:
            flag.append("LOWUTIL")
        if x["bva"] is not None and x["bva"] < 0.97:
            flag.append("AUTO>ORACLE")  # pruned-block gap
        if x["bcache"] < 0.99:
            flag.append("cachebug*")
        if x["heur"] != x["bSPk"] and x["bSPk"] is not None:
            flag.append("SPk≠")
        bu = ("%.1f" % x["butil"]) if x["butil"] is not None else "—"
        au = ("%.1f" % x["autil"]) if x["autil"] is not None else "—"
        bv = ("%.2fx" % x["bva"]) if x["bva"] is not None else "—"
        lines.append(
            f"| {x['shape']} | {x['out']}·{x['Kt']} s{x['skew']:.0f} | {x['heur']} | "
            f"{x['bSPk']} | {bu} | {au} | {bv} | {' '.join(flag)} |"
        )

    # Aggregates
    butils = [x["butil"] for x in rows if x["butil"] is not None]
    bvas = [x["bva"] for x in rows if x["bva"] is not None]
    auto_wins = [x for x in rows if x["bva"] is not None and x["bva"] < 0.97]
    spk_match = [x for x in rows if x["bSPk"] is not None and x["heur"] == x["bSPk"]]
    spk_div = [x for x in rows if x["bSPk"] is not None and x["heur"] != x["bSPk"]]
    low = [x for x in rows if x["butil"] is not None and x["butil"] < LOW_UTIL]
    cache_best = [x for x in rows if x["bcache"] < 0.99]

    lines.append("\n## Aggregates")
    lines.append(
        f"- oracle best util: geomean **{geomean(butils):.1f}%**, "
        f"median {statistics.median(butils):.1f}%, min {min(butils):.1f}%, max {max(butils):.1f}%"
    )
    lines.append(f"- best_vs_auto: geomean **{geomean(bvas):.2f}x**, median {statistics.median(bvas):.2f}x")
    lines.append(f"- AUTO beats oracle (pruned-block gap, <0.97x): **{len(auto_wins)}** shapes")
    lines.append(
        f"- heuristic (S,Pk) == oracle: **{len(spk_match)}/{len(spk_match)+len(spk_div)}**; "
        f"diverges on {len(spk_div)}"
    )
    lines.append(f"- structurally LOW util (<{LOW_UTIL:.0f}% even at oracle): **{len(low)}** shapes")
    lines.append(
        f"- oracle 'best' on cache-bug path (cache_pcc<0.99): **{len(cache_best)}** shapes "
        f"(timing valid; correctness from fresh run)"
    )

    # Q2 detail: where heuristic diverges, what it should have picked
    if spk_div:
        lines.append("\n## (S,Pk) heuristic divergence (re-fit input)")
        lines.append("| shape | out·Kt skew | heur | oracle | best% |")
        lines.append("|---|---|---|---|---|")
        for x in sorted(spk_div, key=lambda y: -(y["butil"] or 0)):
            lines.append(
                f"| {x['shape']} | {x['out']}·{x['Kt']} s{x['skew']:.0f} | {x['heur']} | "
                f"{x['bSPk']} | {('%.1f' % x['butil']) if x['butil'] is not None else '—'} |"
            )

    out = "\n".join(lines)
    print(out)
    if MD:
        open(MD, "w").write(out)
        print(f"\n[written {MD}]")


if __name__ == "__main__":
    main()
