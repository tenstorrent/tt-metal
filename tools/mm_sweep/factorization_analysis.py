#!/usr/bin/env python3
# Part 4: same-core factorization analysis. Reads the exhaustive per-shape sweep JSONs (regime_a_sweep_*)
# and, at each fixed worker count, compares how the Pk*Ns*Sm product should be spent:
#   pure-K  : Ns=Sm=1 (max reduction depth)
#   KxM     : Sm>1, Ns=1 (trade reduction depth for M-parallelism)
#   KxN     : Ns>1, Sm=1 (trade reduction depth for N-parallelism)
#   mixed   : Ns>1 AND Sm>1
# Each cell keeps that factorization's BEST feasible (kb,nsb) by median kernel us. Answers whether
# reduction depth should be exchanged for M or N parallelism at Mt=8.
import json, os, sys

HERE = os.path.dirname(__file__)
FREQ = 1.35e9


def factor_class(cfg):
    Ns, Pk, Sm, kb, nsb = cfg
    if Ns == 1 and Sm == 1:
        return "pure-K"
    if Sm > 1 and Ns == 1:
        return "KxM"
    if Ns > 1 and Sm == 1:
        return "KxN"
    return "mixed"


def analyze(path):
    d = json.load(open(path))
    M, K, N = d["M"], d["K"], d["N"]
    res = [r for r in d["results"] if r.get("cls") == "ok"]
    # group by cores, then by factor class -> best (min median)
    by_cores = {}
    for r in res:
        c = r["cores"]
        by_cores.setdefault(c, {}).setdefault(factor_class(r["cfg"]), []).append(r)
    lines = [f"\n### {M}x{K}x{N} (Mt={ (M+31)//32 }) — {len(res)} ok configs"]
    lines.append("| cores | Pk*Ns*Sm | pure-K | KxM | KxN | mixed |")
    lines.append("|" + "---|" * 6)
    for cores in sorted(by_cores):
        prod = cores // 8
        cells = {}
        for fc in ("pure-K", "KxM", "KxN", "mixed"):
            grp = by_cores[cores].get(fc, [])
            if grp:
                b = min(grp, key=lambda r: r["us_med"])
                cells[fc] = f"{b['us_med']:.1f}us {tuple(b['cfg'])}"
            else:
                cells[fc] = "-"
        lines.append(f"| {cores} | {prod} | {cells['pure-K']} | {cells['KxM']} | {cells['KxN']} | {cells['mixed']} |")
    # overall best + its class
    best = min(res, key=lambda r: r["us_med"])
    lines.append(
        f"\n**Best overall:** {tuple(best['cfg'])} = {best['us_med']:.1f}us "
        f"({best['pct512']:.1f}% of 512) cores={best['cores']} class={factor_class(best['cfg'])}"
    )
    return "\n".join(lines)


def main():
    shapes = sys.argv[1:] or ["256x2048x1024", "256x6144x768", "256x6144x2304", "256x6144x4608"]
    out = [
        "# Part 4: same-core factorization (reduction depth vs M/N parallelism)\n",
        "At each fixed worker count, the best median-us config of each factorization class (best kb/nsb "
        "kept per cell). pure-K = max split-K depth; KxM/KxN trade depth for M/N parallelism.\n",
    ]
    for s in shapes:
        p = f"{HERE}/regime_a_sweep_{s}.json"
        if os.path.exists(p):
            out.append(analyze(p))
        else:
            out.append(f"\n### {s} — (no sweep file {os.path.basename(p)})")
    md = f"{HERE}/FACTORIZATION_ANALYSIS.md"
    open(md, "w").write("\n".join(out) + "\n")
    print("WROTE", md)
    print("\n".join(out))


if __name__ == "__main__":
    main()
