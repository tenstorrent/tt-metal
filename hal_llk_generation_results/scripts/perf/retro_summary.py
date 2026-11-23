#!/usr/bin/env python3
"""Summary table of perf cycles for many finished runs vs the original (asserts off).

Each version is compared only with original runs measured on the SAME board in the same
session (board tag -> original labels). Uses perf_compare.table().

Usage: retro_summary.py --md out.md --json out.json   (paths below are fixed for tools/perf_cycles)
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_compare import table  # noqa: E402

P = "/proj_sw/user_dev/ctr-sbratic/tools/perf_cycles/out"
REPS = [1, 2, 3]
# version -> (out dir, original labels on the same board, version labels, board, kind)
VERSIONS = {}
for board, vers in [("b0", ["7a", "7b", "7c"]), ("b1", ["7d", "7e"])]:
    for v in vers:
        VERSIONS[v] = (f"{P}/retro", [f"{board}_orig_r{i}" for i in REPS],
                       [f"{v}_r{i}" for i in REPS], board)
VERSIONS["8a"] = (P, [f"orig_na_r{i}" for i in REPS], [f"hal8a_na_r{i}" for i in REPS], "b0 (2026-11-20)")
ORDER = ["7a", "7b", "7c", "7d", "7e", "8a"]

# Perf variants whose input/output formats and dest_acc are in test_eltwise_binary.py's main
# parameter set (formats Bfp4_b/Bfp8_b/Float16_b/Float32 all pairs, dest_acc Yes only for Float32 input).
FUNC = {"Bfp8_b", "Float16_b"}


def covered(variant):
    a, _, out, acc = variant.split(" | ")[:4]
    return a.split(".")[-1] in FUNC and out.split(".")[-1] in FUNC and acc.endswith("No")


def ratios(o, n, marker, rt, only_covered=False):
    res = []
    for (v, m, r), ov in o.items():
        if m != marker or r != rt or (only_covered and not covered(v)):
            continue
        nv = n.get((v, m, r))
        if nv:
            res.append((v, st.median(ov), st.median(nv), st.median(nv) / st.median(ov)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    rows, js = [], {}
    for v in ORDER:
        out, ol, nl, board = VERSIONS[v]
        o, n = table(out, ol), table(out, nl)
        r = {}
        for key, marker, rt, cov in [("l1", "TILE_LOOP", "L1_TO_L1", False), ("l1_cov", "TILE_LOOP", "L1_TO_L1", True),
                                     ("unp", "TILE_LOOP", "UNPACK_ISOLATE", False), ("math", "TILE_LOOP", "MATH_ISOLATE", False),
                                     ("pack", "TILE_LOOP", "PACK_ISOLATE", False), ("kern", "KERNEL", "L1_TO_L1", False)]:
            rr = ratios(o, n, marker, rt, cov)
            r[key] = dict(n=len(rr), orig=st.median(x[1] for x in rr), new=st.median(x[2] for x in rr),
                          ratio_median=st.median(x[3] for x in rr), ratio_min=min(x[3] for x in rr),
                          ratio_max=max(x[3] for x in rr), faster_variants=sum(x[3] < 1 for x in rr))
        spread = max(100 * (max(x) - min(x)) / st.median(x) for d in (o, n)
                     for (vv, m, rt), x in d.items() if m == "TILE_LOOP")
        r["rep_spread_max_pct"] = spread
        r["board"] = board
        js[v] = r
        rows.append(v)

    L = ["# Perf cycles of finished runs vs original (asserts off, Elwadd, 16 variants)", "",
         "TILE_LOOP = cycles per tile (median over 3 runs, then median over variants). Ratio = run / original "
         "(> 1 = slower), original measured on the same board in the same session. "
         "'4 func.' = the 4 perf variants whose formats/dest_acc are in test_eltwise_binary.py's main parameter set.", "",
         "| run | board | orig L1->L1 | run L1->L1 | ratio L1->L1 (min-max) | variants faster | ratio L1->L1, 4 func. | unpack | math | pack | KERNEL L1->L1 | max rep spread % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in rows:
        r = js[v]
        L.append(f"| {v} | {r['board']} | {r['l1']['orig']:.2f} | {r['l1']['new']:.2f} | {r['l1']['ratio_median']:.3f} "
                 f"({r['l1']['ratio_min']:.3f}-{r['l1']['ratio_max']:.3f}) | {r['l1']['faster_variants']}/{r['l1']['n']} | "
                 f"{r['l1_cov']['ratio_median']:.3f} | {r['unp']['ratio_median']:.3f} | {r['math']['ratio_median']:.3f} | "
                 f"{r['pack']['ratio_median']:.3f} | {r['kern']['ratio_median']:.3f} | {r['rep_spread_max_pct']:.1f} |")
    groups = {"HAL 7a-7e": ["7a", "7b", "7c", "7d", "7e"]}
    L += ["", "| group | median of per-run L1->L1 ratios | min | max |", "|---|---|---|---|"]
    for g, vs in groups.items():
        x = [js[v]["l1"]["ratio_median"] for v in vs]
        L.append(f"| {g} | {st.median(x):.3f} | {min(x):.3f} | {max(x):.3f} |")
        js[g] = dict(l1_ratio_median=st.median(x), l1_ratio_min=min(x), l1_ratio_max=max(x))
    text = "\n".join(L) + "\n"
    print(text)
    open(a.md, "w").write(text)
    json.dump(js, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
