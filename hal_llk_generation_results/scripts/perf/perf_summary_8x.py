#!/usr/bin/env python3
"""Perf-cycle summary entries for runs 8b-8e in the same format as retro_summary.py (summary.json),
computed from LOCAL copies of tools/perf_cycles/out/retro/{p<run>_orig_r1..3, <run>_r1..3}
(driver_8x.sh step 3: original and run alternating on the run's own board, asserts off).
Uses retro_summary.ratios() / covered() and perf_compare.table() unchanged.
Usage: perf_summary_8x.py <retro dir> <out.json> [run ...]   (runs without perf data are skipped)"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_compare import table  # noqa: E402
from retro_summary import ratios  # noqa: E402

BOARD = {"8b": "b2", "8c": "b1", "8d": "b3", "8e": "b0"}   # driver_8x boards (memory run8cde / run8b)
REPS = [1, 2, 3]


def main():
    retro, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or ["8b", "8c", "8d", "8e"]
    js = {}
    for v in names:
        ol, nl = [f"p{v}_orig_r{i}" for i in REPS], [f"{v}_r{i}" for i in REPS]
        if not all(os.path.isdir(os.path.join(retro, x, "perf_data")) for x in ol + nl):
            print(f"{v}: perf data missing, skipped")
            continue
        o, n = table(retro, ol), table(retro, nl)
        r = {}
        for key, marker, rt, cov in [("l1", "TILE_LOOP", "L1_TO_L1", False), ("l1_cov", "TILE_LOOP", "L1_TO_L1", True),
                                     ("unp", "TILE_LOOP", "UNPACK_ISOLATE", False), ("math", "TILE_LOOP", "MATH_ISOLATE", False),
                                     ("pack", "TILE_LOOP", "PACK_ISOLATE", False), ("kern", "KERNEL", "L1_TO_L1", False)]:
            rr = ratios(o, n, marker, rt, cov)
            r[key] = dict(n=len(rr), orig=st.median(x[1] for x in rr), new=st.median(x[2] for x in rr),
                          ratio_median=st.median(x[3] for x in rr), ratio_min=min(x[3] for x in rr),
                          ratio_max=max(x[3] for x in rr), faster_variants=sum(x[3] < 1 for x in rr))
        r["rep_spread_max_pct"] = max(100 * (max(x) - min(x)) / st.median(x) for d in (o, n)
                                      for (vv, m, rt), x in d.items() if m == "TILE_LOOP")
        r["board"] = BOARD.get(v, "?")
        js[v] = r
        print(f"{v}: L1->L1 orig {r['l1']['orig']:.2f} run {r['l1']['new']:.2f} ratio {r['l1']['ratio_median']:.4f} "
              f"({r['l1']['ratio_min']:.3f}-{r['l1']['ratio_max']:.3f}), faster {r['l1']['faster_variants']}/{r['l1']['n']}")
    json.dump(js, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
