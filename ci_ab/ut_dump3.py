#!/usr/bin/env python3
"""Compact job-log lines from tt-llk perf CSVs (TILE_LOOP rows). usage: ut_dump3.py <tag> <side> <run dir>
T|tag|side|module|key|mean(L1_TO_L1)|mean(PACK_ISOLATE)|mean(UNPACK_ISOLATE)"""
import glob, sys
import pandas as pd

SKIP = ("marker", "run_index")


def key_cols(d):
    return [c for c in d.columns if c not in SKIP and not c.startswith(("mean(", "TEXT_SIZE", "L1_TO_L1_", "PACK_ISOLATE_",
                                                                         "UNPACK_ISOLATE_", "MATH_ISOLATE_", "L1_CONGESTION_"))]


tag, side, rdir = sys.argv[1:4]
for f in sorted(glob.glob(rdir + "/**/perf_*.csv", recursive=True)):
    if f.endswith(".counters.csv") or "post" in f:
        continue
    mod = f.split("/")[-1].split(".")[0]
    d = pd.read_csv(f, low_memory=False)
    if "marker" in d:
        d = d[d.marker == "TILE_LOOP"]
    kc = key_cols(d)
    for _, r in d.iterrows():
        k = ";".join(str(r[c]) for c in kc)
        v = [r.get(f"mean({t})", float("nan")) for t in ("L1_TO_L1", "PACK_ISOLATE", "UNPACK_ISOLATE")]
        print(f"T|{tag}|{side}|{mod}|{k}|{v[0]:.3f}|{v[1]:.3f}|{v[2]:.3f}")
    if len(d):
        print(f"KH|{mod}|" + ";".join(kc))
