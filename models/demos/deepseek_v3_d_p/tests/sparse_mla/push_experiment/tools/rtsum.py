"""rtsum.py <tag> [<tag2>]: per program (runtime id) device time, max across chips, median per context band (us).
import os
Programs run once per chunk; the last 205 occurrences are the measured request's chunks 0..204 (5120 tokens each)."""
import json, sys, statistics as st
from collections import defaultdict
S = os.environ.get("PUSH_EXP_DIR", "/tmp/push_exp") + "/runs"
CH = 5120
BANDS = [(0, 50e3), (50e3, 150e3), (150e3, 300e3), (300e3, 500e3), (500e3, 700e3), (700e3, 900e3), (900e3, 1.1e6)]
def short(n):
    k = n.split(";")[0].split("operations/")[-1]
    for a, b in (("experimental/", ""), ("device/kernels/", ""), ("dataflow/", ""), ("fabric/", "")):
        k = k.replace(a, b)
    return k[:70]
def load(tag, nchunks=205):
    d = json.load(open(f"{S}/{tag}/rt/rank0.json"))
    f = d["meta"]["freq"]
    per = defaultdict(lambda: defaultdict(list))
    for chip, rid, s, e in d["recs"]:
        per[rid][chip].append((s, (e - s) / f / 1e3))
    out = {}
    for rid, chips in per.items():
        seqs = [sorted(v) for v in chips.values()]
        n = min(len(x) for x in seqs)
        if n < nchunks:
            continue
        occ = [max(x[len(x) - n + i][1] for x in seqs) for i in range(n)][-nchunks:]
        first = min(x[len(x) - nchunks][0] for x in seqs)
        out[rid] = (short(d["names"][str(rid)]), occ, first)
    return out
tags = sys.argv[1:]
for tag in tags:
    o = load(tag)
    print(f"== {tag}: programs in chunk order (first-chunk start), us max-across-chips, median per band")
    print(" " * 74 + "".join(f"{int(a/1e3):>5}k" for a, b in BANDS))
    for rid, (nm, occ, first) in sorted(o.items(), key=lambda kv: kv[1][2]):
        meds = []
        for a, b in BANDS:
            v = [occ[i] for i in range(len(occ)) if a <= i * CH < b]
            meds.append(st.median(v) if v else 0)
        if max(meds) < 15:
            continue
        print(f"  {nm:70s}" + "".join(f"{m:6.0f}" for m in meds))
