#!/usr/bin/env python3
"""placeagg.py BUILD_A BUILD_B [--fn=run<] [--cases=a,b]: placement.py's per-local kinds (frame / split / -) for every
image two coverage builds share, aggregated per (case, local, kind A -> kind B) with counts (placement.py prints only
the first two images per case)."""
import collections, concurrent.futures, glob, os, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import placement as pl
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = opts.get("fn", "run<")
cases = set(opts["cases"].split(",")) if "cases" in opts else None
a, b = args
jobs = []
for e in sorted(glob.glob(f"{pl.COV}/{a}/*/*/*.elf")):
    rel = os.path.relpath(e, f"{pl.COV}/{a}")
    case = rel.split("__")[0]
    if (cases and case not in cases) or not os.path.exists(f"{pl.COV}/{b}/{rel}"):
        continue
    jobs.append((case, e, f"{pl.COV}/{b}/{rel}"))
def one(j):
    case, ea, eb = j
    return case, pl.kinds(ea, fn), pl.kinds(eb, fn)
agg = collections.Counter()
kinds_a = collections.Counter()
with concurrent.futures.ThreadPoolExecutor(32) as ex:
    for case, ka, kb in ex.map(one, jobs):
        for v in set(ka) | set(kb):
            kinds_a[(v, ka.get(v))] += 1
            if ka.get(v) != kb.get(v):
                agg[(case, v, ka.get(v), kb.get(v))] += 1
print("kinds in", a, ":", dict(sorted(kinds_a.items(), key=str)))
for (case, v, x, y), n in sorted(agg.items()):
    print(f"  {case:28s} {v:14s} {x} -> {y}  x{n}")
print("images compared:", len(jobs), "with any difference listed above")
