"""python3 -I fusedec_budget_scale.py <ops.csv of fusedec_budget.py (layers 2, 3, 20 + fixed parts)> [--wall 51.14] [--engram-ms 0.61] [--mult typ=32,owner=3,idx=5]
Scales the 3-layer profile to the 40-layer round: typical MoE layer (L3: 15 CSA2 reuse + 15 ratio-1 readers + 2 SWA) x typ, ratio-2 owner + indexer (L2: layers 2, 8, 14) x owner, ratio-1 index layer
(L20: layers 20, 24, 28, 32, 36) x idx; fixed parts (rope tables, embedding, head, accept, drafter write_main + draft) once; commit: the attention-glue part (compressor commit of the owners) x owner, the rest once.
Prints measured / ideal / above-ideal ms per category and for the round, the device-timeline gap, and the reconciliation with the 40-layer unprofiled wall."""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
wall = float(arg("--wall", "51.14"))
eng = float(arg("--engram-ms", "0.61"))
mult = dict(typ=32, owner=3, idx=5)
for kv in arg("--mult", "").split(","):
    if kv:
        k, v = kv.split("=")
        mult[k] = int(v)
rows = list(csv.DictReader(open(path)))
for r in rows:
    for k in ("mean", "slow", "fast", "ideal", "wait", "gap", "dram", "fab", "flops"):
        r[k] = float(r[k])
    r["layer"] = int(r["layer"]) if r["layer"] not in ("", "None") else None
L = {3: mult["typ"], 2: mult["owner"], 20: mult["idx"]}
agg = defaultdict(lambda: dict(mean=0.0, ideal=0.0, slow=0.0, gap=0.0, n=0, wait=0.0))
grp = defaultdict(lambda: dict(mean=0.0, ideal=0.0, n=0, cls=defaultdict(float)))
cause = defaultdict(lambda: dict(mean=0.0, ideal=0.0, n=0))
lay = defaultdict(lambda: dict(mean=0.0, ideal=0.0, gap=0.0, n=0))
fixed = defaultdict(lambda: dict(mean=0.0, ideal=0.0, gap=0.0, n=0))
for r in rows:
    if r["layer"] is not None:
        m = L.get(r["layer"], 0)
        lay[r["layer"]]["mean"] += r["mean"]
        lay[r["layer"]]["ideal"] += r["ideal"]
        lay[r["layer"]]["gap"] += r["gap"]
        lay[r["layer"]]["n"] += 1
    else:
        m = 1
        if r["phase"] == "commit" and r["cat"] == "attention glue":
            m = mult["owner"]
        f = fixed[r["phase"]]
        f["mean"] += r["mean"] * m
        f["ideal"] += r["ideal"] * m
        f["gap"] += r["gap"] * m
        f["n"] += m
    gx = grp[(r["cat"], r["name"][:80])]
    gx["mean"] += r["mean"] * m
    gx["ideal"] += r["ideal"] * m
    gx["n"] += m
    gx["cls"][r["cls"]] += (r["mean"] - r["ideal"]) * m
    cz = cause[r["cls"]]
    cz["mean"] += r["mean"] * m
    cz["ideal"] += r["ideal"] * m
    cz["n"] += m
    x = agg[r["cat"]]
    x["mean"] += r["mean"] * m
    x["ideal"] += r["ideal"] * m
    x["slow"] += r["slow"] * m
    x["gap"] += r["gap"] * m
    x["n"] += m
    x["wait"] += r["wait"] * m
tot = sum(x["mean"] for x in agg.values())
idl = sum(x["ideal"] for x in agg.values())
gap = sum(x["gap"] for x in agg.values())
nops = sum(x["n"] for x in agg.values())
print("## Layer types measured in this profile (per layer, us)\n")
print(
    "| layer | ops | kernel us (mean chip) | ideal us | above ideal us | device gap us | typ / owner / idx multiplicity |"
)
print("|---|---|---|---|---|---|---|")
for l, x in sorted(lay.items()):
    print(
        f"| L{l} | {x['n']} | {x['mean']*1e6:.0f} | {x['ideal']*1e6:.0f} | {(x['mean']-x['ideal'])*1e6:.0f} | {x['gap']*1e6:.0f} | x{L.get(l, 0)} |"
    )
print("\n## Scaled 40-layer round\n")
print(
    f"kernel sum {tot*1e3:.2f} ms (+ Engram {eng:.2f} ms from the 8-layer profile s1) ; ideal sum {idl*1e3:.2f} ms ; time above ideal {(tot-idl)*1e3:.2f} ms ; device-timeline gap between ops {gap*1e3:.2f} ms over {nops} ops ({gap/nops*1e6:.2f} us per op)"
)
print(
    f"kernel + Engram + gap = {(tot+gap)*1e3+eng:.2f} ms ; unprofiled 40-layer wall (w40_k3, natural routing of random token ids) {wall:.2f} ms ; difference {wall-((tot+gap)*1e3+eng):+.2f} ms ({100*(wall-((tot+gap)*1e3+eng))/wall:+.1f} % of the wall)"
)
print(
    "\n| category | ops | measured ms | ideal ms | above ideal ms | meas/ideal | slowest-chip sum ms | collective wait ms |"
)
print("|---|---|---|---|---|---|---|---|")
for c, x in sorted(agg.items(), key=lambda kv: -(kv[1]["mean"] - kv[1]["ideal"])):
    print(
        f"| {c} | {x['n']} | {x['mean']*1e3:.3f} | {x['ideal']*1e3:.3f} | {(x['mean']-x['ideal'])*1e3:.3f} | {x['mean']/max(x['ideal'],1e-9):.1f} | {x['slow']*1e3:.3f} | {x['wait']*1e3:.3f} |"
    )
print("\n| fixed phase (once) | ops | measured ms | ideal ms | device gap ms |")
print("|---|---|---|---|---|")
for p, x in sorted(fixed.items(), key=lambda kv: -kv[1]["mean"]):
    print(f"| {p} | {x['n']} | {x['mean']*1e3:.3f} | {x['ideal']*1e3:.3f} | {x['gap']*1e3:.3f} |")
print("\n## Ranked: time above ideal per op group, scaled to the 40-layer round\n")
print("| rank | op group | ops | measured ms | ideal ms | above ideal ms | meas/ideal | main cause |")
print("|---|---|---|---|---|---|---|---|")
for i, (k, x) in enumerate(sorted(grp.items(), key=lambda kv: -(kv[1]["mean"] - kv[1]["ideal"]))[:40]):
    cz = max(x["cls"].items(), key=lambda kv: kv[1])[0]
    print(
        f"| {i+1} | {k[0]}: {k[1]} | {x['n']} | {x['mean']*1e3:.3f} | {x['ideal']*1e3:.3f} | {(x['mean']-x['ideal'])*1e3:.3f} | {x['mean']/max(x['ideal'],1e-9):.1f} | {cz} |"
    )
print("\n## By cause class, scaled\n")
print("| cause | ops | measured ms | ideal ms | above ideal ms |")
print("|---|---|---|---|---|")
for k, x in sorted(cause.items(), key=lambda kv: -(kv[1]["mean"] - kv[1]["ideal"])):
    print(f"| {k} | {x['n']} | {x['mean']*1e3:.3f} | {x['ideal']*1e3:.3f} | {(x['mean']-x['ideal'])*1e3:.3f} |")
