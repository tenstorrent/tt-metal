"""Best config per (proj, M) vs the model config, by family, from a mm_sweep jsonl."""
import json, sys, re
from collections import defaultdict
rs = [json.loads(l) for l in open(sys.argv[1])]
fam = lambda c: c if c in ("model", "default", "auto_v2") else re.split(r"_", c)[0]
g = defaultdict(list)
for r in rs:
    g[(r["proj"], r["m"])].append(r)
for (p, m), v in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    ok = [r for r in v if r["ok"] and r["pcc_model"] is not None and r["pcc_model"] > 0.9999]
    bad_pcc = [r for r in v if r["ok"] and r["pcc_model"] is not None and r["pcc_model"] <= 0.9999]
    model = next((r for r in v if r["cfg"] == "model" and r["ok"]), None)
    if not model:
        continue
    print(f"== {p} M={m} K={model['K']} N={model['N']} {model['fidelity']}: model {model['ns']/1e3:.1f}us cores={model['cores']} "
          f"util_chip={model['util_chip']:.2f} {model['gbps']:.0f}GB/s  (n={len(v)} ok={len(ok)} err={sum(not r['ok'] for r in v)} badpcc={len(bad_pcc)})")
    best = {}
    for r in ok:
        f = fam(r["cfg"])
        if f not in best or r["ns"] < best[f]["ns"]:
            best[f] = r
    for f, r in sorted(best.items(), key=lambda kv: kv[1]["ns"]):
        tot = r["ns"] + (r.get("reshard_ns") or 0)
        print(f"   {f:8s} {r['cfg']:34s} {r['ns']/1e3:7.1f}us ({100*(r['ns']/model['ns']-1):+5.1f}%) cores={r['cores']:3d} "
              f"util_chip={r['util_chip']:.2f} {r['gbps']:4.0f}GB/s"
              + (f"  +reshard {r['reshard_ns']/1e3:.1f}us => {100*(tot/model['ns']-1):+5.1f}%" if r.get("reshard_ns") else ""))
