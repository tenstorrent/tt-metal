"""Score picks against legacy and the rules, each compared within the session (run) where the pick was timed.
usage: eval_picks.py PICKS_CSV[:label] [PICKS_CSV[:label] ...] --timed T1.csv T2.csv ...
A pick is matched to a timed row of any origin with the same config (spaces ignored); its legacy and rules times come
from the same run. Picks never timed are reported as missing. Prints one line per picks file and writes per-case ratios
next to the first picks file (<picks>.eval.csv)."""
import sys
import numpy as np, pandas as pd
from scipy.stats import beta

args = sys.argv[1:]
i = args.index("--timed")
picks_args, timed = args[:i], args[i + 1 :]
norm = lambda s: str(s).replace(" ", "")
t = pd.concat([pd.read_csv(f, low_memory=False).assign(src=f) for f in timed], ignore_index=True)
t = t[t.status == "ok"].assign(cfg=lambda z: z.config.map(norm))
t["sess"] = t.src + "|" + t.run.astype(str)
leg = t[t.origin == "legacy"].drop_duplicates(["case", "sess"], keep="last").set_index(["case", "sess"]).device_ns
rul = t[t.origin == "heuristic"].drop_duplicates(["case", "sess"], keep="last").set_index(["case", "sess"]).device_ns
byc = {k: g for k, g in t.groupby(["case", "cfg"])}
gm = lambda v: float(np.exp(np.mean(np.log(v))))
cp = lambda k, n: beta.ppf(0.975, k + 1, n - k) if n else np.nan
out = []
for pa in picks_args:
    path, label = (pa.split(":", 1) + [pa])[:2] if ":" in pa else (pa, pa.split("/")[-1])
    p = pd.read_csv(path).drop_duplicates("case")
    rows, missing = [], 0
    for c, cfg in zip(p.case, p.config.map(norm)):
        g = byc.get((c, cfg))
        g = None if g is None else g[[(c, s) in leg.index and (c, s) in rul.index for s in g.sess]]
        if g is None or not len(g):
            missing += 1
            continue
        r = g.iloc[-1]
        rows.append((c, r.device_ns / leg[(c, r.sess)], rul[(c, r.sess)] / leg[(c, r.sess)]))
    e = pd.DataFrame(rows, columns=["case", "pick_vs_legacy", "rules_vs_legacy"]).set_index("case")
    m, q = e.pick_vs_legacy, e.rules_vs_legacy
    k = int((m > 1.05).sum())
    print(
        f"{label:28s} n={len(e):4d} missing={missing:3d} | pick {gm(m):.3f} >5% {k:3d} ({100*k/len(e):.1f}%, up95 {100*cp(k,len(e)):.1f}%)"
        f" >25% {int((m>1.25).sum()):3d} worst {m.max():.2f} | rules {gm(q):.3f} >5% {int((q>1.05).sum()):3d} >25% {int((q>1.25).sum()):3d}"
        f" | pick/rules {gm(m/q):.3f}"
    )
    out.append(e.add_prefix(label + ":"))
pd.concat(out, axis=1).to_csv(picks_args[0].split(":")[0] + ".eval.csv")
