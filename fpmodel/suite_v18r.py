"""Suite result for v18r: picks unchanged from v17 keep the v17 session's times; changed picks that landed on a config the
v17 session timed (legacy or rules) use that; the rest come from the v18r session, compared with that session's own
legacy and rules. Also prediction accuracy of frozen v18r on every timed suite config.
usage: suite_v18r.py  (run from a v18r fpmodel copy)"""
import sys, json

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R

W = "/localdev/rmiller/mm-oob-model-work"
norm = lambda s: str(s).replace(" ", "")
old = pd.read_csv(f"{W}/device/suite719_v17_timed.csv", low_memory=False)
new = pd.read_csv(f"{W}/packov/suite_packov_timed.csv", low_memory=False)
picks = pd.read_csv(f"{W}/packov/suite_v18r_picks_all.csv", low_memory=False).drop_duplicates("case").set_index("case")
old, new = (
    x[x.status == "ok"].drop_duplicates(["case", "origin"]).assign(cfg=lambda z: z.config.map(norm)) for x in (old, new)
)
ot = old.pivot(index="case", columns="origin", values="device_ns")
oc = old.pivot(index="case", columns="origin", values="cfg")
nt = new.pivot(index="case", columns="origin", values="device_ns")
rows = []
for c in ot.index:
    if c not in picks.index:
        continue
    pc = norm(picks.config[c])
    if pc == oc.loc[c, "model"]:
        rows.append(
            (c, ot.loc[c, "model"], ot.loc[c, "legacy"], ot.loc[c, "heuristic"], ot.loc[c, "model"], "same as v17")
        )
    elif pc in (oc.loc[c, "legacy"], oc.loc[c, "heuristic"]):
        t = ot.loc[c, "legacy"] if pc == oc.loc[c, "legacy"] else ot.loc[c, "heuristic"]
        rows.append((c, t, ot.loc[c, "legacy"], ot.loc[c, "heuristic"], ot.loc[c, "model"], "on a timed config"))
    elif c in nt.index and "model" in nt and not np.isnan(nt.loc[c, "model"]):
        # v18r session: scale v17's time by that session's legacy so v17 vs v18r stays comparable
        r = ot.loc[c, "model"] / ot.loc[c, "legacy"] * nt.loc[c, "legacy"]
        rows.append((c, nt.loc[c, "model"], nt.loc[c, "legacy"], nt.loc[c, "heuristic"], r, "new session"))
r = pd.DataFrame(rows, columns=["case", "v18r", "legacy", "rules", "v17", "src"]).set_index("case").dropna()
gm = lambda v: float(np.exp(np.mean(np.log(v))))
out = {"n": int(len(r)), "by_src": r.src.value_counts().to_dict()}
for k in ("v18r", "v17", "rules"):
    q = r[k] / r.legacy
    out[k] = dict(
        vs_legacy=round(gm(q), 3),
        regr=int((q > 1.05).sum()),
        severe=int((q > 1.25).sum()),
        worst=round(float(q.max()), 3),
    )
out["v18r_vs_rules"] = round(gm(r.v18r / r.rules), 3)
out["v18r_vs_v17"] = round(gm(r.v18r / r.v17), 3)
out["v18r_regressions"] = (r.v18r / r.legacy)[r.v18r / r.legacy > 1.05].round(3).sort_values().to_dict()
# prediction accuracy on every timed suite config, frozen v18r (WH)
p = json.load(open(f"{W}/fresh/frozen_v18r_wh.json"))
bst = lgb.Booster(model_file=f"{W}/fresh/frozen_v18r_wh.lgb")
x = pd.concat([old, new], ignore_index=True).drop_duplicates(["case", "cfg"], keep="last").reset_index(drop=True)
x["arch_"] = "wh"
x = M.annotate(x)
b, g, pr = R.base_parts(x, p)
for name, pred in (("v18r", b * np.exp(bst.predict(R.features(x, g, pr)))), ("base", b)):
    e = np.abs(np.log(pred / x.device_ns.to_numpy()))
    out[f"acc_{name}"] = dict(
        n=int(len(x)),
        median_abs_err=round(float(np.median(e)), 3),
        within10=round(float((e < np.log(1.10)).mean()), 3),
        within25=round(float((e < np.log(1.25)).mean()), 3),
    )
print(json.dumps(out, indent=1))
json.dump(out, open(f"{W}/packov/suite_v18r_result.json", "w"), indent=1)
