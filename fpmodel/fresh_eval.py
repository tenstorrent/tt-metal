"""Fresh-set acceptance: model vs rules vs legacy (same session), overall and per stratum, with the regression distribution;
plus regret on the fully swept subset. usage: fresh_eval.py DIR"""
import sys, numpy as np, pandas as pd

D = sys.argv[1]
gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float))))) if len(x) else float("nan")
cases = pd.read_csv(f"{D}/fresh_cases.csv").set_index("case")
t = pd.read_csv(f"{D}/fresh_timed.csv", low_memory=False)
st = t.pivot_table(index="case", columns="origin", values="status", aggfunc="last")
ns = t.pivot_table(index="case", columns="origin", values="device_ns", aggfunc="last")
print("status:", {o: st[o].value_counts().to_dict() for o in st})
ok = ns.dropna(subset=["legacy", "heuristic", "model"])
ok = ok[(st.loc[ok.index] == "ok").all(axis=1)]
ok = ok.assign(
    m=ok.model / ok.legacy,
    r=ok.heuristic / ok.legacy,
    mr=ok.model / ok.heuristic,
    stratum=cases.stratum.reindex(ok.index).values,
)
bins = [0, 0.5, 0.8, 0.95, 1.05, 1.10, 1.25, 1.5, np.inf]
lab = ["<0.50", "0.50-0.80", "0.80-0.95", "0.95-1.05", "1.05-1.10", "1.10-1.25", "1.25-1.50", ">1.50"]
print(f"\nn={len(ok)} with legacy, rules and model all ok")
for k, n in (("m", "model"), ("r", "rules")):
    x = ok[k]
    reg = x[x > 1.05]
    print(
        f"  {n:6s} vs legacy: geomean {gm(x):.3f}  best {x.min():.2f}  p25/50/75 {x.quantile(.25):.2f}/{x.median():.2f}/{x.quantile(.75):.2f}  p95 {x.quantile(.95):.3f}"
        f"  worst {x.max():.2f}  regressions {len(reg)} (severity geomean {gm(reg):.3f})"
    )
print(
    pd.DataFrame(
        {
            n: pd.cut(ok[k], bins, labels=lab, right=False).value_counts().reindex(lab)
            for k, n in (("m", "model/legacy"), ("r", "rules/legacy"), ("mr", "model/rules"))
        }
    ).to_string()
)
print(
    f"  model vs rules: geomean {gm(ok.mr):.3f}, model >5% faster {(ok.mr < 1/1.05).sum()}, >5% slower {(ok.mr > 1.05).sum()}, same config {(ok.model == ok.heuristic).sum()}"
)
print("\nper stratum (model | rules):")
g = ok.groupby("stratum").agg(
    n=("m", "size"),
    model_gm=("m", gm),
    rules_gm=("r", gm),
    model_regr=("m", lambda x: int((x > 1.05).sum())),
    rules_regr=("r", lambda x: int((x > 1.05).sum())),
    model_gt125=("m", lambda x: int((x > 1.25).sum())),
    rules_gt125=("r", lambda x: int((x > 1.25).sum())),
    model_worst=("m", "max"),
    rules_worst=("r", "max"),
    m_vs_r=("mr", gm),
)
print(g.round(3).to_string())
try:
    s = pd.read_csv(f"{D}/fresh_sweep.csv", low_memory=False)
    s = s[s.status == "ok"]
    picks = pd.read_csv(f"{D}/fresh_picks.csv").set_index("case").config
    rows = []
    for c, x in s.groupby("case"):
        cand = x[x.origin.isin(["enumerated", "heuristic"])]
        if not len(cand) or c not in picks.index:
            continue
        best = cand.device_ns.min()
        mp = cand[cand.config == picks[c]].device_ns
        rp = cand[cand.origin == "heuristic"].device_ns
        lg = x[x.origin == "legacy"].device_ns
        if len(mp) and len(rp):
            rows.append(
                dict(
                    case=c,
                    model=mp.min() / best,
                    rules=rp.min() / best,
                    legacy=(lg.min() / best) if len(lg) else np.nan,
                    stratum=cases.stratum.get(c),
                )
            )
    r = pd.DataFrame(rows)
    print(
        f"\nregret on the fully swept subset (n={len(r)}): model {gm(r.model):.3f} (p95 {r.model.quantile(.95):.2f}, worst {r.model.max():.2f})  "
        f"rules {gm(r.rules):.3f} (p95 {r.rules.quantile(.95):.2f}, worst {r.rules.max():.2f})  legacy {gm(r.legacy.dropna()):.3f}"
    )
    print(
        "  by stratum:",
        r.groupby("stratum")
        .agg(n=("model", "size"), model=("model", gm), rules=("rules", gm))
        .round(3)
        .to_dict("index"),
    )
except FileNotFoundError:
    print("\n(no sweep yet)")
