"""Usage-weighted view of a timed suite run: per real model, total matmul device time (sum over its traced calls of
call count x per-call time) with the model's picks, the rules' picks and legacy; plus the share of calls that regress.
usage: usage_eval.py TIMED_CSV USAGE_CSV BOARD [OUT_JSON]"""
import sys, json, numpy as np, pandas as pd

timed, usage, board = sys.argv[1:4]
t = pd.read_csv(timed, low_memory=False)
ok = (
    t[t.status == "ok"]
    .pivot_table(index="case", columns="origin", values="device_ns", aggfunc="last")
    .dropna(subset=["legacy", "heuristic", "model"])
)
u = pd.read_csv(usage)
u = u[u.board == board]
u = u.groupby(["model", "case"], as_index=False)["count"].sum()
u = u[u.case.isin(ok.index)]
short = (
    lambda s: s.split("HF_MODEL:")[1].rstrip("]")
    if "HF_MODEL:" in s
    else "/".join(s.split("::")[0].split("/")[-3:-1] or [s])
)
rows = []
for m, g in u.groupby("model"):
    x = ok.loc[g.case]
    w = g["count"].to_numpy()
    tl, tr, tm = (w * x.legacy).sum(), (w * x.heuristic).sum(), (w * x.model).sum()
    reg = (x.model / x.legacy > 1.05).to_numpy()
    rows.append(
        dict(
            model=short(m),
            source=m,
            cases=len(g),
            calls=int(w.sum()),
            legacy_us=tl / 1e3,
            rules_vs_legacy=tr / tl,
            model_vs_legacy=tm / tl,
            model_vs_rules=tm / tr,
            regr_cases=int(reg.sum()),
            regr_call_share=float((w * reg).sum() / w.sum()),
        )
    )
r = pd.DataFrame(rows).sort_values("legacy_us", ascending=False)
# collapse duplicate model entries (same demo traced under two test ids): keep the larger
r = r.drop_duplicates("model", keep="first")
allc = u.groupby("case")["count"].sum()
x = ok.loc[allc.index]
w = allc.to_numpy()
tot = dict(
    board=board,
    cases=int(len(allc)),
    calls=int(w.sum()),
    models=int(len(r)),
    rules_vs_legacy=float((w * x.heuristic).sum() / (w * x.legacy).sum()),
    model_vs_legacy=float((w * x.model).sum() / (w * x.legacy).sum()),
    regr_call_share=float((w * (x.model / x.legacy > 1.05)).sum() / w.sum()),
    rules_regr_call_share=float((w * (x.heuristic / x.legacy > 1.05)).sum() / w.sum()),
)
print(f"{board}: {tot['cases']} traced cases, {tot['calls']} calls, {tot['models']} models")
print(f"  call-weighted matmul time vs legacy: model {tot['model_vs_legacy']:.3f}  rules {tot['rules_vs_legacy']:.3f}")
print(
    f"  share of calls more than 5% slower than legacy: model {tot['regr_call_share']:.3%}  rules {tot['rules_regr_call_share']:.3%}"
)
print(
    r[
        [
            "model",
            "cases",
            "calls",
            "legacy_us",
            "model_vs_legacy",
            "rules_vs_legacy",
            "model_vs_rules",
            "regr_cases",
            "regr_call_share",
        ]
    ]
    .round(3)
    .to_string(index=False)
)
if len(sys.argv) > 4:
    json.dump(dict(total=tot, models=r.round(4).to_dict("records")), open(sys.argv[4], "w"), indent=1)
