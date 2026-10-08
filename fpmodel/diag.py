import sys

sys.path.insert(0, ".")
from data import *
from model import *

d = load(list(SETS))
pred = np.zeros(len(d))
P = {}
for a in ("wh", "bh"):
    m = (d.arch_ == a).to_numpy()
    g = geometry(d[m])
    t, pr = predict(g, PARAMS, parts=True)
    pred[m] = t
    for k, v in pr.items():
        P.setdefault(k, np.zeros(len(d)))[m] = v
d["err"] = np.log(pred / d.device_ns)
d["nK"] = P["nK"]
d["bound"] = np.where(P["read"] > P["comp"], "read", "comp")
d["us"] = d.device_ns / 1e3
d["src"] = d.a_mem.str[:4] + "/" + d.b_mem.str[:4]
d["nKbin"] = pd.cut(d.nK, [0, 1, 2, 4, 16, 1e9])
d["usbin"] = pd.cut(d.us, [0, 30, 100, 1000, 1e9])
# within-problem error (only relative matters): centre per problem
d["cerr"] = d.err - d.groupby("problem_id").err.transform("median")
for k in ["family", "bound", "nKbin", "usbin", "src", "fidelity", "a_dtype"]:
    print(
        d.groupby(["arch_", k], observed=True)
        .agg(n=("err", "size"), med=("err", "median"), cspread=("cerr", lambda s: s.abs().median()))
        .round(2)
        .unstack(0)
        .to_string(),
        "\n",
    )
