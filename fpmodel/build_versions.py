"""Build the per-version data files for the results page: one JSON per model version (definition, CV results, device
results with roofline points) plus index.json and experiments.json. usage: build_versions.py OUTDIR"""
import os, sys, json, math, glob, subprocess

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from data import load, SETS, CAND, evaluate, gm

W = "/localdev/rmiller/mm-oob-model-work"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
BINS = [1.05, 1.10, 1.25, 1.5, np.inf]
DIST = [0, 0.5, 0.8, 0.95, 1.05, 1.10, 1.25, 1.5, np.inf]
DLAB = ["<0.50", "0.50-0.80", "0.80-0.95", "0.95-1.05", "1.05-1.10", "1.10-1.25", "1.25-1.50", ">1.50"]
TB = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}
CYC = {"LoFi": 16, "HiFi2": 32, "HiFi3": 48, "HiFi4": 64}
r3 = lambda x: None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 3)


def tail(vl):
    vl = pd.Series(vl).dropna()
    reg = vl[vl > 1.05]
    return dict(
        n=int(len(vl)),
        n_regr=int(len(reg)),
        bins=pd.cut(reg, BINS).value_counts().sort_index().astype(int).tolist(),
        severity=r3(gm(reg)) if len(reg) else None,
        worst=r3(vl.max()),
        p95=r3(vl.quantile(0.95)),
        geomean=r3(gm(vl)),
        best=r3(vl.min()),
        q=[r3(v) for v in vl.quantile([0.25, 0.5, 0.75])],
        n_gt125=int((vl > 1.25).sum()),
        dist=pd.cut(vl, DIST, labels=DLAB, right=False).value_counts().reindex(DLAB).astype(int).tolist(),
    )


# ---------- cross-validation ----------
D = load(list(SETS))


def tiebreak(pred, m=0.05):
    s = np.full(len(D), 1e12)
    x = D.assign(pred=pred)
    for _, g in x[x.origin.isin(CAND)].groupby("problem_id", sort=False):
        near = g[g.pred <= g.pred.min() * (1 + m)]
        s[near.assign(k=near.in0_block_w.fillna(1)).sort_values(["k", "pred"]).index[0]] = 0
    return s


def cv_block(pred):
    r = evaluate(D, pred, "", quiet=True)
    out = {}
    for s, g in r.groupby("set", sort=False):
        out[s] = dict(
            regret=r3(gm(g.regret)),
            p95_regret=r3(g.regret.quantile(0.95)),
            max_regret=r3(g.regret.max()),
            vs_legacy=tail(g.vs_legacy),
        )
    return out, r


def rules_cv():
    r = evaluate(D, np.load("pred_cv.npy"), "", quiet=True)
    return {
        s: dict(
            regret=r3(gm(g.rules_regret.dropna())),
            p95_regret=r3(g.rules_regret.quantile(0.95)),
            max_regret=r3(g.rules_regret.max()),
            vs_legacy=tail(g.rules_vs_legacy),
        )
        for s, g in r.groupby("set", sort=False)
    }


# ---------- device sets ----------
def tensor_bytes(shape, dt):
    s = [int(v) for v in str(shape).split("x")]
    lead = math.prod(s[:-2]) if len(s) > 2 else 1
    return lead * math.ceil(s[-2] / 32) * math.ceil(s[-1] / 32) * TB[dt]


def device_block(path, set_id, label, when, note=""):
    d = pd.read_csv(path, low_memory=False)
    st = d.pivot_table(index="case", columns="origin", values="status", aggfunc="last")
    ns = d.pivot_table(index="case", columns="origin", values="device_ns", aggfunc="last")
    ok = ns.dropna(subset=["legacy", "heuristic", "model"])
    ok = ok[(st.loc[ok.index] == "ok").all(axis=1)]
    m, r, mr = ok.model / ok.legacy, ok.heuristic / ok.legacy, ok.model / ok.heuristic
    info = (
        d[(d.origin == "legacy") & d.case.isin(ok.index)]
        .drop_duplicates("case", keep="last")
        .set_index("case")
        .loc[ok.index]
    )
    roof = []
    for c, x in info.iterrows():
        outdt = x.out_dtype if isinstance(x.out_dtype, str) and x.out_dtype in TB else x.a_dtype
        sizes = [
            (x.a_mem, tensor_bytes(x.a_shape, x.a_dtype)),
            (x.b_mem, tensor_bytes(x.b_shape, x.b_dtype)),
            (x.out_mem, tensor_bytes(f"{x.batch}x{x.M}x{x.N}", outdt)),
        ]
        dram = sum(s for mm, s in sizes if mm == "dram")
        gx, gy = [int(v) for v in str(x.grid).replace("-", "x").split("x")[:2]]
        flops = 2.0 * x.batch * x.M * x.K * x.N
        peak = gx * gy * 1e9 * 2 * 32**3 / CYC[x.fidelity] / 1e12
        tf = lambda o: flops / ok.loc[c, o] / 1e3
        roof.append(
            [
                round(flops / dram, 2) if dram else None,
                x.fidelity,
                round(peak, 1),
                round(tf("legacy"), 4),
                round(tf("heuristic"), 4),
                round(tf("model"), 4),
                r3(m[c]),
                r3(r[c]),
                bool(x.a_mem != "dram" or x.b_mem != "dram"),
            ]
        )
    return dict(
        id=set_id,
        label=label,
        when=when,
        note=note,
        n=int(len(ok)),
        status={o: st[o].value_counts().to_dict() for o in st},
        model=tail(m),
        rules=tail(r),
        model_vs_rules=dict(
            geomean=r3(gm(mr)),
            faster=int((mr < 1 / 1.05).sum()),
            slower=int((mr > 1.05).sum()),
            same=int((ok.model == ok.heuristic).sum()),
            dist=tail(mr)["dist"],
        ),
        roof=roof,
    )


def sweep_regret(sweep_csv, picks_csv):
    s = pd.read_csv(sweep_csv, low_memory=False)
    s = s[s.status == "ok"]
    picks = pd.read_csv(picks_csv).set_index("case").config
    rows = []
    for c, x in s.groupby("case"):
        cand = x[x.origin.isin(CAND)]
        if not len(cand) or c not in picks.index:
            continue
        best = cand.device_ns.min()
        mp, rp = cand[cand.config == picks[c]].device_ns, cand[cand.origin == "heuristic"].device_ns
        if len(mp) and len(rp):
            rows.append((mp.min() / best, rp.min() / best))
    a = np.array(rows)
    if not len(a):
        return None
    return dict(
        n=int(len(a)),
        model=dict(regret=r3(gm(a[:, 0])), p95=r3(np.quantile(a[:, 0], 0.95)), worst=r3(a[:, 0].max())),
        rules=dict(regret=r3(gm(a[:, 1])), p95=r3(np.quantile(a[:, 1], 0.95)), worst=r3(a[:, 1].max())),
    )


# ---------- versions ----------
V = json.load(open("versions.json"))
commit = {"v5_tb": "3888c8c", "v5_argmin": "3888c8c", "v6": "779a8de"}
rules = rules_cv()
CVPRED = {
    "v5_tb": tiebreak(np.load("pred_cv.npy")),
    "v5_argmin": np.load("pred_cv.npy"),
    "v6": np.load("pred_cv_x_noc_rl_burstl1+dram_eff=0.77_launch_us=0.5_rl_init=340_u2d=100.npy"),
}
DEV = {
    "v5_tb": [
        device_block(
            f"{W}/device/holdout_fp_timed.csv",
            "holdout70001",
            "Frozen holdout (300 cases)",
            "2026-10-08",
            "First and only use of this holdout; rules timed on the same build (4b385bb).",
        ),
        device_block(
            f"{W}/device/suite719_fp_timed.csv",
            "suite719",
            "Real-case suite (719)",
            "2026-10-08",
            "The rules were tuned on this suite.",
        ),
    ],
    "v5_argmin": [
        device_block(
            f"{W}/device/suite719_fp_argmin_merged.csv",
            "suite719",
            "Real-case suite (719)",
            "2026-10-08",
            "366 changed picks re-timed with legacy and rules in a second session; the rest from the first run.",
        )
    ],
    "v6": [],
}
fresh = f"{W}/fresh/fresh_timed.csv"
if os.path.exists(fresh):
    b = device_block(
        fresh,
        "fresh90001",
        "Fresh random set (seeds 90001-7)",
        "2026-10-08",
        "Out-of-sample: generated after v6 was frozen; includes out-of-distribution shapes.",
    )
    if os.path.exists(f"{W}/fresh/fresh_sweep.csv"):
        b["regret"] = sweep_regret(f"{W}/fresh/fresh_sweep.csv", f"{W}/fresh/fresh_picks.csv")
    DEV["v6"].append(b)
index = []
for vid, meta in V.items():
    cv, _ = cv_block(CVPRED[vid])
    doc = dict(id=vid, commit=commit.get(vid), **meta, cv=cv, rules_cv=rules, device=DEV.get(vid, []))
    if os.path.exists(f"{OUT}/coverage_{vid}.json"):  # written by coverage.py
        doc["coverage"] = f"data/coverage_{vid}.json"
    json.dump(doc, open(f"{OUT}/{vid}.json", "w"), separators=(",", ":"))
    index.append(dict(id=vid, title=meta["title"], file=f"data/{vid}.json"))
json.dump(dict(versions=index, latest="v6"), open(f"{OUT}/index.json", "w"), indent=1)

# ---------- CV-only experiments ----------
EXP = [
    ("base (v5 argmin)", "pred_cv.npy"),
    ("+ reload re-init (fitted)", "pred_cv_x_rl.npy"),
    ("+ DRAM in-flight efficiency", "pred_cv_x_dram.npy"),
    ("+ unpack reuse", "pred_cv_x_unp.npy"),
    ("+ per-core launch", "pred_cv_x_launch.npy"),
    ("DRAM eff pinned 0.77", "pred_cv_x_base+dram_eff=0.77.npy"),
    ("no burst congestion", "pred_cv_x_base+burst_KB=1e9.npy"),
    ("launch pinned 2 us", "pred_cv_x_base+launch_us=2.npy"),
    ("+ write overlap", "pred_cv_x_wov.npy"),
    ("+ link load", "pred_cv_x_noc.npy"),
    ("+ link load, DRAM 0.77", "pred_cv_x_noc+dram_eff=0.77.npy"),
    ("+ link load, no burst", "pred_cv_x_noc+burst_KB=1e9.npy"),
    (
        "+ link, DRAM 0.77, measured launch/spill",
        "pred_cv_x_noc_rl+dram_eff=0.77_launch_us=0.5_rl_init=340_u2d=100.npy",
    ),
    ("  same, no burst", "pred_cv_x_noc_rl+dram_eff=0.77_launch_us=0.5_rl_init=340_u2d=100_burst_KB=1e9.npy"),
    ("  same, burst on L1 only (= v6)", "pred_cv_x_noc_rl_burstl1+dram_eff=0.77_launch_us=0.5_rl_init=340_u2d=100.npy"),
]
exp = []
for name, f in EXP:
    if not os.path.exists(f):
        continue
    cv, _ = cv_block(np.load(f))
    exp.append(dict(name=name, cv=cv))
json.dump(exp, open(f"{OUT}/experiments.json", "w"), separators=(",", ":"))
print("wrote", [i["id"] for i in index], "experiments", len(exp))
