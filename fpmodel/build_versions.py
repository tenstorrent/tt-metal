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
BASE_SETS = ["wh_designed", "wh_target", "bh_designed"]  # the training sets up to v10
D = load(BASE_SETS)
D_ALL = load(list(SETS))  # v11 on: plus the swept fresh / miss sets


def tiebreak(pred, m=0.05):
    s = np.full(len(D), 1e12)  # v5 only (base sets)
    x = D.assign(pred=pred)
    for _, g in x[x.origin.isin(CAND)].groupby("problem_id", sort=False):
        near = g[g.pred <= g.pred.min() * (1 + m)]
        s[near.assign(k=near.in0_block_w.fillna(1)).sort_values(["k", "pred"]).index[0]] = 0
    return s


def cv_block(pred):
    r = evaluate(D if len(pred) == len(D) else D_ALL, pred, "", quiet=True)
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
        clk = 1.35e9 if "black" in str(x.get("arch", "")) else 1e9
        peak = gx * gy * clk * 2 * 32**3 / CYC[x.fidelity] / 1e12
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
    bh = "black" in str(info.arch.iloc[0]) if "arch" in info and len(info) else False
    return dict(
        id=set_id,
        arch="bh" if bh else "wh",
        dram_TBs=0.512 if bh else 0.288,
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
commit = {
    "v5_tb": "3888c8c",
    "v5_argmin": "3888c8c",
    "v6": "779a8de",
    "v7": "0e8ab14",
    "v8": "0554d53",
    "v9": "bf6978e",
}
rules = rules_cv()
CVPRED = {
    "v5_tb": tiebreak(np.load("pred_cv.npy")),
    "v5_argmin": np.load("pred_cv.npy"),
    "v6": np.load("pred_cv_x_noc_rl_burstl1+dram_eff=0.77_launch_us=0.5_rl_init=340_u2d=100.npy"),
    "v7": np.load("pred_cv_m7_pad.npy"),  # current code with the v8 term (pad) switched off
    "v8": np.load("pred_cv_v8.npy"),
    "v9": np.load("pred_cv_v9.npy"),
    "v10": np.load("pred_cv_v10.npy"),
    "v11": np.load("pred_cv_v11.npy"),
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
if os.path.exists(f"{W}/fresh/fresh2_timed.csv"):
    DEV["v6"].append(
        device_block(
            f"{W}/fresh/fresh2_timed.csv",
            "fresh91001",
            "Fresh random set, draw 2 (seeds 91001-7)",
            "2026-10-08",
            "Second out-of-sample draw, picked with the frozen v6 constants.",
        )
    )
if os.path.exists(f"{W}/device/suite719_v6_timed.csv"):
    b = device_block(
        f"{W}/device/suite719_v6_timed.csv",
        "suite719",
        "Real-case suite (719)",
        "2026-10-08",
        "Not in v6's training data; the rules were tuned on this suite.",
    )
    if os.path.exists(f"{W}/usage/usage_v6_wh.json"):  # written by usage_eval.py
        b["usage"] = json.load(open(f"{W}/usage/usage_v6_wh.json"))
    DEV["v6"].append(b)


FROZEN = {v: f"{W}/fresh/frozen_{v}_" for v in ("v7", "v9", "v10", "v11")}
FROZEN["v8"] = f"{W}/fresh/frozen_v8_"


def accuracy(path, vid, arch):
    """predicted vs measured device time on every timed config of a set, with the version's frozen constants (current
    model7 code, terms without constants in the file switched off)"""
    import importlib, model7

    cf = FROZEN.get(vid, "") + arch + ".json"
    if not os.path.exists(cf):
        return None
    importlib.reload(model7)
    p = json.load(open(cf))
    for name, (_, _, term) in model7.CONSTANTS.items():
        if term and name not in p:
            model7.OFF.add(term)
    t = pd.read_csv(path, low_memory=False)
    t = (
        t[(t.status == "ok") & (t.device_ns > 0)]
        .drop_duplicates(["case", "origin"], keep="last")
        .reset_index(drop=True)
    )
    t = t[t.family.isin(["2d", "1d_in0", "1d_in1", "reuse", "multicore"])].reset_index(drop=True)
    t["arch_"] = arch
    t = model7.annotate(t)
    pred = model7.predict(model7.geometry(t), p)
    meas = t.device_ns.to_numpy(float)
    e = np.log(pred / meas)
    acc, n = 0, 0
    for _, g in t.assign(pred=pred).groupby("case"):
        a, q = g.device_ns.to_numpy(), g.pred.to_numpy()
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if abs(np.log(a[i] / a[j])) > np.log(1.05):
                    n += 1
                    acc += (a[i] < a[j]) == (q[i] < q[j])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(t), min(len(t), 900), replace=False)
    return dict(
        n=int(len(t)),
        median_abs_err=r3(np.median(np.abs(np.expm1(e)))),
        within10=r3(np.mean(np.abs(e) < np.log(1.10))),
        within25=r3(np.mean(np.abs(e) < np.log(1.25))),
        bias=r3(np.expm1(np.median(e))),
        pair_acc=r3(acc / n) if n else None,
        pairs=int(n),
        points=[[round(float(meas[i]) / 1e3, 2), round(float(pred[i]) / 1e3, 2)] for i in idx],
    )


def add(vid, path, set_id, label, note, usage=None, done=None):
    if os.path.exists(path) and (done is None or os.path.exists(done)):  # done: the run's completion flag
        b = device_block(path, set_id, label, "2026-10-09", note)
        try:
            b["accuracy"] = accuracy(path, vid, b["arch"])
        except Exception as ex:  # the page still builds; the block just has no accuracy panel
            print("accuracy failed", vid, set_id, ex)
        if usage and os.path.exists(usage):
            b["usage"] = json.load(open(usage))
        DEV.setdefault(vid, []).append(b)


BHU = (
    "From bh-30's full sweeps: constants fitted on the 153 bh_0200 problems, scored on the other designed BH problems."
)
# (version, timed csv, set id, label, note, usage json, completion flag)
RUNS = [
    (
        "v7",
        "fresh/fresh3_timed.csv",
        "fresh92001",
        "Fresh random set, draw 3 (seeds 92001-7)",
        "Out-of-sample: generated and picked after v7 was frozen.",
        None,
        "fresh/chain_v7.done",
    ),
    (
        "v7",
        "miss/bh_unseen_v7_timed.csv",
        "bh-unseen",
        "BH designed problems never used in training (1021)",
        BHU,
        None,
        None,
    ),
    (
        "v7",
        "fresh/freshbh_timed.csv",
        "freshbh93001",
        "BH fresh random set (seeds 93001-7), bh-30",
        "Out-of-sample on BH: picked on bh-30 with the frozen v7 BH constants.",
        None,
        "fresh/freshbh.done",
    ),
    (
        "v7",
        "device/suite719_v7_timed.csv",
        "suite719",
        "Real-case suite (719)",
        "Not in training; the rules were tuned on this suite. Changed picks re-timed with legacy and rules in one session.",
        "usage/usage_v7_wh.json",
        None,
    ),
    (
        "v7",
        "fresh/fresh2_v7_timed.csv",
        "draw2-informed",
        "Draw 2 (seeds 91001-7), not out-of-sample for v7",
        "v7's terms were found from draw 2's misses, so this set informed v7; it is not counted in the bounds.",
        None,
        None,
    ),
    (
        "v8",
        "miss/bh_unseen_v8_timed.csv",
        "bh-unseen",
        "BH designed problems never used in training (1021)",
        BHU,
        None,
        None,
    ),
    (
        "v9",
        "fresh/fresh4_timed.csv",
        "fresh94001",
        "Fresh random set, draw 4 (seeds 94001-7)",
        "Out-of-sample: generated and picked after v9 was frozen.",
        None,
        "fresh/chain_v8.done",
    ),
    (
        "v9",
        "fresh/freshbh2_timed.csv",
        "freshbh95001",
        "BH fresh random set 2 (seeds 95001-7), bh-30",
        "Out-of-sample on BH: picked on bh-30 with the frozen v9 BH constants.",
        None,
        "fresh/freshbh2.done",
    ),
    (
        "v9",
        "device/suite719_v9_timed.csv",
        "suite719",
        "Real-case suite (719)",
        "Not in training; the rules were tuned on this suite. Changed picks re-timed with legacy and rules in one session.",
        "usage/usage_v9_wh.json",
        "miss/v9suite.done",
    ),
    (
        "v9",
        "miss/bh_unseen_v9_timed.csv",
        "bh-unseen",
        "BH designed problems never used in training (1021)",
        BHU,
        None,
        None,
    ),
    (
        "v10",
        "miss/bh_unseen_v10_timed.csv",
        "bh-unseen",
        "BH designed problems never used in training (1021)",
        BHU,
        None,
        None,
    ),
    (
        "v11",
        "fresh/fresh5_timed.csv",
        "fresh96001",
        "Fresh random set, draw 5 (seeds 96001-7)",
        "Out-of-sample: generated and picked after v11 was frozen.",
        None,
        "fresh/chain_v10.done",
    ),
    (
        "v11",
        "miss/bh_unseen_v11_timed.csv",
        "bh-unseen",
        "BH designed problems never used in training (1021)",
        BHU,
        None,
        None,
    ),
]
for vid, path, sid, label, note, usage, done in RUNS:
    add(vid, f"{W}/{path}", sid, label, note, f"{W}/{usage}" if usage else None, f"{W}/{done}" if done else None)


def bounds(blocks, bh=False):
    """pooled regression rates over the fresh draws (WH, or BH with bh=True) with one-sided 95% Clopper-Pearson upper
    bounds"""
    from scipy.stats import beta

    fr = [
        b
        for b in blocks
        if b["id"].startswith("freshbh" if bh else "fresh") and (bh or not b["id"].startswith("freshbh"))
    ]
    if not fr:
        return None
    n = sum(b["n"] for b in fr)
    out = dict(sets=[b["id"] for b in fr], n=n)
    for who in ("model", "rules"):
        for key, k in (("gt105", sum(b[who]["n_regr"] for b in fr)), ("gt125", sum(b[who]["n_gt125"] for b in fr))):
            out[f"{who}_{key}"] = dict(k=k, rate=r3(k / n), upper95=r3(beta.ppf(0.95, k + 1, n - k)))
    return out


index = []
for vid, meta in V.items():
    cv, _ = cv_block(CVPRED[vid])
    doc = dict(id=vid, commit=commit.get(vid), **meta, cv=cv, rules_cv=rules, device=DEV.get(vid, []))
    doc["bounds"] = bounds(doc["device"])
    doc["bounds_bh"] = bounds(doc["device"], bh=True)
    sheet = {
        "v7": "abl/v7_nopad.json",
        "v8": "abl/v8_bhharv.json",
        "v9": "abl/v9.json",
        "v10": "abl/v10.json",
        "v11": "abl/v11.json",
    }.get(vid)
    if sheet and os.path.exists(sheet):  # written by cv7.py: per-arch value, fold spread, pinned, at bound
        import model7

        doc["constants_sheet"] = [
            dict(
                name=k,
                meaning=model7.CONSTANTS.get(k, (None, ""))[1],
                source={
                    a: ("measured: " + model7.PINNED[k][1]) if a in model7.PINNED.get(k, ({},))[0] else "fitted"
                    for a in v
                },
                **{a: v[a] for a in v},
            )
            for k, v in json.load(open(sheet))["constants"].items()
        ]
    if os.path.exists(f"{OUT}/coverage_{vid}.json"):  # written by coverage.py
        doc["coverage"] = f"data/coverage_{vid}.json"
    json.dump(doc, open(f"{OUT}/{vid}.json", "w"), separators=(",", ":"))
    index.append(dict(id=vid, title=meta["title"], file=f"data/{vid}.json"))
json.dump(dict(versions=index, latest="v11"), open(f"{OUT}/index.json", "w"), indent=1)

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
