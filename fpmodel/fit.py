"""Calibrate the model's event constants on sweep data and judge the picks out of fold.

Loss: within-problem log-time error (each problem's median removed; only the order inside a problem matters for
selection) plus a small absolute term so latencies stay physical. Every problem weighs the same.
Protocols:
  wh_cv    5-fold by problem over WH designed + targeted (fit on 4/5, pick on 1/5)
  bh_cv    the same on BH
  bh_xfer  constants fitted on all WH, BH spec rates (clock, NoC width, DRAM) swapped in; no BH data used
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, numpy as np, pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, ".")
from data import *
from model import *

NAMES = list(PARAMS)
LO = {"dram_eff": 0.2, "noc_eff": 0.1, "l1_frac": 0.05}
HI = {"dram_eff": 1.0, "noc_eff": 1.0, "l1_frac": 3.0}
ABS_W = 0.5  # absolute log error weight (relative error 1): keeps predicted times calibrated, not just ordered


def residuals(x, g, logt, pid_codes, w):
    p = dict(zip(NAMES, np.exp(x)))
    e = np.log(predict(g, p)) - logt
    med = pd.Series(e).groupby(pid_codes).transform("median").to_numpy()
    return np.concatenate([(e - med) * w, ABS_W * e * w])


def subset(g, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in g.items()}


def fit(d, rows, p0=None):
    x = d.iloc[rows]
    g = geometry(x)
    pid = pd.factorize(x.problem_id)[0]
    w = 1.0 / np.sqrt(np.bincount(pid)[pid])
    p0 = dict(PARAMS, **(p0 or {}))
    lo = np.log([LO.get(n, 1e-3) for n in NAMES])
    hi = np.log([HI.get(n, 1e7) for n in NAMES])
    x0 = np.clip(np.log([p0[n] for n in NAMES]), lo + 1e-9, hi - 1e-9)
    r = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        args=(g, np.log(x.device_ns.to_numpy()), pid, w),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=300,
    )
    return dict(zip(NAMES, np.exp(r.x)))


def predict_rows(d, rows, p):
    return predict(geometry(d.iloc[rows]), p)


if __name__ == "__main__":
    d = load(list(SETS))
    wh = np.flatnonzero((d.arch_ == "wh").to_numpy())
    bh = np.flatnonzero((d.arch_ == "bh").to_numpy())
    pred = np.full(len(d), np.nan)
    fitted = {}
    for name, rows in (("wh", wh), ("bh", bh)):
        probs = d.problem_id.iloc[rows].unique()
        rng = np.random.default_rng(0)
        rng.shuffle(probs)
        for k in range(5):
            test = np.isin(d.problem_id.iloc[rows], probs[k::5])
            p = fit(d, rows[~test])
            pred[rows[test]] = predict_rows(d, rows[test], p)
        fitted[name] = fit(d, rows)
    evaluate(d, pred, "fitted, 5-fold by problem")
    print("pair acc, median |log err|, p90:", rank_quality(d, pred))
    xfer = predict_rows(d, bh, fitted["wh"])
    evaluate(d.iloc[bh].reset_index(drop=True), xfer, "WH constants -> BH spec")
    print(json.dumps({a: {k: round(v, 3) for k, v in p.items()} for a, p in fitted.items()}, indent=1))
    json.dump(fitted, open("fitted.json", "w"), indent=1)
    np.save("pred_cv.npy", pred)
