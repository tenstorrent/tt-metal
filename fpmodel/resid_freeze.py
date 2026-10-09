"""Freeze base + residual: base constants (a fitted JSON on all training data) and the residual tree trained on all
training rows of that arch with those constants. usage: resid_freeze.py ARCH CONST_JSON OUT_MODEL  (env as resid_cv.py)"""
import os, sys, json

sys.path.insert(0, ".")
import numpy as np, lightgbm as lgb
import model7 as M
import resid_cv as R
from data import load, SETS, CAND

arch, cpath, out = sys.argv[1:4]
p = json.load(open(cpath))
d = M.annotate(load(list(SETS)))
x = d[(d.arch_ == arch)].reset_index(drop=True)
b, g, pr = R.base_parts(x, p)
f = R.features(x, g, pr)
y = R.centered_target(x, b)
cand = x.origin.isin(CAND).to_numpy()
w = 1.0 / np.sqrt(x.problem_id.map(x.problem_id.value_counts()).to_numpy())
m = lgb.LGBMRegressor(**R.PARAMS).fit(f[cand], y[cand], sample_weight=w[cand])
m.booster_.save_model(out)
c = m.predict(f)
print(
    arch,
    "rows",
    int(cand.sum()),
    "in-sample |corr| median %.3f p90 %.3f" % (np.median(np.abs(c)), np.quantile(np.abs(c), 0.9)),
)
