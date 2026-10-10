"""Coverage of a target set by the residual's training data: for each target config, the mean distance to its 10 nearest
training configs in standardized (signed-log) mechanism-feature space; error by distance band; adversarial validation
(which features separate target from training). usage: coverage_map.py TARGET_CSV [TARGET_CSV ...]  (frozen v18r, WH)"""
import sys, json

sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
import model7 as M
import resid_cv as R
from data import load, SETS, CAND
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

W = "/localdev/rmiller/mm-oob-model-work"
p = json.load(open(f"{W}/fresh/frozen_v18r_wh.json"))
bst = lgb.Booster(model_file=f"{W}/fresh/frozen_v18r_wh.lgb")
d = M.annotate(load(list(SETS)))
d = d[(d.arch_ == "wh") & d.origin.isin(CAND)].reset_index(drop=True)
b, g, pr = R.base_parts(d, p)
F = R.features(d, g, pr)
norm = lambda s: str(s).replace(" ", "")
s = pd.concat([pd.read_csv(f, low_memory=False) for f in sys.argv[1:]])
s = s[s.status == "ok"].assign(cfg=lambda z: z.config.map(norm)).drop_duplicates(["case", "cfg"], keep="last")
s = s.reset_index(drop=True).assign(arch_="wh")
s = M.annotate(s)
bs, gs, ps = R.base_parts(s, p)
Fs = R.features(s, gs, ps)
s["pred"] = bs * np.exp(bst.predict(Fs))
s["err"] = np.abs(np.log(s.pred / s.device_ns))
cols = [c for c in F.columns if c != "log_base_us"]
T = lambda X: np.sign(X[cols].fillna(-1)) * np.log1p(np.abs(X[cols].fillna(-1)))
A, B = T(F), T(Fs)
mu, sd = A.mean(), A.std().replace(0, 1)
A, B = ((A - mu) / sd).to_numpy(), ((B - mu) / sd).to_numpy()
nn = NearestNeighbors(n_neighbors=11).fit(A)
s["dist"] = nn.kneighbors(B, n_neighbors=10)[0].mean(1)
dt = nn.kneighbors(A[np.random.default_rng(0).choice(len(A), 3000, replace=False)])[0][:, 1:].mean(1)
q = np.quantile(dt, [0.5, 0.9, 0.99])
print("training configs' own 10-NN distance: median %.2f  p90 %.2f  p99 %.2f" % tuple(q))
s["band"] = pd.cut(s.dist, [0, q[0], q[1], q[2], np.inf], labels=["dense", "typical", "sparse", "outside"])
agg = s.groupby("band", observed=True).err.agg(
    n="size", median_err="median", p90_err=lambda v: v.quantile(0.9), within25=lambda v: (v < np.log(1.25)).mean()
)
print(agg.round(3).to_string())
print("Spearman(distance, |error|) = %.2f" % s[["dist", "err"]].corr("spearman").iloc[0, 1])
X, y = np.vstack([A, B]), np.r_[np.zeros(len(A)), np.ones(len(B))]
clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1, class_weight="balanced")
pp = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
print("adversarial AUC (target vs training): %.3f" % roc_auc_score(y, pp))
clf.fit(X, y)
imp = pd.Series(clf.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
print("features that separate target from training:", (imp / imp.sum()).head(8).round(2).to_dict())
s[["case", "cfg", "origin", "device_ns", "pred", "err", "dist", "band"]].to_csv(
    sys.argv[1] + ".coverage.csv", index=False
)
