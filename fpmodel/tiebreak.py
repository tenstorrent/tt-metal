"""Uncertainty-aware pick: among candidates predicted within m of the best, take the shallowest K block
(then the predicted best). Margin chosen on WH, checked on BH."""
import sys, os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, ".")
from data import *

d = load(list(SETS))
pred = np.load(sys.argv[1] if len(sys.argv) > 1 else "pred_cv.npy")
d["pred"] = pred


def score(m, key):
    s = np.empty(len(d))
    for p, g in d.groupby("problem_id", sort=False):
        c = g.origin.isin(CAND).to_numpy()
        best = g.pred[c].min()
        near = (g.pred <= best * (1 + m)).to_numpy() & c
        k = key(g)
        # rank: within the near set by key then pred; others after
        s[g.index] = np.where(near, k * 1e6 + g.pred / best, 1e12 + g.pred)
    return s


keys = {"kb": lambda g: g.in0_block_w.fillna(1).to_numpy()}
for m in (0.0, 0.03, 0.05, 0.08, 0.12):
    for kn, kf in keys.items():
        evaluate(d, score(m, kf), f"margin {m:.2f} prefer low {kn}")
