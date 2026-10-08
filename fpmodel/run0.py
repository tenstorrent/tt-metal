import sys

sys.path.insert(0, ".")
from data import *
from model import *

d = load(list(SETS))
pred = np.zeros(len(d))
for a in ("wh", "bh"):
    m = (d.arch_ == a).to_numpy()
    g = geometry(d[m])
    pred[m] = predict(g, PARAMS)
evaluate(d, pred, "first-principles, unfitted")
print("pair acc, median |log err|, p90:", rank_quality(d, pred))
