import json, sys, numpy as np

spans = json.load(open(sys.argv[1]))
XS = [1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15]
NY = 10


def phys(i):
    return f"{XS[i//NY]},{i%NY+2}"


def lpt(weights, ncores):
    order = sorted(range(len(weights)), key=lambda a: -weights[a])  # stable
    load = [0] * ncores
    items = [[] for _ in range(ncores)]
    for it in order:
        c = min(range(ncores), key=lambda k: (load[k], k))
        items[c].append(it)
        load[c] += weights[it]
    return items, load


def schedule(chunk, wfun, St=32, heads_per_group=4, streams=2, ncores=110, L=14):
    per = ncores // streams
    extra = ncores % streams
    first = 0
    cores = []
    for s in range(streams):
        sc = per + (1 if s < extra else 0)
        rows = []
        w = []
        for h in range(heads_per_group):
            for q in range(St):
                rows.append(q)
                w.append(wfun(q))
        items, load = lpt(w, sc)
        for c in range(sc):
            pos = c % L
            cores.append(
                dict(
                    idx=first + c,
                    inj=(pos == L - 1) or (c + 1 == sc),
                    sink=(pos == 0),
                    rows=[rows[i] for i in items[c]],
                    load=load[c],
                )
            )
        first += sc
    return cores


chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 8
wcur = lambda q: ((q + 1 + chunk - 1) // chunk) * chunk + 1
cores = schedule(chunk, wcur)
rounds = max(len(c["rows"]) for c in cores)
name = sys.argv[2]
X = []
y = []
for c in cores:
    sp = spans[name].get(phys(c["idx"]))
    if sp is None:
        continue
    tiles = sum(q + 1 for q in c["rows"])
    chunks = sum((q + 1 + chunk - 1) // chunk for q in c["rows"])
    X.append([1, len(c["rows"]), tiles, chunks, int(c["inj"]), int(c["sink"])])
    y.append(sp)
X = np.array(X, float)
y = np.array(y)
coef, res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print(name, "chunk", chunk, "rounds", rounds, "cores", len(y))
print(
    "fit: const %.1f + rows %.1f + tiles %.2f + chunks %.2f + inj %.1f + sink %.1f  (rmse %.1f us)"
    % (*coef, np.sqrt(np.mean((pred - y) ** 2)))
)
print("measured max %.1f median %.1f ; predicted max %.1f" % (y.max(), np.median(y), pred.max()))


# what LPT with fitted per-row weights would give (same injector/sink positions): minimize max predicted
def predict(cores):
    out = []
    for c in cores:
        tiles = sum(q + 1 for q in c["rows"])
        chunks = sum((q + 1 + chunk - 1) // chunk for q in c["rows"])
        out.append(
            coef[0]
            + coef[1] * len(c["rows"])
            + coef[2] * tiles
            + coef[3] * chunks
            + coef[4] * c["inj"]
            + coef[5] * c["sink"]
        )
    return np.array(out)


for label, wf in [
    ("current", wcur),
    ("tiles", lambda q: q + 1),
    ("fitted", lambda q: int(round(100 * (coef[1] + coef[2] * (q + 1) + coef[3] * ((q + 1 + chunk - 1) // chunk))))),
]:
    cs = schedule(chunk, wf)
    p = predict(cs)
    print(
        "  LPT weights=%s: predicted max %.1f median %.1f rounds %d"
        % (label, p.max(), np.median(p), max(len(c["rows"]) for c in cs))
    )
