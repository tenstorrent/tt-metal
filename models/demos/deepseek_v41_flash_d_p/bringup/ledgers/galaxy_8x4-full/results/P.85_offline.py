import collections
import glob
import json
import math
import sys

D = "/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/census_sessions"
plan = {(s["text"], s["start"]): s for s in json.load(open("/scratch/dnijemcevic/bringup/p85/sessions.json"))}
L84 = "/home/dnijemcevic/repos/tt-metal/models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/results/P.84_replicas.json"
C, HOME, recs = 32, {}, []
for f in glob.glob(f"{D}/*.json"):
    r = json.load(open(f)); s = plan.get((r["text"], r["start"]))
    if not s: continue
    chunks = []
    for ch in r["chunks"]:
        layers = []
        for l in ch["layers"]:
            if not HOME: HOME.update({e: i for i, chip in enumerate(l["chips"]) for e in chip["experts"]})
            layers.append({e: c for chip in l["chips"] for e, c in zip(chip["experts"], chip["counts"])})
        chunks.append(layers)
    recs.append((s, chunks))
L = 20
tiles = lambda c: math.ceil(c / 32)
def chunk_ms(layers, reps, even=False):
    tot = 0.0
    for li, rows in enumerate(layers):
        ld = [0] * C; rep = reps[li] if reps else {}
        for e, c in rows.items():
            if e not in rep: ld[HOME[e]] += tiles(c)
        for e in sorted(rep, key=lambda e: -rows.get(e, 0)):
            c = rows.get(e, 0); h, r = HOME[e], rep[e]
            a = c // 2 if even else min(range(0, c + 1, 32) if c else [0], key=lambda a: max(ld[h] + tiles(a), ld[r] + tiles(c - a)))
            ld[h] += tiles(a); ld[r] += tiles(c - a)
        tot += 205 + 6.77 * max(ld)
    return tot / 1000
def choose(train):
    reps = []
    for li in range(L):
        tot = collections.Counter()
        for layers in train: tot.update(layers[li])
        ld = [0.0] * C
        for e, v in tot.items(): ld[HOME[e]] += v
        free, rep = set(range(C)), {}
        for e, _ in tot.most_common(4 * C):
            if not free: break
            cand = [c for c in free if c != HOME[e]]
            if not cand: continue
            r = min(cand, key=lambda c: ld[c]); rep[e] = r; free.discard(r); ld[r] += tot[e] / 2; ld[HOME[e]] -= tot[e] / 2
        reps.append(rep)
    return reps
groups = collections.defaultdict(list)
for s, ch in recs: groups[(s["role"], s["group"])] += ch
pick = [c for (role, g), v in groups.items() if role == "pick" for c in v]
new = choose(pick)
old = [{int(e): r for e, r in l.items()} for l in json.load(open(L84))["layers"]]
re84 = choose([c for s, ch in recs if s.get("role84") == "pick" for c in ch])  # P.84's inputs, real history
mean = lambda xs: sum(xs) / len(xs)
print(f"sessions {len(recs)}, chunks {sum(len(c) for _, c in recs)}; pick chunks {len(pick)}")
print(f"{'group':30s} {'chunks':>6s} {'none':>6s} {'P.84':>6s} {'P.84re':>7s} {'new':>6s} {'new even':>9s}")
for k, v in sorted(groups.items()):
    a, b, r, c, d = (mean([chunk_ms(x, t) for x in v]) for t in (None, old, re84, new)), None, None, None, None
    a, b, r, c = list(a); d = mean([chunk_ms(x, new, even=True) for x in v])
    print(f"{k[0] + '/' + k[1]:30s} {len(v):6d} {a:6.1f} {b:6.1f} {r:7.1f} {c:6.1f} {d:9.1f}")
ov = mean([len(set(old[li]) & set(re84[li])) / 32 for li in range(L)])
ovn = mean([len(set(old[li]) & set(new[li])) / 32 for li in range(L)])
print(f"replica-set overlap per layer: P.84 table vs P.84 re-picked on real history {ov:.2f}; P.84 vs new {ovn:.2f}")
if len(sys.argv) > 1:
    json.dump({"chosen_on": "pick repos of agentic_cc + agentic_ds, real-history session census (P.85)", "layers": [{str(e): r for e, r in rep.items()} for rep in new]}, open(sys.argv[1], "w"), indent=0)
