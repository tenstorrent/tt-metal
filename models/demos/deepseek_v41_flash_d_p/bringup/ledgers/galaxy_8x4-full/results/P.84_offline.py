"""P.84 offline: static replicas, one extra expert slot per chip (32 a layer), chosen on code + agentic-pick sessions,
scored on agentic-held-out sessions (and prose / multilingual / code) with DS41-299's fit
(per layer 205 us + 6.77 us x the busiest chip's 32-row tiles). Home placement read from the census (chips[].experts)."""
import ast
import collections
import glob
import json
import math
import sys

CEN = "/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/census"
SES = json.load(open("/scratch/dnijemcevic/bringup/p84_census/sessions.json"))
PICK_S = {s["file"] for i, s in enumerate(SES) if i % 2 == 0}  # sessions 1, 3, 5, 7
def session(p):
    return next(s["file"] for s in SES if s["start"] <= p < s["start"] + s["n"])
def load(path):
    d = json.load(open(path)); out = []
    for ch in d["chunks"]:
        layers = []
        for l in ch["layers"]:
            chips = l["chips"] if isinstance(l["chips"], list) else ast.literal_eval(l["chips"])
            layers.append({e: c for chip in chips for e, c in zip(chip["experts"], chip["counts"])})
        out.append((ch["start"], layers))
    return d, out
HOME = None
def home_map(path):
    d = json.load(open(path)); l = d["chunks"][0]["layers"][0]
    chips = l["chips"] if isinstance(l["chips"], list) else ast.literal_eval(l["chips"])
    return {e: i for i, chip in enumerate(chips) for e in chip["experts"]}
agentic = {}
for f in sorted(glob.glob(f"{CEN}/census_p*_k4_tagentic.json")):
    if "p145408" in f: continue  # X.9's run (sessions 2/3 boundary): kept out of both halves
    d, chunks = load(f)
    for start, layers in chunks: agentic[start] = layers
HOME = home_map(glob.glob(f"{CEN}/census_p*_k4_tagentic.json")[0])
C = 32; L = len(next(iter(agentic.values())))
pick = [v for p, v in agentic.items() if session(p) in PICK_S]
held = [v for p, v in agentic.items() if session(p) not in PICK_S]
other = {t: [v for _, v in load(f"{CEN}/census_p145408_k4_t{t}.json")[1]] for t in ("code", "prose", "multilingual", "math")}
def tiles(c): return math.ceil(c / 32)
def chunk_ms(layers, reps):
    tot = 0.0
    for li, rows in enumerate(layers):
        load_ = [0] * C
        rep = reps[li] if reps else {}
        for e, c in rows.items():
            if e not in rep: load_[HOME[e]] += tiles(c)
        for e in sorted(rep, key=lambda e: -rows.get(e, 0)):  # best split of each replicated expert's rows
            c = rows.get(e, 0); h, r = HOME[e], rep[e]
            best = min(range(0, c + 1, 32) if c else [0], key=lambda a: max(load_[h] + tiles(a), load_[r] + tiles(c - a)))
            load_[h] += tiles(best); load_[r] += tiles(c - best)
        tot += 205 + 6.77 * max(load_)
    return tot / 1000
def choose(train):
    reps = []
    for li in range(L):
        tot = collections.Counter()
        for layers in train: tot.update(layers[li])
        hot = [e for e, _ in tot.most_common(4 * C)]
        chip_load = [0.0] * C
        for e, v in tot.items(): chip_load[HOME[e]] += v
        free = set(range(C)); rep = {}
        for e in hot:  # each chip one replica slot; the hottest go to the least-loaded chips
            if not free: break
            cand = [c for c in free if c != HOME[e]]
            if not cand: continue
            r = min(cand, key=lambda c: chip_load[c])
            rep[e] = r; free.discard(r); chip_load[r] += tot[e] / 2; chip_load[HOME[e]] -= tot[e] / 2
        reps.append(rep)
    return reps
reps = choose(other["code"] + pick)
def mean(xs): return sum(xs) / len(xs)
print(f"agentic chunks: pick {len(pick)} ({len(PICK_S)} sessions), held-out {len(held)}; replicas 32 a layer (13 experts a chip)")
for name, chunks in [("agentic held-out", held), ("agentic pick", pick)] + [(t, v) for t, v in other.items()]:
    a, b = mean([chunk_ms(c, None) for c in chunks]), mean([chunk_ms(c, reps) for c in chunks])
    print(f"{name:18s} experts fit {a:5.1f} -> {b:5.1f} ms a chunk ({b - a:+.1f})")
print(f"replica weights: 1 expert x 3 x 5120 x 2304 x 0.53 B = {3*5120*2304*0.53/2**20:.1f} MiB a layer a chip; "
      f"{20*3*5120*2304*0.53/2**30:.2f} GiB a chip for layers 0-19, {40*3*5120*2304*0.53/2**30:.2f} GiB for 40")
json.dump({"chosen_on": ["code p145408 k4"] + sorted(f"agentic {s}" for s in PICK_S), "held_out": sorted(s["file"] for s in SES if s["file"] not in PICK_S),
           "rule": "per layer the 32 experts with the most rows on the picking chunks; one replica a chip (13th slot), never on the expert's home chip; hottest to the least-loaded chips",
           "layers": [{str(e): r for e, r in rep.items()} for rep in reps]}, open(sys.argv[1], "w"), indent=0)
