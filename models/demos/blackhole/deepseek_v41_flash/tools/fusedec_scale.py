"""python3 -I fusedec_scale.py <tab.json> [trace M] -- extrapolate the layer-subset profile (layers 0,1,2,3,14,20,21,24) to the 40-layer round by layer-type multiplicity.
Types: L0 x1, L1 (+Engram1) x1, L2 type (CSA owner + indexer: layers 2, 8) x2, L14 (owner + indexer + Engram14) x1, L3 type (CSA2 reuse) x15, L20 x1, L21 type (ratio-1 readers) x15,
L24 type (index-only) x4. Fixed parts (start, embedding, head, accept_commit_draft, drafter write_main, draft) once; commit scales with the layer count (8 profiled layers -> 40).
Prints per-category ms of the round (mean chip; slowest chip), percent of the round, per-type layer us."""
import json
import os
import sys
from collections import defaultdict

res = json.load(open(sys.argv[1]))
tr = sys.argv[2] if len(sys.argv) > 2 else "M"
R = res[tr]
PC = R["phase_cats"]
mult = {"layer0": 1, "layer1": 1, "layer2": 2, "layer14": 1, "layer3": 15, "layer20": 1, "layer21": 15, "layer24": 4}
fixed = ["start", "embedding", "head", "accept_commit_draft", "drafter_write_main_full", "drafter_draft_full"]
eng = {"engram1": 1, "engram14": 1}
tot = defaultdict(lambda: [0.0, 0.0])
rows = []
for ph, m in list(mult.items()) + list(eng.items()) + [(p, 1) for p in fixed] + [("commit", 1)]:
    if ph not in PC:
        continue
    s = 0.0
    for c, (a, b) in PC[ph].items():
        mm = m * (1.5 if (ph == "commit" and c == "attention glue") else 1)
        tot[c][0] += a * mm
        tot[c][1] += b * mm
        s += a
    rows.append((ph, m, s))
T = sum(v[0] for v in tot.values())
print(f"extrapolated kernel round (mean chip) {T/1e3:.2f} ms")
for ph, m, s in rows:
    print(f"  {ph:26s} x{m:2d}  {s:8.1f} us   -> {s*m/1e3:6.2f} ms")
if "--subs" in sys.argv:
    want = sys.argv[sys.argv.index("--subs") + 1]
    PS = R["phase_subs"]
    sub = defaultdict(float)
    for ph, m in list(mult.items()) + list(eng.items()) + [(p, 1) for p in fixed] + [("commit", 1)]:
        for k, v in PS.get(ph, {}).items():
            if want == "ALL" or k.startswith(want):
                sub[k] += v * m
    print(f"sub-ops of '{want}' scaled to the round (us, mean chip):")
    for k, v in sorted(sub.items(), key=lambda kv: -kv[1])[: int(os.environ.get("TOPN", "40"))]:
        print(f"{v/1e3:9.1f}  {k[:140]}")
    sys.exit(0)
print("| category | ms (mean chip) | ms (slowest chip) | % |")
print("|---|---|---|---|")
for c, (a, b) in sorted(tot.items(), key=lambda kv: -kv[1][0]):
    print(f"| {c} | {a/1e3:.2f} | {b/1e3:.2f} | {100*a/T:.1f} |")
