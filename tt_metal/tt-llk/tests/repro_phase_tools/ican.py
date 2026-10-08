#!/usr/bin/env python3
"""ican.py <run.cyc> <trisc>: icache L1 line reads of one TRISC over its hottest loop: lines, sets, reads per pass."""
import sys
from collections import Counter


def n(v):
    try:
        return int(v, 2)
    except:
        return None


t = sys.argv[2]
f = open(sys.argv[1])
keys = f.readline().split()
ix = {k: i for i, k in enumerate(keys)}
pcs = []
ic = []
mp = []
req = []
hit = []
mshr = []
for line in f:
    v = line.split()
    pcs.append(n(v[ix[f"t{t}_cmt_pc"]]))
    ic.append(n(v[ix[f"t{t}_ic_addr"]]) if v[ix[f"t{t}_ic_rden"]] == "1" else None)
    mp.append(v[ix[f"t{t}_mp"]] == "1")
    req.append(n(v[ix[f"t{t}_ic_req"]]))
    hit.append(n(v[ix[f"t{t}_ic_hit"]]))
    mshr.append(n(v[ix[f"t{t}_ic_mshr"]]))
c = Counter(p for i, p in enumerate(pcs) if p and (i == 0 or pcs[i - 1] != p))
hot = [p for p, k in c.items() if k >= 500]
head = min(hot)
starts = [i for i in range(1, len(pcs)) if pcs[i] == head and pcs[i - 1] != head]
a, b = starts[0], starts[-1]
npass = len(starts) - 1
lines = Counter(ic[i] * 16 for i in range(a, b) if ic[i] is not None)
nsets = 16 if t == "1" else 64
print(
    f"loop head {head:#x}, {npass} passes, cycles {b-a}, {(b-a)/npass:.1f} per pass, mispredicts {sum(mp[a:b])}"
)
print(
    f"icache L1 line reads in loop: {sum(lines.values())} ({sum(lines.values())/npass:.2f} per pass); counters req {req[b]-req[a]} hit {hit[b]-hit[a]} mshr {mshr[b]-mshr[a]}"
)
for ln, k in lines.most_common(12):
    print(f"  line {ln:#x} set {(ln>>4)%nsets:2d}: {k}")
hotlines = sorted({(p >> 4) << 4 for p in hot})
print(
    "hot code lines by set:",
    {
        s: [hex(l) for l in hotlines if (l >> 4) % nsets == s]
        for s in sorted({(l >> 4) % nsets for l in hotlines})
        if sum(1 for l in hotlines if (l >> 4) % nsets == s) > 2
    },
)
print("first pass start", a)
