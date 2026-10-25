# Loop-range instruction counts keyed by router source text (so line shifts between variants don't matter).
# usage: loopsrc.py hash variant... (first variant is the base)
import sys, re
sys.path.insert(0, "/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad")
from linemap import parse
from loop79 import loop
from collections import Counter
h, V = sys.argv[1], sys.argv[2:]
rows = {}
for v in V:
    lines = open(f"fabric_exp/variants/{v}/fabric_erisc_router_v2.cpp").read().split("\n")
    elf = f"fabric_exp/variants/{v}/elf/{h}/active_erisc.elf"
    body = loop(elf); a, b = body[0][0], body[-1][0]
    c = Counter()
    for addr, op, args, (f, ln) in parse(elf):
        if a <= addr <= b:
            key = lines[ln - 1].strip()[:80] if f == "fabric_erisc_router_v2.cpp" else f"{f}:{ln}"
            c[key] += 1
    rows[v] = c
keys = set().union(*rows.values())
print("  ".join(f"{v:>5s}" for v in V))
for k in sorted(keys, key=lambda k: -max(abs(rows[v][k] - rows[V[0]][k]) for v in V)):
    if len(set(rows[v][k] for v in V)) > 1:
        print("  ".join(f"{rows[v][k]:5d}" for v in V), "|", k)
