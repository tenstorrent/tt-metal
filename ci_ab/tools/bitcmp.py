# Byte comparison of two bit dump directories (bitdump_plugin.py): per module, equal / differ / only in one side.
# usage: bitcmp.py <dir a> <dir b> [--list]
import collections
import os
import sys

a, b = sys.argv[1], sys.argv[2]


def index(d):
    m = {}
    p = os.path.join(d, "index.tsv")
    if os.path.exists(p):
        for line in open(p):
            h, nid = line.rstrip("\n").split("\t", 1)
            m[h] = nid
    return m


ia, ib = index(a), index(b)
fa = {f for f in os.listdir(a) if f.endswith(".bin")}
fb = {f for f in os.listdir(b) if f.endswith(".bin")}
res = collections.Counter()
diff = []
for f in sorted(fa | fb):
    nid = ia.get(f.split("__")[0]) or ib.get(f.split("__")[0]) or "?"
    mod = nid.split("::")[0]
    if f not in fa or f not in fb:
        res[(mod, "one side")] += 1
        continue
    same = open(os.path.join(a, f), "rb").read() == open(os.path.join(b, f), "rb").read()
    res[(mod, "equal" if same else "differ")] += 1
    if not same:
        diff.append(nid)
for (mod, k), n in sorted(res.items()):
    print(f"BITS {mod} {k} {n}")
tot = collections.Counter()
for (mod, k), n in res.items():
    tot[k] += n
print("BITS total " + " ".join(f"{k} {n}" for k, n in sorted(tot.items())))
for nid in sorted(set(diff))[: (10000 if "--list" in sys.argv else 20)]:
    print("BITS differ", nid[:300])
