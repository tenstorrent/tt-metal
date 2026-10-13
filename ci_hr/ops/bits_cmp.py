#!/usr/bin/env python3
"""Compare eb_bits_plugin hash files across runs: per test, the set of output hashes of each variant (all repetitions).
usage: bits_cmp.py <dir with hash_<i>_<variant>.json> <base variant>"""
import glob, json, os, sys
d, base = sys.argv[1], sys.argv[2]
per = {}
for f in sorted(glob.glob(os.path.join(d, "hash_*.json"))):
    v = os.path.basename(f)[:-5].split("_", 2)[2]
    h = json.load(open(f))["hashes"]
    for t, hs in h.items():
        per.setdefault(t, {}).setdefault(v, set()).update(hs)
nb = ne = nd = 0
for t in sorted(per):
    vs = per[t]
    if base not in vs:
        continue
    for v, hs in vs.items():
        if v == base:
            continue
        nb += 1
        if hs == vs[base] and len(hs) > 0:
            ne += 1
        else:
            nd += 1
            print(f"DIFF {v} {t.split('::')[-1]}: base {len(vs[base])} hashes, {v} {len(hs)}, common {len(hs & vs[base])}")
    if len(vs[base]) > len({h.split(':')[0] for h in vs[base]}):
        pass
print(f"bits: {ne} of {nb} (test, variant) pairs identical to {base}, {nd} differ; tests {len(per)}")
for t in sorted(per):
    for v, hs in per[t].items():
        uniq = {h for h in hs}
        if v == base and len(uniq) > 1 and len(uniq) != len({h for h in uniq}):
            print(f"note {t}: {len(uniq)} distinct outputs within {v}")
