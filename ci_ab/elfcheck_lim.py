#!/usr/bin/env python3
"""Guard: MAINP2 (M2) == MAINP on every common key; LIMK and HEADP pack ELFs of untilize variants differ from MAINP."""
import sys
sys.argv, args = sys.argv[:1], sys.argv[1:]
import importlib.util, os
spec = importlib.util.spec_from_file_location("ec", os.path.join(os.path.dirname(os.path.abspath(__file__)), "elfcheck_arms.py"))
src = open(spec.origin).read().split("S = dict(")[0]
ns = {}; exec(src, ns)
E = {a.split("=")[0]: ns["scan"](a.split("=")[1]) for a in args}
ok = True
for s, e in E.items(): print(f"ELF {s}: {len(e)} variants")
def cmp(a, b):
    ks = [k for k in E[a] if k in E[b]]
    same = {t: sum(E[a][k][t] == E[b][k][t] for k in ks) for t in ("unpack", "math", "pack")}
    print(f"ELF {a} vs {b}: {len(ks)} common, identical unpack/math/pack {same['unpack']}/{same['math']}/{same['pack']}")
    return ks, same
ks, same = cmp("MAINP", "M2"); ok &= len(ks) > 0 and all(v == len(ks) for v in same.values())
for h in ("HEADP", "LIMK", "KDF"):
    ks, same = cmp("MAINP", h); ok &= len(ks) > 0 and same["pack"] < len(ks)
print("GUARD ok" if ok else "GUARD FAIL")
