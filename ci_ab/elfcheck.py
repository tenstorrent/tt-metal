#!/usr/bin/env python3
"""ELF guard: usage elfcheck.py SIDE=<build dir> ...  Keys every variant by its build.h without the UT_HEAD_API line
(so a config has one key on every side), hashes unpack/math/pack ELFs. Requires: M2 == M on every common key;
H1 != M and H2 != M on the pack ELF of every untilize_allcfg variant. Prints GUARD ok or GUARD FAIL."""
import glob, hashlib, sys

def scan(d):
    out = {}
    for b in glob.glob(f"{d}/tt-llk-build/sources/*/*/build.h"):
        v = b.rsplit("/", 1)[0]; src = v.split("/")[-2]
        txt = "".join(l for l in open(b) if "UT_HEAD_API" not in l)
        k = src + ":" + hashlib.sha1(txt.encode()).hexdigest()[:16]
        h = {}
        for t in ("unpack", "math", "pack"):
            try:
                h[t] = hashlib.sha1(open(f"{v}/elf/{t}.elf", "rb").read()).hexdigest()[:12]
            except OSError:
                h[t] = None
        out[k] = h
    return out

S = dict(a.split("=", 1) for a in sys.argv[1:]); E = {s: scan(d) for s, d in S.items()}
ok = True
for s, e in E.items():
    print(f"ELF {s}: {len(e)} variants")
def cmp(a, b, src=None):
    ks = [k for k in E[a] if k in E[b] and (src is None or k.startswith(src))]
    same = {t: sum(E[a][k][t] == E[b][k][t] for k in ks) for t in ("unpack", "math", "pack")}
    print(f"ELF {a} vs {b} {src or 'all'}: {len(ks)} common, identical unpack/math/pack {same['unpack']}/{same['math']}/{same['pack']}")
    return ks, same
if "M2" in E:
    ks, same = cmp("M", "M2")
    ok &= len(ks) > 0 and all(v == len(ks) for v in same.values())
for h in ("H1", "H2"):
    if h in E:
        ks, same = cmp("M", h, "untilize_allcfg_perf.cpp")
        ok &= len(ks) > 0 and same["pack"] == 0
        cmp("M", h, "fast_untilize_test.cpp")
print("GUARD ok" if ok else "GUARD FAIL")
