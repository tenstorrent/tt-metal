#!/usr/bin/env python3
"""ELF guard: usage elfcheck.py SIDE=<build dir> ...  Keys every variant by its build.h without the UT_HEAD_API line
(so a config has one key on every side), hashes the loadable sections of the unpack/math/pack ELFs. Requires: M2 == M on every common key;
H1 != M and H2 != M on the pack ELF of every untilize_allcfg variant. Prints GUARD ok or GUARD FAIL."""
import glob, hashlib, struct, sys


def alloc_hash(path):
    """sha1 of the loadable sections (SHF_ALLOC, not NOBITS) of an ELF32: debug info carries build paths."""
    b = open(path, "rb").read()
    shoff, = struct.unpack_from("<I", b, 0x20); shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    sec = [struct.unpack_from("<IIIIIIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    h = hashlib.sha1()
    for name, typ, flags, addr, off, size, *_ in sec:
        if flags & 2 and typ != 8:
            h.update(struct.pack("<II", addr, size)); h.update(b[off:off + size])
    return h.hexdigest()[:12]

def scan(d):
    out = {}
    for b in glob.glob(f"{d}/tt-llk-build/sources/*/*/build.h"):
        v = b.rsplit("/", 1)[0]; src = v.split("/")[-2]
        txt = "".join(l for l in open(b) if "UT_HEAD_API" not in l)
        k = src + ":" + hashlib.sha1(txt.encode()).hexdigest()[:16]
        h = {}
        for t in ("unpack", "math", "pack"):
            try:
                h[t] = alloc_hash(f"{v}/elf/{t}.elf")
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
for h in E:
    if h in ("M", "M2"):
        continue
    if h.startswith("U"):
        ks, same = cmp("H1", h, "untilize_allcfg_perf.cpp")
        ok &= len(ks) > 0 and same["unpack"] < len(ks)
        continue
    ks, same = cmp("M", h, "untilize_allcfg_perf.cpp")
    ok &= len(ks) > 0 and same["pack"] == 0
    if h != "H1" and "H1" in E:
        ks, same = cmp("H1", h, "untilize_allcfg_perf.cpp")
        ok &= len(ks) > 0 and same["pack"] == 0
print("GUARD ok" if ok else "GUARD FAIL")
