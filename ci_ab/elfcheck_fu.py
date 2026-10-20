#!/usr/bin/env python3
"""ELF guard for the fast-untilize arms: usage elfcheck_fu.py SIDE=<build dir> ... Keys every variant by its build.h (without UT_HEAD_API),
hashes the loadable sections. Requires: M2 == M on every common key; every D arm differs from H1 in the unpack ELF of some variant and is identical to H1 in
the pack ELF on every common key; H1 differs from M in the pack ELF."""
import glob, hashlib, struct, sys


def alloc_hash(path):
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


def cmp(a, b):
    ks = [k for k in E[a] if k in E[b]]
    same = {t: sum(E[a][k][t] == E[b][k][t] for k in ks) for t in ("unpack", "math", "pack")}
    print(f"ELF {a} vs {b}: {len(ks)} common, identical unpack/math/pack {same['unpack']}/{same['math']}/{same['pack']}")
    return ks, same


if "M2" in E:
    ks, same = cmp("M", "M2")
    ok &= len(ks) > 0 and all(v == len(ks) for v in same.values())
if "H1" in E and "M" in E:
    ks, same = cmp("M", "H1")
    ok &= len(ks) > 0 and same["pack"] < len(ks)
for h in E:
    if h in ("M", "M2", "H1"):
        continue
    ks, same = cmp("H1", h)
    ok &= len(ks) > 0 and same["unpack"] < len(ks) and same["pack"] == len(ks)
print("GUARD ok" if ok else "GUARD FAIL")
