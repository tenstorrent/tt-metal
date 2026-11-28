#!/usr/bin/env python3
"""Scratch, not for merge. elfpair.py <cacheA> <cacheB> <tag>: pairs the JIT ELFs of two caches by path below the build key dir
(kernels/<name>/<hash>/<proc>/<proc>.elf, firmware/<proc>/<proc>.elf) and compares the bytes of their SHF_ALLOC PROGBITS sections."""
import glob, hashlib, os, struct, sys

def alloc_hash(p):
    b = open(p, "rb").read()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        return None
    shoff, = struct.unpack_from("<I", b, 0x20); shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    h = hashlib.md5()
    for i in range(shnum):
        _, typ, flags, _, off, size = struct.unpack_from("<IIIIII", b, shoff + i * shentsize)
        if typ == 1 and flags & 2:
            h.update(b[off:off + size])
    return h.hexdigest()[:12]

def load(c):
    d = {}
    for p in glob.glob(f"{c}/*/kernels/*/*/*/*.elf") + glob.glob(f"{c}/*/firmware/*/*.elf"):
        if p.endswith(".xip.elf"):
            continue
        d[os.path.relpath(p, c).split("/", 1)[1]] = alloc_hash(p)
    return d

A, B = load(sys.argv[1]), load(sys.argv[2]); tag = sys.argv[3]
k = sorted(set(A) & set(B)); diff = [x for x in k if A[x] != B[x]]
tr = [x for x in k if "/trisc" in x]; trd = [x for x in diff if "/trisc" in x]
fw = [x for x in diff if x.startswith("firmware/")]
print(f"CIAB ELFPAIR {tag} pairs {len(k)} differ {len(diff)} trisc_pairs {len(tr)} trisc_differ {len(trd)} fw_differ {len(fw)} onlyA {len(set(A)-set(B))} onlyB {len(set(B)-set(A))}")
def byname(d):  # kernels/<name>/<hash>/<proc>/x.elf -> {(name, proc): sorted hashes}, for caches whose hash dirs differ
    r = {}
    for x, h in d.items():
        q = x.split("/")
        r.setdefault((q[1], q[3]) if q[0] == "kernels" else (q[0], q[1]), []).append(h or "")
    return {k: sorted(v) for k, v in r.items()}
NA, NB = byname(A), byname(B); nk = sorted(set(NA) & set(NB)); nd = [x for x in nk if NA[x] != NB[x]]
print(f"CIAB ELFNAME {tag} names {len(nk)} differ {len(nd)} trisc_names {sum('trisc' in x[1] for x in nk)} trisc_differ {sum('trisc' in x[1] for x in nd)}")
for x in diff[:int(os.environ.get("ELFPAIR_SHOW", "0"))]:
    print(f"CIAB ELFDIFF {tag} {x} {A[x]} {B[x]}")
