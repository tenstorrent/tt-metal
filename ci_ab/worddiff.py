#!/usr/bin/env python3
"""Scratch, not for merge. Word level .text comparison of the JIT kernel ELFs of two caches. Builds are paired by kernel name and
a key from the generated headers (as elfhash.py). For each ELF pair: the word count, the differing words, and whether every
differing word is a Tensix MOP instruction on both sides that differs only in its low 16 bits (the loop count field).
Prints "WORDS <name> <key> <proc> <words> <ndiff> <identical|mop-only|LEN|OTHER> [index:a->b ...]" and a "WORDS SUM" line.
usage: worddiff.py <cache_a> <cache_b> <tag_a> <tag_b>"""
import glob
import hashlib
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

ca, cb, ta, tb = sys.argv[1:5]
OBJ = os.environ.get("RG_OBJCOPY") or "riscv-tt-elf-objcopy"


def key(d):
    h = hashlib.sha256()
    for f in ("kernel_args_generated.h", "defines_generated.h", "chlkc_descriptors.h", "named_ct_arg_map_generated.h"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update("\n".join(l for l in open(p, errors="replace").read().splitlines()
                               if "FULL_KERNEL_NAME" not in l).encode())
    return h.hexdigest()[:16]


def builds(cache):
    out = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(cache, "**/kernels/*/*"), recursive=True)):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*", "*.elf")):
            out[(os.path.basename(os.path.dirname(d)), key(d))].append(d)
    return out


def text(elf):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        subprocess.run([OBJ, "-O", "binary", "-j", ".text", elf, t.name], check=True, capture_output=True)
        return open(t.name, "rb").read()


def tt(w):  # a .ttinsn word back to the Tensix encoding
    return ((w >> 2) | ((w & 3) << 30)) & 0xFFFFFFFF


A, B = builds(ca), builds(cb)
cnt = defaultdict(int)
for k in sorted(set(A) | set(B)):
    if k not in A or k not in B:
        print("WORDS NOPAIR", k[0], k[1][:8], "only in", ta if k in A else tb)
        cnt["nopair"] += 1
        continue
    for da, db in zip(A[k], B[k]):
        for ea in sorted(glob.glob(os.path.join(da, "*", "*.elf"))):
            proc = os.path.basename(os.path.dirname(ea))
            eb = os.path.join(db, proc, os.path.basename(ea))
            if not os.path.exists(eb):
                print("WORDS NOELF", k[0], k[1][:8], proc)
                cnt["noelf"] += 1
                continue
            xa, xb = text(ea), text(eb)
            n = len(xa) // 4
            if len(xa) != len(xb):
                print("WORDS", k[0], k[1][:8], proc, n, len(xb) // 4, "LEN")
                cnt["len"] += 1
                continue
            wa = [int.from_bytes(xa[i:i + 4], "little") for i in range(0, len(xa), 4)]
            wb = [int.from_bytes(xb[i:i + 4], "little") for i in range(0, len(xb), 4)]
            d = [(i, a, b) for i, (a, b) in enumerate(zip(wa, wb)) if a != b]
            if not d:
                cnt["identical"] += 1
                continue
            mop = all(tt(a) >> 24 == 1 and tt(b) >> 24 == 1 and (tt(a) ^ tt(b)) >> 16 == 0 for _, a, b in d)
            cnt["mop-only" if mop else "OTHER"] += 1
            det = " ".join(f"{i}:{tt(a):08x}->{tt(b):08x}" for i, a, b in d[:6])
            print("WORDS", k[0], k[1][:8], proc, n, len(d), "mop-only" if mop else "OTHER", det, flush=True)
print("WORDS SUM", ta, tb, " ".join(f"{k}={v}" for k, v in sorted(cnt.items())), flush=True)
