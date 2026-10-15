#!/usr/bin/env python3
"""Scratch, not for merge. For each compute_pool_2d and bilinear build present in both caches (paired by kernel name and
generated headers, as worddiff.py), the trisc2 disassembly of M against X with addresses and branch targets dropped, printed
as a unified diff ("DIS" lines, at most 80 per build) for the builds whose .text length differs, plus a count line for all.
usage: disasm_diff.py <cache_a> <cache_b>"""
import difflib
import glob
import hashlib
import os
import re
import subprocess
import sys
from collections import defaultdict

ca, cb = sys.argv[1:3]
OD = os.environ.get("RG_OBJDUMP") or "riscv-tt-elf-objdump"


def key(d):
    h = hashlib.sha256()
    for f in ("kernel_args_generated.h", "defines_generated.h", "chlkc_descriptors.h", "named_ct_arg_map_generated.h"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update("\n".join(l for l in open(p, errors="replace").read().splitlines()
                               if "FULL_KERNEL_NAME" not in l).encode())
    return h.hexdigest()[:16]


def builds(cache):
    out = {}
    for d in sorted(glob.glob(os.path.join(cache, "**/kernels/*/*"), recursive=True)):
        name = os.path.basename(os.path.dirname(d))
        e = os.path.join(d, "trisc2", "trisc2.elf")
        if name in ("compute_pool_2d", "bilinear") and os.path.exists(e):
            out.setdefault((name, key(d)), e)
    return out


def dis(elf):
    txt = subprocess.run([OD, "-d", "--no-show-raw-insn", "-j", ".text", elf], capture_output=True, text=True).stdout
    lines = []
    for m in re.finditer(r"^\s*[0-9a-f]+:\s+(.*)$", txt, re.M):
        s = re.sub(r"\s*<[^>]*>", "", m.group(1).split("#")[0]).strip()
        s = re.sub(r"^(j|jal|beq|bne|blt|bge|bltu|bgeu|beqz|bnez|blez|bgez|bltz|bgtz|bgt|ble|bgtu|bleu)(\s+.*?,?)\s*[0-9a-f]+$",
                   r"\1\2 T", s)
        lines.append(re.sub(r"\s+", " ", s))
    return lines


A, B = builds(ca), builds(cb)
for k in sorted(set(A) & set(B)):
    la, lb = dis(A[k]), dis(B[k])
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    ins = sum(j2 - j1 for t, i1, i2, j1, j2 in ops if t in ("insert", "replace"))
    dele = sum(i2 - i1 for t, i1, i2, j1, j2 in ops if t in ("delete", "replace"))
    print(f"DISCOUNT {k[0]} {k[1][:8]} M {len(la)} X {len(lb)} blocks {len(ops)} +{ins} -{dele}", flush=True)
    if len(la) != len(lb):
        for n, l in enumerate(difflib.unified_diff(la, lb, "M", "X", n=3, lineterm="")):
            if n >= 80:
                break
            print("DIS", k[1][:8], l, flush=True)
