#!/usr/bin/env python3
"""Scratch, not for merge. Code hashes of the JIT kernels in a TT_METAL_CACHE: for every kernels/<name>/<hash>/ build
whose name matches a filter, the disassembly hash (addresses kept, symbols dropped) of each processor's ELF and a key
from the generated headers (pairs the same build across arms). Prints "ELF <tag> <name> <key> <proc>=<hash> ...".
usage: elfhash.py <tag> <cache_dir> [name_substring...]"""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys

tag, cache = sys.argv[1], sys.argv[2]
filt = sys.argv[3:]
od = next((p for p in ("/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump", shutil.which("riscv-tt-elf-objdump") or "")
           if p and os.path.exists(p)), None)
if od is None:
    sys.exit("no riscv-tt-elf-objdump")


def code_hash(elf):
    txt = subprocess.run([od, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    lines = [m.group(1) + " " + re.sub(r"\s*<[^>]*>", "", m.group(2).split("#")[0]).strip()
             for m in re.finditer(r"^\s*([0-9a-f]+):\s+(.*)$", txt, re.M)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16] + "/%d" % len(lines)


def key(d):
    h = hashlib.sha256()
    for f in ("kernel_args_generated.h", "defines_generated.h", "chlkc_descriptors.h", "named_ct_arg_map_generated.h"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update("\n".join(l for l in open(p, errors="replace").read().splitlines()
                               if "FULL_KERNEL_NAME" not in l).encode())
    return h.hexdigest()[:16]


n = 0
for d in sorted(glob.glob(os.path.join(cache, "**/kernels/*/*"), recursive=True)):
    name = os.path.basename(os.path.dirname(d))
    if not os.path.isdir(d) or (filt and not any(f in name for f in filt)):
        continue
    elfs = sorted(glob.glob(os.path.join(d, "*", "*.elf")))
    if not elfs:
        continue
    hs = [f"{os.path.basename(os.path.dirname(e))}={code_hash(e)}" for e in elfs]
    print("ELF", tag, name, key(d), *hs, flush=True)
    n += 1
print("ELFCOUNT", tag, n, flush=True)
