#!/usr/bin/env python3
"""Scratch, not for merge. elfhash.py plus sizes and firmware: for every kernels/<name>/<hash>/ build in a TT_METAL_CACHE
whose name matches a filter, per processor ELF the disassembly hash (addresses kept, symbols dropped), the instruction
count and the packed size (sum of the PT_LOAD file sizes rounded to words, what ll_api::memory loads into the kernel
config ring), and a key from the generated headers. Then the same for every firmware ELF of the cache.
Prints "ELF <tag> <name> <key> <proc>=<hash>/<ninstr>/<packed> ..." and "FW <tag> <relpath> <hash>/<ninstr>/<packed>".
usage: elfhash2.py <tag> <cache_dir> [name_substring...]"""
import glob
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys

tag, cache = sys.argv[1], sys.argv[2]
filt = sys.argv[3:]
od = next((p for p in ("/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump", shutil.which("riscv-tt-elf-objdump") or "")
           if p and os.path.exists(p)), None)
if od is None:
    sys.exit("no riscv-tt-elf-objdump")


def packed(elf):
    b = open(elf, "rb").read()
    if b[:4] != b"\x7fELF" or b[4] != 1:
        return -1
    phoff, = struct.unpack_from("<I", b, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    n = 0
    for i in range(phnum):
        p_type, _, _, _, p_filesz = struct.unpack_from("<IIIII", b, phoff + i * phentsize)
        if p_type == 1:
            n += (p_filesz + 3) // 4 * 4
    return n


def code_hash(elf):
    txt = subprocess.run([od, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    lines = [m.group(1) + " " + re.sub(r"\s*<[^>]*>", "", m.group(2).split("#")[0]).strip()
             for m in re.finditer(r"^\s*([0-9a-f]+):\s+(.*)$", txt, re.M)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16] + "/%d/%d" % (len(lines), packed(elf))


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
fw = sorted(glob.glob(os.path.join(cache, "**/firmware/**/*.elf"), recursive=True))
for e in fw:
    print("FW", tag, re.sub(r"^.*/firmware/", "firmware/", e), code_hash(e), flush=True)
print("FWCOUNT", tag, len(fw), flush=True)
