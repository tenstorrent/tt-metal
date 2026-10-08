#!/usr/bin/env python3
"""Round 3 eltwise binary: ELFs of two JIT kernel caches (main, opt-in), compared per kernel name and ELF as sets. The cache
does not record compile-time arguments, so a kernel's variants cannot be paired by name and defines; instead every variant's
ELF is disassembled (addresses and branch targets masked) and hashed, and the multisets of hashes are compared: equal means
the opt-in changed none of that kernel's code. usage: elf_set_diff.py <cache A> <cache B> [kernel name regex]"""
import collections
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys

od = shutil.which("riscv-tt-elf-objdump") or next(
    (p for p in ("/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump", "/work/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump") if os.path.exists(p)),
    None,
)
if od is None:
    sys.exit("ELFSET no objdump")
pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None


def digest(elf):
    txt = subprocess.run([od, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout.splitlines()
    ins = [re.sub(r"\b[0-9a-f]{4,8}\b <[^>]*>", "T", m.group(1).split("#")[0]).strip() for m in (re.match(r"^\s*[0-9a-f]+:\s*(.*)$", l) for l in txt) if m]
    return hashlib.sha1("\n".join(ins).encode()).hexdigest()[:12]


def collect(root):
    out = collections.defaultdict(list)
    for elf in glob.glob(f"{root}/**/kernels/*/*/*/*.elf", recursive=True):
        if elf.endswith(".xip.elf"):
            continue
        parts = elf.split("/")
        name, rel = parts[-4], "/".join(parts[-2:])
        if pat and not pat.search(name):
            continue
        out[(name, rel)].append(digest(elf))
    return out


a, b = collect(sys.argv[1]), collect(sys.argv[2])
same = diff = 0
for key in sorted(set(a) | set(b)):
    ca, cb = collections.Counter(a.get(key, [])), collections.Counter(b.get(key, []))
    if ca == cb:
        same += 1
        print(f"ELFSET identical {key[0]} {key[1]}: {sum(ca.values())} variants")
    else:
        diff += 1
        print(f"ELFSET differ {key[0]} {key[1]}: main {sum(ca.values())} variants, opt-in {sum(cb.values())}, {sum((ca & cb).values())} in common")
print(f"ELFSET kernels x ELFs identical {same} differ {diff}")
