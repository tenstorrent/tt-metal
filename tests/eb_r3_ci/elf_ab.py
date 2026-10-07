#!/usr/bin/env python3
"""Round 3 eltwise binary (#58818 review): evidence that the two sides of an in-process dump ran different code. The compute
kernel variants of one JIT cache are grouped by their generated defines and descriptors with the toggle lines dropped; a group
with two or more variants is one comparison, and its TRISC ELFs are compared by the bytes of their executable sections.
usage: elf_ab.py <cache root> <name>=<regex of toggle lines> ..."""
import glob
import hashlib
import os
import re
import struct
import sys


def code_hash(path):
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF":
        return None
    shoff = struct.unpack_from("<I", b, 0x20)[0]
    shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    h = hashlib.sha1()
    for i in range(shnum):
        _, typ, flags, _, offset, size = struct.unpack_from("<IIIIII", b, shoff + i * shentsize)
        if typ == 1 and flags & 0x4:
            h.update(b[offset : offset + size])
    return h.hexdigest()


root = sys.argv[1]
for spec in sys.argv[2:]:
    name, _, rx = spec.partition("=")
    pat = re.compile(rx)
    groups = {}
    for d in glob.glob(f"{root}/**/kernels/{name}/*/", recursive=True):
        defs = ""
        for f in ("defines_generated.h", "chlkc_descriptors.h", "kernel_args_generated.h", "named_args_generated.h"):
            if os.path.exists(d + f):
                defs += open(d + f, errors="replace").read()
        key = "\n".join(l for l in defs.splitlines() if not pat.search(l))
        elfs = {}
        for e in glob.glob(d + "**/*.elf", recursive=True):
            t = os.path.basename(e)
            if t.startswith("trisc"):
                elfs[t] = code_hash(e)
        groups.setdefault(key, []).append(elfs)
    multi = [g for g in groups.values() if len(g) > 1]
    diff = {}
    for t in ("trisc0.elf", "trisc1.elf", "trisc2.elf"):
        diff[t] = sum(1 for g in multi if len({e.get(t) for e in g}) > 1)
    print(
        f"DUMP elfab {name} [{rx}]: variants {sum(len(g) for g in groups.values())}, comparisons (groups of 2+) {len(multi)}, "
        f"code differs in unpack {diff['trisc0.elf']}, math {diff['trisc1.elf']}, pack {diff['trisc2.elf']}",
        flush=True,
    )
