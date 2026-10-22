#!/usr/bin/env python3
"""Loaded-image identity: for ELFs paired as in elfcmp.py, compare the bytes of every SHF_ALLOC section with file contents
(PROGBITS: code and data) except .profiler_meta, plus section addresses. Pure-python ELF32 reader.
usage: elfload.py <build_a> <build_b>"""
import struct, sys, importlib.util
def sections(p):
    d = open(p, 'rb').read()
    shoff, = struct.unpack_from('<I', d, 0x20); shentsize, shnum, shstrndx = struct.unpack_from('<HHH', d, 0x2e)
    sh = [struct.unpack_from('<IIIIIIIIII', d, shoff + i * shentsize) for i in range(shnum)]
    strtab = sh[shstrndx]; names = d[strtab[4]:strtab[4] + strtab[5]]
    out = {}
    for name, typ, flags, addr, off, size, *_ in sh:
        n = names[name:names.index(b'\0', name)].decode()
        if typ == 1 and flags & 2 and n != '.profiler_meta':
            out[n] = (addr, d[off:off + size])
    return out
import os, hashlib
def variant_key(vdir):
    try: lines = [l for l in open(os.path.join(vdir, "build.h")).read().splitlines() if l.strip() not in ("#define UNPACK_BLOCK 0", "#define DATACOPY_PACK_BLOCK 1", "#define DATACOPY_TILES_BACK_TO_BACK 1")]
    except OSError: return os.path.basename(vdir)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]
def elfs(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.elf') and '/sources/' in d + '/':
                p = os.path.join(d, f); src = p.split('/sources/', 1)[1].split('/')[0]
                out[f"{src}/{variant_key(os.path.dirname(os.path.dirname(p)))}/{f}"] = p
    return out
a, b = elfs(sys.argv[1]), elfs(sys.argv[2]); same = diff = 0
for k in sorted(set(a) & set(b)):
    if sections(a[k]) == sections(b[k]): same += 1
    else:
        diff += 1
        if diff <= 5: print('differs', k)
print(f"# {same + diff} paired: {same} loaded images identical, {diff} differ")
