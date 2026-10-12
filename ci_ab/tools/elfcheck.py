"""Swap control: per variant build dir, a hash of each thread's loaded ELF sections for the block configurations.
usage: elfcheck.py <variant>=<build dir>..."""
import glob
import hashlib
import os
import re
import struct
import sys


def image(p):
    d = open(p, "rb").read()
    shoff, = struct.unpack_from("<I", d, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x2E)
    sh = [struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize) for i in range(shnum)]
    h = hashlib.sha1()
    for s in sh:
        nm, typ, flags, addr, off, size = s[:6]
        if flags & 2:
            h.update(struct.pack("<II", addr, size))
            if typ != 8:
                h.update(d[off:off + size])
    return h.hexdigest()[:10]


for arg in sys.argv[1:]:
    v, b = arg.split("=", 1)
    rows = []
    for vd in sorted(glob.glob(f"{b}/tt-llk-build/sources/pack_block_contiguous_perf.cpp/*/")):
        bh = open(f"{vd}/build.h").read() if os.path.exists(f"{vd}/build.h") else ""
        if not re.search(r"PACK_BLOCK_CONTIGUOUS\s*=\s*true", bh):
            continue
        fr = re.search(r"TEST_FACE_R_DIM\s*=\s*(\d+)", bh)
        nf = re.search(r"num_faces\s*=\s*(\d+)", bh)
        key = f"r{fr.group(1) if fr else '?'}f{nf.group(1) if nf else '?'}"
        hs = "/".join(image(f"{vd}/elf/{t}.elf") if os.path.exists(f"{vd}/elf/{t}.elf") else "-" for t in ("unpack", "math", "pack"))
        rows.append(f"{key}:{hs}")
    print("ELF", v, " ".join(rows))
