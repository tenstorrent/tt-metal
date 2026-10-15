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


rows = {}
for b in sys.argv[1:]:
    for vd in sorted(glob.glob(f"{b}/tt-llk-build/sources/matmul_pack_untilize_perf.cpp/*/")):
        bh = open(f"{vd}/build.h").read() if os.path.exists(f"{vd}/build.h") else ""
        mi = re.search(r"PACK_UNTILIZE_INIT\s*=\s*(\d+)", bh)
        rt = re.search(r"PERF_RUN_TYPE\s*=\s*PerfRunType::(\w+)", bh)
        if not (mi and rt):
            continue
        hs = {t: image(f"{vd}/elf/{t}.elf") if os.path.exists(f"{vd}/elf/{t}.elf") else "-" for t in ("unpack", "math", "pack")}
        rows[(rt.group(1), int(mi.group(1)))] = hs
        print("ELF", rt.group(1), "init", mi.group(1), *[f"{t}={h}" for t, h in hs.items()])
ok = True
for rt in sorted({k[0] for k in rows}):
    packs = [rows[(rt, i)]["pack"] for i in (0, 1, 2) if (rt, i) in rows]
    if len(packs) != 3 or len(set(packs)) != 3 or "-" in packs:
        ok = False
print("GUARD", "ok" if ok and rows else "FAIL", "pack ELFs of the three init forms differ in every run type; builds:", len(rows))
sys.exit(0 if ok and rows else 1)
