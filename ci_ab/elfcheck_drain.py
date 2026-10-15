"""Variant ELF guard for the #60463 drain A/B: per variant build dir, a hash of each thread's loaded ELF sections per
(run type, init form). Guard ok when A != B on the pack ELF of every (run type, form) (the drain reached the compiler),
A == A2 and B == B2 (copies build alike). usage: elfcheck_drain.py A=<dir> B=<dir> A2=<dir> B2=<dir>"""
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
for a in sys.argv[1:]:
    v, b = a.split("=", 1)
    for vd in sorted(glob.glob(f"{b}/tt-llk-build/sources/matmul_pack_untilize_perf.cpp/*/")):
        bh = open(f"{vd}/build.h").read() if os.path.exists(f"{vd}/build.h") else ""
        mi = re.search(r"PACK_UNTILIZE_INIT\s*=\s*(\d+)", bh)
        rt = re.search(r"PERF_RUN_TYPE\s*=\s*PerfRunType::(\w+)", bh)
        if not (mi and rt):
            continue
        hs = {t: image(f"{vd}/elf/{t}.elf") if os.path.exists(f"{vd}/elf/{t}.elf") else "-" for t in ("unpack", "math", "pack")}
        rows[(v, rt.group(1), int(mi.group(1)))] = hs
        print("ELF", v, rt.group(1), "init", mi.group(1), *[f"{t}={h}" for t, h in hs.items()])
ok = True
keys = sorted({(k[1], k[2]) for k in rows})
for rt, i in keys:
    g = lambda v: rows.get((v, rt, i), {}).get("pack", "-")
    a, b, a2, b2 = g("A"), g("B"), g("A2"), g("B2")
    row_ok = "-" not in (a, b, a2, b2) and a != b and a == a2 and b == b2
    print("PAIR", rt, "init", i, f"A={a} B={b} A2={a2} B2={b2}", "ok" if row_ok else "FAIL")
    ok = ok and row_ok
print("GUARD", "ok" if ok and keys else "FAIL", "pack ELF A != B, A == A2, B == B2 for every run type and form; pairs:", len(keys))
sys.exit(0 if ok and keys else 1)
