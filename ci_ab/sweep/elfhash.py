"""ELF guard for the reduce all-config A/B. Per side build dir, a hash of each thread's loaded ELF sections per variant
(keyed by the hash of its build.h, which holds every template parameter). Prints one compact line per variant and a
GUARD line: ok when main and head differ on at least one variant of every module that the PR changes (the swap reached
the compiler) and the two sides built the same variant set. usage: elfhash.py main=<dir> head=<dir>"""
import glob
import hashlib
import os
import struct
import sys


def image(p):
    if not os.path.exists(p):
        return "-"
    d = open(p, "rb").read()
    shoff, = struct.unpack_from("<I", d, 0x20)
    shentsize, shnum, _ = struct.unpack_from("<HHH", d, 0x2E)
    h = hashlib.sha1()
    for i in range(shnum):
        nm, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", d, shoff + i * shentsize)
        if flags & 2:
            h.update(struct.pack("<II", addr, size))
            if typ != 8:
                h.update(d[off:off + size])
    return h.hexdigest()[:10]


sides = dict(a.split("=", 1) for a in sys.argv[1:])
tab = {}
for side, b in sides.items():
    for vd in glob.glob(f"{b}/tt-llk-build/sources/*/*/"):
        src = vd.rstrip("/").split("/")[-2]
        bh = f"{vd}/build.h"
        if not os.path.exists(bh):
            continue
        key = hashlib.sha1(open(bh, "rb").read()).hexdigest()[:12]
        tab[(src, key, side)] = tuple(image(f"{vd}/elf/{t}.elf") for t in ("unpack", "math", "pack"))
keys = sorted({(s, k) for s, k, _ in tab})
diff = {}
for s, k in keys:
    m, h = tab.get((s, k, "main")), tab.get((s, k, "head"))
    print("ELF", s, k, "main", *(m or ("-",) * 3), "head", *(h or ("-",) * 3))
    st = diff.setdefault(s, [0, 0, 0])
    if m is None or h is None:
        st[2] += 1
    elif m == h:
        st[0] += 1
    else:
        st[1] += 1
ok = bool(keys)
for s, (same, dif, miss) in sorted(diff.items()):
    print("SRC", s, "identical", same, "different", dif, "missing", miss)
    ok = ok and miss == 0
ok = ok and diff.get("reduce_allcfg_perf.cpp", [0, 0, 0])[1] > 0
print("GUARD", "ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
