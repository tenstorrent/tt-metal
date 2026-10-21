#!/usr/bin/env python3
# node, run type and the loadable-image hash of each TRISC ELF of a harness build. usage: ra_imgs.py <build dir> <hash log>
import hashlib, os, re, struct, sys
root = sys.argv[1] + "/tt-llk-build/sources"; idx = {}
for s in os.listdir(root):
    for h in os.listdir(f"{root}/{s}"): idx[h] = f"{root}/{s}/{h}"
def img(fn):
    b = open(fn, "rb").read(); h = hashlib.sha256(); shoff, = struct.unpack_from("<I", b, 0x20); se, sn = struct.unpack_from("<HH", b, 0x2E); secs = []
    for i in range(sn):
        _, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", b, shoff + i * se)
        if flags & 2 and typ != 8 and size: secs.append((addr, b[off:off + size]))
    for a, d in sorted(secs): h.update(struct.pack("<II", a, len(d))); h.update(d)
    return h.hexdigest()[:16]
seen = set()
for l in open(sys.argv[2]):
    n, h = l.rstrip("\n").split("\t"); d = idx.get(h)
    if not d or (n, h) in seen or not os.path.exists(d + "/build.h"): continue
    seen.add((n, h)); m = re.search(r"PERF_RUN_TYPE\s*=\s*PerfRunType::(\w+)", open(d + "/build.h").read())
    imgs = " ".join(f"{e[:-4]}={img(d + '/elf/' + e)}" for e in sorted(os.listdir(d + "/elf")))
    print(f"{n}\t{m.group(1) if m else '?'}\t{imgs}")
