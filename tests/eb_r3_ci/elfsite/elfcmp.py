#!/usr/bin/env python3
"""Compare the kernel ELFs of two JIT caches. Each build is keyed by (kernel, hash of its generated headers: defines,
descriptors and the named compile-time argument map, minus lines matching --drop) and the thread file; the value is a hash of
the loaded image (PROGBITS sections with SHF_ALLOC). Keys compare as multisets of image hashes.
usage: elfcmp.py <cache A> <cache B> [--drop REGEX] [--kernel REGEX] [--list]"""
import collections
import glob
import hashlib
import os
import re
import struct
import sys

GEN = ("defines_generated.h", "chlkc_descriptors.h", "named_ct_arg_map_generated.h", "named_args_generated.h")


def image_hash(path):
    b = open(path, "rb").read()
    if b[:4] != b"\x7fELF":
        return None
    shoff = struct.unpack_from("<I", b, 0x20)[0]
    shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    h = hashlib.sha1()
    for i in range(shnum):
        _, typ, flags, _, offset, size = struct.unpack_from("<IIIIII", b, shoff + i * shentsize)
        if typ == 1 and flags & 0x2:
            h.update(b[offset : offset + size])
    return h.hexdigest()[:20]


def load(root, drop, kre):
    d = collections.defaultdict(collections.Counter)
    paths = {}
    for p in glob.glob(os.path.join(root, "*", "kernels", "*", "*", "*", "*.elf")):
        if p.endswith(".xip.elf"):
            continue
        parts = p.split(os.sep)
        kname, kdir = parts[-4], os.sep.join(parts[:-2])
        if kre and not re.search(kre, kname):
            continue
        key = hashlib.sha1()
        for g in GEN:
            gp = os.path.join(kdir, g)
            if os.path.exists(gp):
                for line in open(gp, "rb").read().splitlines():
                    if drop and re.search(drop.encode(), line):
                        continue
                    key.update(line + b"\n")
        k = (kname, key.hexdigest()[:16], parts[-1])
        h = image_hash(p)
        d[k][h] += 1
        paths.setdefault((k, h), p)
    return d, paths


def main():
    a, b = sys.argv[1], sys.argv[2]
    drop = sys.argv[sys.argv.index("--drop") + 1] if "--drop" in sys.argv else None
    kre = sys.argv[sys.argv.index("--kernel") + 1] if "--kernel" in sys.argv else None
    da, pa = load(a, drop, kre)
    db, pb = load(b, drop, kre)
    same = diff = oa = ob = 0
    per = collections.Counter()
    tot = collections.Counter()
    for k in sorted(set(da) | set(db)):
        tot[(k[0], k[2])] += 1
        if k not in db:
            oa += 1
            continue
        if k not in da:
            ob += 1
            continue
        if da[k] == db[k]:
            same += 1
        else:
            diff += 1
            per[(k[0], k[2])] += 1
            if "--list" in sys.argv:
                print("DIFF", k, list(da[k]), list(db[k]), pa.get((k, next(iter(da[k])))), pb.get((k, next(iter(db[k])))))
    print(f"keys: identical {same}, differing {diff}, only in A {oa}, only in B {ob}")
    for (kn, th), n in sorted(tot.items()):
        print(f"  {kn} {th}: {per[(kn, th)]} of {n} builds differ")


if __name__ == "__main__":
    main()
