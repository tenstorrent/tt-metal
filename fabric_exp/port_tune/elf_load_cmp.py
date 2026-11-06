#!/usr/bin/env python3
"""elf_load_cmp.py CACHE_A CACHE_B KERNEL_DIR...: per kernel directory (fabric_erisc_router, tt_fabric_test_sender, ...)
and processor, the multiset of loadable images (PT_LOAD segments: address, sizes and bytes) of every JIT-built ELF in
two kernel caches, and whether the two multisets are equal. Exit 1 when the first KERNEL_DIR's differ."""
import collections
import glob
import hashlib
import struct
import sys


def load_hash(path):
    b = open(path, "rb").read()
    assert b[:4] == b"\x7fELF" and b[4] == 1, path
    phoff, = struct.unpack_from("<I", b, 28)
    phentsize, phnum = struct.unpack_from("<HH", b, 42)
    h = hashlib.sha256()
    for i in range(phnum):
        t, off, va, _pa, fsz, msz, _fl, _al = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if t == 1:
            h.update(struct.pack("<III", va, fsz, msz))
            h.update(b[off:off + fsz])
    return h.hexdigest()


def collect(cache, kind):
    d = collections.defaultdict(list)
    for p in glob.glob(f"{cache}/**/kernels/{kind}/**/*.elf", recursive=True):
        if p.endswith("_weakened.elf") or p.endswith(".xip.elf"):
            continue
        proc = p.split(f"/kernels/{kind}/")[1].split("/")[1]
        d[proc].append(load_hash(p))
    return {k: sorted(v) for k, v in d.items()}


def main():
    a, b, kinds = sys.argv[1], sys.argv[2], sys.argv[3:]
    first_equal = True
    for n, kind in enumerate(kinds):
        ca, cb = collect(a, kind), collect(b, kind)
        for proc in sorted(set(ca) | set(cb)):
            x, y = ca.get(proc, []), cb.get(proc, [])
            eq = x == y and len(x) > 0
            print(f"{kind:24s} {proc:26s} ELFs {len(x)}/{len(y)}, distinct {len(set(x))}/{len(set(y))}: "
                  f"{'identical' if eq else 'DIFFERENT'}")
            if n == 0 and not eq:
                first_equal = False
        if not ca and not cb:
            print(f"{kind:24s} no ELFs")
            if n == 0:
                first_equal = False
    sys.exit(0 if first_equal else 1)


if __name__ == "__main__":
    main()
