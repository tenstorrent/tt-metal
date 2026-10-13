#!/usr/bin/env python3
"""Print "<hash> <relative path>" for every */elf/*.elf under a build dir, hashing only the loaded image (SHF_ALLOC
sections: name, address and bytes; NOBITS by size), so build directory names in the debug sections do not count."""
import hashlib, os, struct, sys

def image(path):
    b = open(path, "rb").read()
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", b, 0x2E)
    hdrs = [struct.unpack_from("<IIIIIIIIII", b, shoff + i * shentsize) for i in range(shnum)]
    stro = hdrs[shstrndx][4]
    h = hashlib.sha256()
    for name_off, typ, flags, addr, off, size, *_ in hdrs:
        if not flags & 0x2:
            continue
        name = b[stro + name_off: b.index(b"\0", stro + name_off)]
        h.update(name + struct.pack("<III", typ, addr, size))
        if typ != 8:
            h.update(b[off:off + size])
    return h.hexdigest()[:16]

root = sys.argv[1]
for dp, _, fn in sorted(os.walk(root)):
    for f in sorted(fn):
        if f.endswith(".elf") and os.path.basename(dp) == "elf":
            p = os.path.join(dp, f)
            print(image(p), "./" + os.path.relpath(p, root))
