"""Print '<path relative to the JIT cache> <hash of loaded sections>' for every ELF under a JIT cache (scratch)."""
import hashlib, os, struct, sys
root = sys.argv[1]
for dp, _, fs in sorted(os.walk(root)):
    for fn in sorted(fs):
        if not fn.endswith(".elf"):
            continue
        p = os.path.join(dp, fn); d = open(p, "rb").read()
        try:
            shoff, = struct.unpack_from("<I", d, 0x20); se, sn, _ = struct.unpack_from("<HHH", d, 0x2E)
            h = hashlib.sha1()
            for i in range(sn):
                nm, typ, fl, addr, off, size = struct.unpack_from("<IIIIII", d, shoff + i * se)
                if fl & 2:
                    h.update(struct.pack("<II", addr, size))
                    if typ != 8:
                        h.update(d[off:off + size])
            print(os.path.relpath(p, root).split("/", 1)[-1], h.hexdigest()[:12])
        except struct.error:
            print(os.path.relpath(p, root), "bad")
