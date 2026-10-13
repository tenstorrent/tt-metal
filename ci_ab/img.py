# Hash of the loadable sections of every trisc1 ELF under a kernel cache, order-independent.
import hashlib, os, struct, sys
hs = []
for dp, _, fn in os.walk(sys.argv[1]):
    for f in fn:
        if f != "trisc1.elf": continue
        b = open(os.path.join(dp, f), "rb").read(); h = hashlib.sha256(); shoff, = struct.unpack_from("<I", b, 0x20); se, sn = struct.unpack_from("<HH", b, 0x2E)
        for i in range(sn):
            _, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", b, shoff + i * se)
            if flags & 2 and typ != 8 and size: h.update(struct.pack("<II", addr, size)); h.update(b[off:off + size])
        hs.append(h.hexdigest())
print(hashlib.sha256("".join(sorted(hs)).encode()).hexdigest()[:12], f"({len(hs)} trisc1)")
