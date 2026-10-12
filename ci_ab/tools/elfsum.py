"""Swap control: hashes of the loaded sections of each matmul compute kernel's three TRISC ELFs in a JIT cache.
usage: elfsum.py <variant> <cache dir>"""
import glob, hashlib, struct, sys
def image(p):
    d = open(p, "rb").read()
    shoff, = struct.unpack_from("<I", d, 0x20); shentsize, shnum, _ = struct.unpack_from("<HHH", d, 0x2E)
    h = hashlib.sha1()
    for i in range(shnum):
        nm, typ, flags, addr, off, size = struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize)[:6]
        if flags & 2:
            h.update(struct.pack("<II", addr, size))
            if typ != 8: h.update(d[off:off + size])
    return h.hexdigest()[:8]
v, c = sys.argv[1:3]
out = {}
for e in glob.glob(f"{c}/**/kernels/*bmm_large_block_zm_fused_bias_activation*/*/trisc*/trisc*.elf", recursive=True):
    parts = e.split("/")
    out.setdefault(parts[-4], set()).add(parts[-2] + ":" + image(e))
for k in sorted(out):
    per = {}
    for x in out[k]:
        t, h = x.split(":"); per.setdefault(t, []).append(h)
    print("RAE", v, k, " ".join(f"{t}:{len(hs)}:{hashlib.sha1(' '.join(sorted(hs)).encode()).hexdigest()[:8]}" for t, hs in sorted(per.items())))
