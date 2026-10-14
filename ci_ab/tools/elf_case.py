# One call of a prof_rand case in a fresh JIT cache, then the loadable-image hash of every compute (trisc) kernel ELF.
# usage: python elf_case.py <case> (TT_METAL_CACHE set by the caller)
import hashlib, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttnn
import prof_div as p


def img(fn):
    b = open(fn, "rb").read(); h = hashlib.sha256(); shoff, = struct.unpack_from("<I", b, 0x20); se, sn = struct.unpack_from("<HH", b, 0x2E)
    for i in range(sn):
        _, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", b, shoff + i * se)
        if flags & 2 and typ != 8 and size: h.update(struct.pack("<II", addr, size)); h.update(b[off:off + size])
    return h.hexdigest()[:16]


dev = ttnn.open_device(device_id=0)
try:
    setup, run = p.CASES[sys.argv[1]]
    run(dev, *setup(dev)); ttnn.synchronize_device(dev)
finally:
    ttnn.close_device(dev)
out = []
for d, _, fs in os.walk(os.environ["TT_METAL_CACHE"]):
    for f in fs:
        if f.endswith(".elf") and "/kernels/" in d and os.path.basename(d).startswith("trisc") and not f.endswith("_weakened.elf"):
            out.append(f"{d.split('/kernels/')[1].split('/')[0]}/{os.path.basename(d)}={img(os.path.join(d, f))}")
print("ELF", sys.argv[1], " ".join(sorted(out)))
