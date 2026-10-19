"""ELF guard for a multi-side A/B: per side build dir, hash each variant's loaded ELF sections (keyed by build.h); print
identical/different counts for consecutive side pairs. GUARD ok when every side built the same variant set and every pair
named in MUSTDIFF (env, "a:b a:b") differs on some variant. usage: elfguard.py side=<dir> ..."""
import glob, hashlib, os, struct, sys


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
names = list(sides)
tab = {}
for side, b in sides.items():
    for vd in glob.glob(f"{b}/tt-llk-build/sources/*/*/"):
        if os.path.exists(f"{vd}/build.h"):
            key = (vd.rstrip("/").split("/")[-2], hashlib.sha1(open(f"{vd}/build.h", "rb").read()).hexdigest()[:12])
            tab.setdefault(key, {})[side] = tuple(image(f"{vd}/elf/{t}.elf") for t in ("unpack", "math", "pack"))
ok = all(len(v) == len(sides) for v in tab.values())
for k, v in sorted(tab.items()):
    print("ELF", k[0], k[1], *[f"{s}={'/'.join(v.get(s, ('-',)))}" for s in names])
pairs = list(zip(names, names[1:]))
must = [tuple(p.split(":")) for p in os.environ.get("MUSTDIFF", "").split()]
for a, b in sorted(set(pairs) | set(must)):
    ks = [k for k in tab if a in tab[k] and b in tab[k]]
    same = sum(tab[k][a] == tab[k][b] for k in ks)
    print("PAIR", a, b, "identical", same, "different", len(ks) - same)
    if (a, b) in must:
        ok = ok and same < len(ks)
print("GUARD", "ok" if ok else "FAIL", "variants", len(tab))
sys.exit(0 if ok else 1)
