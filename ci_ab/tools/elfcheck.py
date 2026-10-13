"""Per variant build dir: a hash of each thread's loaded ELF sections for every built variant.
usage: elfcheck.py <variant>=<build dir>...   prints 'ELF <variant> <key>=<u>/<m>/<p>' lines and a summary."""
import glob, hashlib, struct, sys

def image(p):
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

res = {}
for arg in sys.argv[1:]:
    v, b = arg.split("=", 1)
    m = {}
    for vd in sorted(glob.glob(f"{b}/tt-llk-build/sources/*/*/")):
        key = vd.rstrip("/").split("/")[-2] + "/" + vd.rstrip("/").split("/")[-1][:10]
        m[key] = tuple(image(f"{vd}elf/{t}.elf") if glob.glob(f"{vd}elf/{t}.elf") else "-" for t in ("unpack", "math", "pack"))
    res[v] = m
    for k, hs in m.items():
        print("ELF", v, k, "/".join(hs))
base = res[sys.argv[1].split("=")[0]]
for v, m in res.items():
    keys = sorted(set(base) & set(m))
    diff = [sum(base[k][t] != m[k][t] for k in keys) for t in range(3)]
    print(f"ELFSUM {v} dirs={len(m)} common={len(keys)} differ_from_first unpack={diff[0]} math={diff[1]} pack={diff[2]}")
