# Per kernel (the JIT cache's kernels/<name>/ directory), the set of loadable-section hashes of its ELFs; compares two
# caches. usage: kelf.py <cacheA> <cacheB> [labelA labelB]
import hashlib, os, struct, sys
from collections import defaultdict


def img(fn):
    b = open(fn, "rb").read()
    h = hashlib.sha256()
    shoff, = struct.unpack_from("<I", b, 0x20)
    se, sn = struct.unpack_from("<HH", b, 0x2E)
    secs = []
    for i in range(sn):
        _, typ, flags, addr, off, size = struct.unpack_from("<IIIIII", b, shoff + i * se)
        if flags & 2 and typ != 8 and size:
            secs.append((addr, b[off : off + size]))
    for a, d in sorted(secs):
        h.update(struct.pack("<II", a, len(d)))
        h.update(d)
    return h.hexdigest()[:16]


def load(root):
    k = defaultdict(set)
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".elf"):
                continue
            parts = dp.split(os.sep)
            if "kernels" not in parts:
                continue
            name = parts[parts.index("kernels") + 1]
            k[name].add(f"{f}:{img(os.path.join(dp, f))}")
    return k


a, b = load(sys.argv[1]), load(sys.argv[2])
la, lb = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("A", "B")
same, diff = [], []
for n in sorted(set(a) | set(b)):
    (same if a.get(n) == b.get(n) else diff).append(n)
print(f"KELF {la} vs {lb}: {len(same)} kernels identical, {len(diff)} differ")
for n in diff:
    print(f"KELF differ {n}: {la} {len(a.get(n, ()))} ELFs, {lb} {len(b.get(n, ()))} ELFs, {len(a.get(n, set()) & b.get(n, set()))} shared")
for n in same:
    print(f"KELF same {n} ({len(a[n])} ELFs)")
