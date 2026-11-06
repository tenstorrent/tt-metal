#!/usr/bin/env python3
"""cov_summary.py BUILD_A BUILD_B: per case and processor of two cov.py router builds (same captures), how many router
images are identical by loadable bytes, the largest loadable image (text + data) in each build and the size change
range of the changed images; then the largest image per config class (1D/ring, 2D/torus/mux/VC2, VC1, 1-ERISC,
Galaxy) and processor. A build that failed (no ELF) is listed: an image that no longer fits fails its link."""
import collections, glob, os, re, struct, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune")
import elf_load_cmp as e


def load_size(p):
    b = open(p, "rb").read()
    phoff, = struct.unpack_from("<I", b, 28); pe, pn = struct.unpack_from("<HH", b, 42)
    segs = [struct.unpack_from("<8I", b, phoff + i * pe) for i in range(pn)]
    return sum(s[4] for s in segs if s[0] == 1)


def elfs(build):
    out = {}
    for d in glob.glob(os.path.join(build, "*__kernels__fabric_erisc_router__*")):
        name = os.path.basename(d)
        case, rest = name.split("__kernels__fabric_erisc_router__")
        proc = rest.split("__", 1)[1]
        f = [p for p in glob.glob(os.path.join(d, "*.elf")) if not p.endswith(".xip.elf")]
        out[name] = (case, proc, f[0] if f else None)
    return out


def klass(case):
    if case.startswith("glx"):
        return "Galaxy " + ("torus" if "torus" in case else "2D")
    if "1erisc" in case:
        return "1-ERISC (" + case.replace("bh_", "").replace("_1erisc", "") + ")"
    if "vc1" in case:
        return "VC1 split"
    if case in ("bh_1d", "bh_ring", "bh_neighbor"):
        return "1D / ring / neighbor"
    return "2D / torus / mux / VC2"


A, B = elfs(sys.argv[1]), elfs(sys.argv[2])
per = collections.defaultdict(lambda: collections.Counter())
sizes = collections.defaultdict(lambda: [0, 0, None, None])
cls = collections.defaultdict(lambda: [0, 0])
for k in sorted(set(A) | set(B)):
    ca, pa, fa = A.get(k, (None, None, None)); cb, pb, fb = B.get(k, (None, None, None))
    case, proc = ca or cb, pa or pb
    key = (case, proc)
    if not fa or not fb:
        per[key]["failed " + ("A" if not fa else "B")] += 1
        continue
    same = e.load_hash(fa) == e.load_hash(fb)
    sa, sb = load_size(fa), load_size(fb)
    per[key]["identical" if same else "changed"] += 1
    s = sizes[key]
    s[0], s[1] = max(s[0], sa), max(s[1], sb)
    if not same:
        s[2] = sb - sa if s[2] is None else min(s[2], sb - sa)
        s[3] = sb - sa if s[3] is None else max(s[3], sb - sa)
    c = cls[(klass(case), proc)]
    c[0], c[1] = max(c[0], sa), max(c[1], sb)
print("| case | processor | identical | changed | failed | largest A | largest B | changed: B - A |")
print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
for (case, proc), c in sorted(per.items()):
    s = sizes[(case, proc)]
    rng = f"{s[2]:+d} to {s[3]:+d} B" if s[2] is not None else "-"
    print(f"| {case} | {proc} | {c['identical']} | {c['changed']} | {c['failed A'] + c['failed B']} | {s[0]} | {s[1]} "
          f"| {rng} |")
print()
print("| config class | processor | largest A | largest B |")
print("| --- | --- | ---: | ---: |")
for (k, proc), (a, b) in sorted(cls.items()):
    print(f"| {k} | {proc} | {a} | {b} |")
