#!/usr/bin/env python3
"""bnd_loadcmp.py A B [--cases C1,C2] [--skip SUBSTR,...]: per case of two coverage builds (COV/builds/<name>), how many
router ELFs have byte-identical loadable contents (every SHF_ALLOC section that has file bytes: name, address and
bytes; .bss and other NOBITS sections by name, address and size), and which differ. Used for V4_BOUNDS.md's identity
check (v4bdd at 0 against v4rxv2i3). Prints per case same / differ / only-in-one, then the first few differing ELFs."""
import collections, concurrent.futures, glob, os, struct, sys

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m") + "/builds"


def loadable(path):
    d = open(path, "rb").read()
    assert d[:4] == b"\x7fELF" and d[4] == 1, path  # ELF32
    e_shoff, = struct.unpack_from("<I", d, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", d, 0x2E)
    sh = [struct.unpack_from("<IIIIIIIIII", d, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    strtab = sh[e_shstrndx]
    names = d[strtab[4]:strtab[4] + strtab[5]]
    out = []
    for (nm, typ, flags, addr, off, size, *_rest) in sh:
        if not flags & 0x2:  # SHF_ALLOC
            continue
        name = names[nm:names.index(b"\0", nm)].decode()
        out.append((name, addr, size, None if typ == 8 else d[off:off + size]))  # 8 = SHT_NOBITS
    return sorted(out)


def main():
    a, b = sys.argv[1:3]
    cases = skip = None
    for i, x in enumerate(sys.argv):
        if x == "--cases":
            cases = set(sys.argv[i + 1].split(","))
        if x == "--skip":
            skip = sys.argv[i + 1].split(",")
    ea = {os.path.relpath(e, f"{COV}/{a}") for e in glob.glob(f"{COV}/{a}/*/*/*.elf")}
    eb = {os.path.relpath(e, f"{COV}/{b}") for e in glob.glob(f"{COV}/{b}/*/*/*.elf")}
    case = lambda rel: rel.split("__")[0]
    keep = lambda rel: (cases is None or case(rel) in cases) and not (skip and any(s in case(rel) for s in skip))
    both = sorted(r for r in ea & eb if keep(r))
    with concurrent.futures.ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda r: (r, loadable(f"{COV}/{a}/{r}") == loadable(f"{COV}/{b}/{r}")), both))
    c = collections.defaultdict(lambda: [0, 0, 0])
    diff = []
    for r, same in res:
        c[case(r)][0 if same else 1] += 1
        if not same:
            diff.append(r)
    for r in (ea ^ eb):
        if keep(r):
            c[case(r)][2] += 1
    tot = [0, 0, 0]
    for k in sorted(c):
        print(f"{k:28s} same {c[k][0]:4d}  differ {c[k][1]:4d}  only-in-one {c[k][2]:4d}")
        tot = [x + y for x, y in zip(tot, c[k])]
    print(f"{'TOTAL':28s} same {tot[0]:4d}  differ {tot[1]:4d}  only-in-one {tot[2]:4d}")
    for r in diff[:12]:
        print("  differs:", r)


if __name__ == "__main__":
    main()
