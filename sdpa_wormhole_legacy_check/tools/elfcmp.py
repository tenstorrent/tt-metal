#!/usr/bin/env python3
"""Compare JIT kernel ELFs between two TT_METAL_CACHE roots (BASE vs HEAD).
Kernels are keyed by (kernel_name, hash). For every ELF we hash each SHF_ALLOC section
(NOBITS: size only). Also computes a config signature from the generated text files
(defines/compile-arg headers) so hash-mismatched kernels can be matched by config."""
import hashlib, os, struct, sys, collections

SHF_ALLOC = 2

def alloc_sections(path):
    d = open(path, "rb").read()
    assert d[:4] == b"\x7fELF" and d[4] == 1, path
    shoff, = struct.unpack_from("<I", d, 0x20)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x2E)
    secs = []
    for i in range(shnum):
        secs.append(struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize))
    strtab = secs[shstrndx]
    def nm(off):
        s = strtab[4] + off
        return d[s:d.index(b"\0", s)].decode()
    out = {}
    for s in secs:
        name, typ, flags, addr, off, size = s[:6]
        if not flags & SHF_ALLOC or size == 0:
            continue
        body = b"" if typ == 8 else d[off:off + size]
        out[nm(name)] = (addr, size, hashlib.sha1(body).hexdigest()[:12])
    return out

def scan(root):
    """returns {(kname, hash): {"elfs": {rel: secs}, "sig": str, "path": dir}}"""
    res = {}
    for dp, dns, fns in os.walk(root):
        parts = dp.split(os.sep)
        if "kernels" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("kernels")
        if len(parts) != i + 3:
            continue
        key = (parts[i + 1], parts[i + 2])
        elfs, sigh = {}, hashlib.sha1()
        for sub, _, files in sorted(os.walk(dp)):
            for f in sorted(files):
                p = os.path.join(sub, f); rel = os.path.relpath(p, dp)
                if f.endswith(".elf"):
                    if f.split(".")[0] + ".elf" != f and f.split(".")[0] + ".elf.xip.elf" != f:
                        continue  # in-flight temp file
                    try:
                        elfs[rel] = alloc_sections(p)
                    except Exception:
                        continue
                elif f.endswith((".h", ".hpp", ".cpp")) and "kernel_args" not in f:
                    sigh.update(rel.encode()); sigh.update(open(p, "rb").read())
        ent = {"elfs": elfs, "sig": sigh.hexdigest()[:12], "path": dp}
        if key in res and (res[key]["elfs"] != elfs or res[key]["sig"] != ent["sig"]):
            print(f"WARNING intra-tag mismatch for {key}: {res[key]['path']} vs {dp}")
        if key in res and len(res[key]["elfs"]) >= len(elfs):
            continue
        res[key] = ent
    return res

def cmp_elfs(a, b):
    diffs = []
    for rel in sorted(set(a) | set(b)):
        if rel not in a or rel not in b:
            diffs.append(f"{rel}: only in {'HEAD' if rel in b else 'BASE'}"); continue
        sa, sb = a[rel], b[rel]
        for s in sorted(set(sa) | set(sb)):
            if sa.get(s) != sb.get(s):
                diffs.append(f"{rel}:{s} BASE={sa.get(s)} HEAD={sb.get(s)}")
    return diffs

def main(base, head, filt=""):
    A, B = scan(base), scan(head)
    by_sig_B = collections.defaultdict(list)
    for k, v in B.items():
        by_sig_B[(k[0], v["sig"])].append(k)
    same = diff = 0
    rows = []
    for k in sorted(set(A) | set(B)):
        if filt and filt not in k[0]:
            continue
        if k in A and k in B:
            d = cmp_elfs(A[k]["elfs"], B[k]["elfs"])
            sigtag = "" if A[k]["sig"] == B[k]["sig"] else " (gen-headers differ)"
            n = sum(len(v) for v in A[k]["elfs"].values())
            if d:
                diff += 1; rows.append(f"DIFF  {k[0]}/{k[1]}{sigtag} [{len(A[k]['elfs'])} elfs]"); rows += ["      " + x for x in d]
            else:
                same += 1; rows.append(f"SAME  {k[0]}/{k[1]}{sigtag} [{len(A[k]['elfs'])} elfs, {n} alloc sections]")
        elif k in A:
            rows.append(f"ONLY-BASE {k[0]}/{k[1]}")
        else:
            if not any(k in v for v in by_sig_B.values() if False):
                pass
            rows.append(f"ONLY-HEAD {k[0]}/{k[1]}")
    print("\n".join(rows))
    print(f"SUMMARY same={same} diff={diff} base_kernels={len(A)} head_kernels={len(B)}")

if __name__ == "__main__":
    main(*sys.argv[1:])
