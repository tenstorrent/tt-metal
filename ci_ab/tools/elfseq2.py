#!/usr/bin/env python3
"""Per (node, run type): do two labels' TRISC ELFs have the same instruction sequence (profiler zone-id lui immediates
masked)? usage: elfseq2.py <slot A dir> <slot B dir> <img A tsv> <img B tsv> <hash A> <hash B> [builds subdir]"""
import os, re, subprocess, sys, collections
OBJ = os.environ.get("OBJD", "/localdev/mvlahovic/llk_analysis_builds/sfpi_7.87.0/sfpi/compiler/bin/riscv-tt-elf-objdump")
def seq(fn):
    out = subprocess.run([OBJ, "-d", "-j", ".text", fn], capture_output=True, text=True).stdout.splitlines()
    r = []
    for l in out:
        m = re.match(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(.*)$", l)
        if not m: continue
        ins = re.sub(r"#.*$", "", re.sub(r"<[^>]*>", "", m.group(1))).strip()
        ins = re.sub(r"^(lui\s+\w+,)0x[0-9a-f]+$", r"\1Z", ins)
        ins = re.sub(r"\b[0-9a-f]{4,8}\b", "A", ins) if ins.split()[0] in ("j", "jal", "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz") else ins
        r.append(ins)
    return r
def idx(slot, bd):
    root = f"{slot}/{bd}/tt-llk-build/sources"; d = {}
    for s in os.listdir(root):
        for h in os.listdir(f"{root}/{s}"): d[h] = f"{root}/{s}/{h}"
    return d
def nodes(hashlog):
    m = collections.defaultdict(set)
    for l in open(hashlog):
        n, h = l.rstrip("\n").split("\t"); m[n].add(h)
    return m
A, B, hA, hB = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]; bd = sys.argv[5] if len(sys.argv) > 5 else "builds"
iA, iB = idx(A, bd), idx(B, bd); nA, nB = nodes(hA), nodes(hB)
def rt(d): m = re.search(r"PERF_RUN_TYPE\s*=\s*PerfRunType::(\w+)", open(d + "/build.h").read()); return m.group(1) if m else "-"
cnt = collections.Counter()
for n in sorted(set(nA) & set(nB)):
    for ha in nA[n]:
        da = iA.get(ha)
        if not da or not os.path.exists(da + "/build.h"): continue
        r = rt(da)
        db = next((iB[hb] for hb in nB[n] if hb in iB and os.path.exists(iB[hb] + "/build.h") and rt(iB[hb]) == r), None)
        if not db: continue
        diff = [t for t in ("unpack", "math", "pack") if seq(f"{da}/elf/{t}.elf") != seq(f"{db}/elf/{t}.elf")]
        key = "same" if not diff else "differ:" + ",".join(diff)
        cnt[key] += 1
        print(f"{n}\t{r}\t{key}")
print("SUMMARY", dict(cnt))
