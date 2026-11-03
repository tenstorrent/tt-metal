#!/usr/bin/env python3
"""funccmp.py BASE_BUILD NEW_BUILD CMP_TXT: for every image bytecmp listed as differing, the functions whose code differs
(instruction text with addresses normalised to symbol+offset), grouped per case: which functions changed, in how many
images. Loop functions (receiver::run, sender::run, single::run) vs init (kernel_main and the init helpers)."""
import collections, concurrent.futures as cf, re, subprocess, sys
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
base, new, cmp_txt = sys.argv[1:4]


def funcs(elf):
    out = subprocess.run([B + "objdump", "-d", "-C", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    res, cur = {}, None
    for l in out.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.*)>:$", l)
        if m:
            name = m.group(2)
            if name.startswith(".L"):
                continue  # local labels continue the current function
            cur = name
            res[cur] = []
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$", l)
        if m and cur:
            ops = re.sub(r"#.*", "", m.group(2))
            ops = re.sub(r"\b[0-9a-f]{3,8} <([^>]*)>", r"<\1>", ops)  # branch/jump targets: symbol+offset
            ops = re.sub(r"<\.L[^>]*>", "<L>", ops)
            res[cur].append(m.group(1) + " " + ops.strip())
    return res


def short(n):
    n = re.sub(r"\(.*", "", n)
    for k in ("receiver::run", "sender::run", "single::run", "kernel_main"):
        if k in n:
            return k
    return re.sub(r"<.*", "", n).split("::")[-1][:40]


def one(rel):
    a = funcs(f"{CB}/{base}/{rel}")
    b = funcs(f"{CB}/{new}/{rel}")
    changed = set()
    for n in set(a) | set(b):
        if a.get(n) != b.get(n):
            changed.add(short(n))
    return rel.split("__")[0], tuple(sorted(changed))


import json, os
def text_bytes(elf):
    return subprocess.run([B + "objcopy", "-O", "binary", "-j", ".text", elf, "/dev/stdout"], capture_output=True).stdout
def differs(rel):
    return rel if text_bytes(f"{CB}/{base}/{rel}") != text_bytes(f"{CB}/{new}/{rel}") else None
ra = json.load(open(f"{CB}/{base}/results.json")); rb = json.load(open(f"{CB}/{new}/results.json"))
keys = [k for k in rb if rb[k]["ok"] and k in ra and ra[k]["ok"] and (cmp_txt == "all" or k.startswith(cmp_txt))]
cands = []
for k in keys:
    d = f"{CB}/{new}/{k}"
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".elf"):
                cands.append(os.path.relpath(os.path.join(dp, f), f"{CB}/{new}"))
with cf.ProcessPoolExecutor(16) as ex:
    rels = [r for r in ex.map(differs, cands) if r]
print(f"images {len(cands)}, .text differs {len(rels)}")
by = collections.defaultdict(collections.Counter)
with cf.ProcessPoolExecutor(16) as ex:
    for case, ch in ex.map(one, rels):
        by[case][ch] += 1
for case in sorted(by):
    loops = collections.Counter()
    for ch, n in by[case].items():
        loops[tuple(c for c in ch if c.endswith("::run"))] += n
    print(f"{case:28s}", "; ".join(f"{n}x loop {{{', '.join(ch) or 'none: init only'}}}" for ch, n in loops.most_common()))
