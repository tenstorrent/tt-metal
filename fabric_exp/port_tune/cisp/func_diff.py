#!/usr/bin/env python3
"""func_diff.py PAIRS_JSON [PROC]: per image pair (pair_images2.py's JSON: a, b), the functions of each ELF compared by
code with addresses and immediates masked: functions only in a, only in b, present in both with different code, and
the number with the same code. Summarised over all pairs of PROC (default active_erisc): the function names that are
ever only in a / only in b / differ, with counts, and how many pairs have every shared function code-identical."""
import collections, json, re, subprocess, sys

P = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
REG_IMM = re.compile(r"-?\b0x[0-9a-f]+\b|-?\b[0-9a-f]{4,}\b|-?\b\d+\b")


def functions(elf):
    out = subprocess.run([P + "objdump", "-d", "-C", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    fns, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            name = m.group(1)
            if not name.startswith(".L"):  # a local label continues the current function
                cur = name
                fns.setdefault(cur, [])
            continue
        m = re.match(r"\s+[0-9a-f]+:\s+(.*)$", line)
        if m and cur:
            fns[cur].append(REG_IMM.sub("N", re.sub(r" <[^>]*>|#.*$", "", m.group(1))).strip())
    return {k: tuple(v) for k, v in fns.items()}


def short(name):
    return re.sub(r"\(.*", "", name)[:90]


pairs = [p for p in json.load(open(sys.argv[1])) if p["proc"] == (sys.argv[2] if len(sys.argv) > 2 else "active_erisc")]
only_a, only_b, differ = collections.Counter(), collections.Counter(), collections.Counter()
clean = 0
for p in pairs:
    fa, fb = functions(p["a"]), functions(p["b"])
    oa = set(fa) - set(fb); ob = set(fb) - set(fa)
    d = {k for k in set(fa) & set(fb) if fa[k] != fb[k]}
    only_a.update(short(k) for k in oa); only_b.update(short(k) for k in ob); differ.update(short(k) for k in d)
    clean += not d
print(f"{len(pairs)} pairs; every shared function code-identical in {clean}")
for title, c in (("only in A", only_a), ("only in B", only_b), ("in both, different code", differ)):
    print(f"  {title}:")
    for k, n in c.most_common(20):
        print(f"    x{n:3d} {k}")
