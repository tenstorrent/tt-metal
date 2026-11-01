#!/usr/bin/env python3
"""hunks.py ELF_A ELF_B [--regs]: the receiver loop function of two builds as a diff of normalized instructions (branch
targets, addresses and sp offsets normalized; with --regs also register names), each hunk with its source lines."""
import difflib, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def lines(elf):
    out = subprocess.run([sys.executable, os.path.join(HERE, "listing.py"), elf], capture_output=True, text=True).stdout
    res = []
    for l in out.splitlines()[1:]:
        m = re.match(r"(.)\s([0-9a-f]+)\s\s(.{44})\s(\S*)\s+(\S*)$", l)
        if not m:
            m = re.match(r"(.)\s([0-9a-f]+)\s\s(.{44})\s(\S*)\s*(\S*)", l)
        ins = m.group(3).strip()
        ins = re.sub(r"\s*#.*$", "", ins)
        ins = re.sub(r"[0-9a-f]+ <[^>]*>", "T", ins)
        ins = re.sub(r"-?\d+\(sp\)", "N(sp)", ins)
        ins = re.sub(r"addi (\w+),sp,-?\d+", r"addi \1,sp,N", ins)
        if "--regs" in sys.argv:
            ins = re.sub(r"\b(a[0-7]|s\d+|t[0-6]|ra)\b", "R", ins)
        res.append((ins, m.group(4), m.group(5)))
    return res


a, b = lines(sys.argv[1]), lines(sys.argv[2])
sm = difflib.SequenceMatcher(a=[x[0] for x in a], b=[x[0] for x in b], autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    print(f"@@ {tag} A[{i1}:{i2}] B[{j1}:{j2}]")
    for x in a[i1:i2]:
        print(f"  - {x[0]:40s} {x[1]:30s} {x[2][:90]}")
    for x in b[j1:j2]:
        print(f"  + {x[0]:40s} {x[1]:30s} {x[2][:90]}")

# Net instruction differences, independent of block order: the multiset of normalized instructions, A vs B, each with
# the source line of its first occurrence on the side that has more.
if "--net" in sys.argv:
    import collections
    ca = collections.Counter(x[0] for x in a)
    cb = collections.Counter(x[0] for x in b)
    where_a = {}
    for x in a:
        where_a.setdefault(x[0], []).append(x)
    where_b = {}
    for x in b:
        where_b.setdefault(x[0], []).append(x)
    print("== net (order-independent)")
    for k in sorted(set(ca) | set(cb)):
        d = cb[k] - ca[k]
        if d:
            src = (where_b if d > 0 else where_a)[k]
            print(f"  {d:+3d} {k:40s} " + "; ".join(sorted({f'{s[1]} {s[2].split("<")[0]}' for s in src}))[:150])
