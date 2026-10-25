#!/usr/bin/env python3
"""hotdiff.py A.elf B.elf: compare two router ELFs' .text with branch/jump targets normalized away, so that
code that only moved does not count. Prints instruction counts, per-function sizes that changed, and the
normalized difference (instructions added/removed/replaced), with the differing hunks when --show is given."""
import difflib, re, subprocess, sys

OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
BR = re.compile(r"^(b\w*|j|jal|jr|c\.j|c\.jal|c\.b\w*)$")


def load(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    funcs, ins, cur = {}, [], None
    for l in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", l)
        if m:
            if not m.group(1).startswith("."):
                cur = m.group(1)
                funcs.setdefault(cur, 0)
            continue
        m = re.match(r"\s+([0-9a-f]+):\t\S+(?: \S+)*\s*\t(\S+)\s*(.*)$", l)
        if not m:
            continue
        addr, op, args = m.group(1), m.group(2), re.sub(r"\s*#.*$", "", m.group(3))
        args = re.sub(r" <[^>]*>", "", args)
        if BR.match(op):
            args = re.sub(r"\b[0-9a-f]{3,}\b", "L", args)
        ins.append((cur, f"{op} {args}".strip(), int(addr, 16)))
        funcs[cur] += 1
    return funcs, ins


a_f, a = load(sys.argv[1])
b_f, b = load(sys.argv[2])
show = "--show" in sys.argv
print(f"insns {len(a)} -> {len(b)} ({len(b) - len(a):+d})")
for f in sorted(set(a_f) | set(b_f)):
    if a_f.get(f, 0) != b_f.get(f, 0):
        print(f"  func {f}: {a_f.get(f, 0)} -> {b_f.get(f, 0)}")
sm = difflib.SequenceMatcher(None, [x[1] for x in a], [x[1] for x in b], autojunk=False)
add = rem = rep = 0
for t, i1, i2, j1, j2 in sm.get_opcodes():
    if t == "equal":
        continue
    if t == "insert":
        add += j2 - j1
    elif t == "delete":
        rem += i2 - i1
    else:
        rep += max(i2 - i1, j2 - j1)
    if show:
        print(f"--- {t} A[{i1}:{i2}] @{a[i1][2] if i1 < len(a) else 0:x} / B[{j1}:{j2}] @{b[j1][2] if j1 < len(b) else 0:x}")
        for x in a[i1:i2]:
            print(f"  - {x[2]:5x} {x[1]}")
        for x in b[j1:j2]:
            print(f"  + {x[2]:5x} {x[1]}")
print(f"normalized diff: +{add} -{rem} ~{rep}")
