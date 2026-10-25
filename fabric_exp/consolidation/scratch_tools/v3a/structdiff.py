#!/usr/bin/env python3
# structdiff.py A.elf B.elf [func]: like hotdiff but also normalizes register names and stack offsets, to separate
# register-allocation churn from structural change.
import difflib, re, subprocess, sys
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
BR = re.compile(r"^(b\w*|j|jal|jr)$")
REG = re.compile(r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b")
def load(elf, fn):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    ins, cur = [], None
    for l in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", l)
        if m:
            if not m.group(1).startswith("."): cur = m.group(1)
            continue
        m = re.match(r"\s+([0-9a-f]+):\t\S+(?: \S+)*\s*\t(\S+)\s*(.*)$", l)
        if not m or (fn and cur != fn): continue
        op, args = m.group(2), re.sub(r" <[^>]*>", "", re.sub(r"\s*#.*$", "", m.group(3)))
        if BR.match(op): args = re.sub(r"\b[0-9a-f]{3,}\b", "L", args)
        args = REG.sub("R", args)
        args = re.sub(r"-?\d+\(R\)", "M(R)", args)
        ins.append((int(m.group(1), 16), f"{op} {args}".strip()))
    return ins
fn = sys.argv[3] if len(sys.argv) > 3 else None
a, b = load(sys.argv[1], fn), load(sys.argv[2], fn)
sm = difflib.SequenceMatcher(None, [x[1] for x in a], [x[1] for x in b], autojunk=False)
tot = {"insert": 0, "delete": 0, "replace": 0}
for t, i1, i2, j1, j2 in sm.get_opcodes():
    if t == "equal": continue
    tot[t] += max(i2 - i1, j2 - j1)
    if "--show" in sys.argv:
        print(f"--- {t} A@{a[i1][0] if i1 < len(a) else 0:x}+{i2-i1} B@{b[j1][0] if j1 < len(b) else 0:x}+{j2-j1}")
        for x in a[i1:i2]: print(f"  - {x[0]:5x} {x[1]}")
        for x in b[j1:j2]: print(f"  + {x[0]:5x} {x[1]}")
print(f"{len(a)} -> {len(b)}; structural diff (regs/stack offsets/branch targets normalized): {tot}")
