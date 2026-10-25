#!/usr/bin/env python3
# loopdiff.py A.elf B.elf: find the 32-pass inner loop of kernel_main (counter "addi r,r,-1; bnez r,<header>"
# with the largest body) in each ELF and compare the contiguous loop bodies (registers, stack offsets and branch
# targets normalized), plus the count of loop-body stack loads.
import difflib, re, subprocess, sys
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
REG = re.compile(r"\b(zero|ra|sp|gp|tp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b")
def load(elf):
    out = subprocess.run([OD, "-d", "--section=.text", elf], capture_output=True, text=True).stdout
    ins = []
    for l in out.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\t\S+(?: \S+)*\s*\t(\S+)\s*(.*)$", l)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), re.sub(r" <[^>]*>", "", re.sub(r"\s*#.*$", "", m.group(3)))))
    best = None
    for i in range(1, len(ins)):
        a, op, args = ins[i]
        if op == "bnez":
            r, tgt = args.split(",")
            pa, pop, pargs = ins[i - 1]
            if pop == "addi" and pargs == f"{r},{r},-1" and int(tgt, 16) < a:
                size = (a - int(tgt, 16)) // 4
                if best is None or size > best[0]:
                    best = (size, int(tgt, 16), a)
    _, lo, hi = best
    body = [x for x in ins if lo <= x[0] <= hi]
    return lo, hi, body
def norm(x):
    op, args = x[1], x[2]
    if re.match(r"^(b\w*|j|jal)$", op): args = re.sub(r"\b[0-9a-f]{3,}\b", "L", args)
    args = re.sub(r"-?\d+\(R\)", "M(R)", REG.sub("R", args))
    return op + " " + args
la, ha, a = load(sys.argv[1]); lb, hb, b = load(sys.argv[2])
spa = sum(1 for x in a if x[1] == "lw" and "(sp)" in x[2]); spb = sum(1 for x in b if x[1] == "lw" and "(sp)" in x[2])
sm = difflib.SequenceMatcher(None, [norm(x) for x in a], [norm(x) for x in b], autojunk=False)
d = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
print(f"loop A {la:x}-{ha:x} {len(a)} insns, {spa} sp loads | B {lb:x}-{hb:x} {len(b)} insns, {spb} sp loads | structural diff {d}")
if "--show" in sys.argv:
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "equal": continue
        print(f"--- {t}")
        for x in a[i1:i2]: print(f"  - {x[0]:5x} {x[1]} {x[2]}")
        for x in b[j1:j2]: print(f"  + {x[0]:5x} {x[1]} {x[2]}")
