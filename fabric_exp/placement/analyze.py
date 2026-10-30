#!/usr/bin/env python3
"""analyze.py OBJ [--slots]: run()'s instructions, and its stack loads/stores, by loop depth (backward branches nest;
depth 2 = the pass loop), with the source line of each stack access when --slots."""
import collections, re, subprocess, sys
B = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-"
BR = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "bgt", "ble", "bgtu",
      "bleu", "j"}
obj = sys.argv[1]
src = {}
sym = next(l.split()[-1] for l in subprocess.run([B + "nm", obj], capture_output=True, text=True).stdout.splitlines()
           if l.split()[-1].startswith("_Z3run"))
d = subprocess.run([B + "objdump", "-d", "-l", "--no-show-raw-insn", "--disassemble=" + sym, obj],
                   capture_output=True, text=True).stdout
ins, cur = [], "?"
for l in d.splitlines():
    m = re.match(r"^(/\S+):(\d+)", l)
    if m:
        p, n = m.group(1), int(m.group(2))
        if p not in src:
            try:
                src[p] = open(p).read().splitlines()
            except OSError:
                src[p] = []
        txt = src[p][n - 1].strip()[:60] if n <= len(src[p]) else ""
        cur = f"{p.split('/')[-1]}:{n} {txt}"
        continue
    m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)", l)
    if m:
        ins.append((int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip(), cur))
loops = []
for a, mn, ops, _ in ins:
    if mn in BR:
        m = re.search(r"\b([0-9a-f]+) <", ops)
        if m and int(m.group(1), 16) <= a:
            loops.append((int(m.group(1), 16), a))
depth = lambda a: sum(1 for t, b in loops if t <= a <= b)
by = collections.Counter()
sp = collections.defaultdict(lambda: [0, 0, collections.Counter()])
for a, mn, ops, s in ins:
    dd = min(depth(a), 2)
    by[(dd, "insn")] += 1
    m = re.search(r"(-?\d+)\(sp\)", ops)
    if m:
        st = mn in ("sw", "sh", "sb", "c.sw", "c.swsp")
        by[(dd, "st" if st else "ld")] += 1
        if dd >= 2:
            sp[int(m.group(1))][1 if st else 0] += 1
            sp[int(m.group(1))][2][s] += 1
    if mn in ("lw", "lh", "lhu", "lb", "lbu") and "(sp)" not in ops:
        by[(dd, "mem ld")] += 1
    if mn in ("sw", "sh", "sb") and "(sp)" not in ops:
        by[(dd, "mem st")] += 1
frame = next((int(m.group(1)) for _, mn, ops, _ in ins[:3] for m in [re.match(r"sp,sp,-(\d+)", ops.replace(" ", ""))] if m), 0)
print(f"frame {frame:4} | pass loop: insn {by[(2,'insn')]:4} sp ld {by[(2,'ld')]:3} sp st {by[(2,'st')]:3} "
      f"mem ld {by[(2,'mem ld')]:3} mem st {by[(2,'mem st')]:3} | whole: insn {sum(v for k, v in by.items() if k[1]=='insn')}")
if "--slots" in sys.argv:
    for off in sorted(sp):
        ld, st, c = sp[off]
        print(f"   sp+{off:<4} ld {ld:2} st {st:2}  " + " | ".join(f"{k} x{n}" for k, n in c.most_common(2)))

if "--inv" in sys.argv:
    # the pass loop: the widest backward-branch range nested inside another one
    inner = [(t, b) for t, b in loops if any(t2 < t and b <= b2 for t2, b2 in loops if (t2, b2) != (t, b))]
    t0, b0 = max(inner, key=lambda r: r[1] - r[0])
    REGS = re.compile(r"\b(ra|t[0-6]|s[0-9]|s1[01]|a[0-7])\b")
    reads, writes = set(), set()
    for a, mn, ops, s in ins:
        if t0 <= a <= b0:
            regs = REGS.findall(ops)
            if not regs:
                continue
            if mn.startswith(("s", "c.s")) and mn not in ("slli", "srli", "srai", "sll", "srl", "sra", "sub", "snez", "seqz", "sltu", "slt", "sltiu", "slti", "sext.b", "sext.h", "seqz") or mn.startswith("b"):
                reads.update(regs)  # stores and branches read every register operand
            else:
                writes.add(regs[0]); reads.update(regs[1:])
            if mn in ("jal", "jalr", "call"):
                writes.update(["ra", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0", "t1", "t2", "t3", "t4", "t5", "t6"])
    inv = sorted(reads - writes)
    print(f"pass loop {t0:x}-{b0:x}: {len(inv)} registers read but never written in it:")
    for r in inv:
        last = None
        for a, mn, ops, s in ins:
            if a >= t0:
                break
            regs = REGS.findall(ops)
            if regs and regs[0] == r and not (mn.startswith(("s", "c.s")) and mn not in ("slli", "srli", "srai", "sub", "snez", "seqz", "sltu", "slt")):
                last = (mn, ops, s)
        print(f"   {r:4} set by {last[0] + ' ' + last[1] if last else '?':28} {last[2][:70] if last else ''}")

if "--regs" in sys.argv:
    inner = [(t, b) for t, b in loops if any(t2 < t and b <= b2 for t2, b2 in loops if (t2, b2) != (t, b))]
    t0, b0 = max(inner, key=lambda r: r[1] - r[0])
    REGS = re.compile(r"\b(t[0-6]|s[0-9]|s1[01]|a[0-7])\b")
    w = collections.defaultdict(collections.Counter)
    calls = 0
    for a, mn, ops, s in ins:
        if t0 <= a <= b0:
            if mn in ("jal", "jalr", "call") or "auipc" in mn:
                calls += mn == "jalr"
            regs = REGS.findall(ops)
            if regs and not mn.startswith("b") and not (mn.startswith(("sw", "sh", "sb"))):
                w[regs[0]][s.split(" ", 1)[1][:55] if " " in s else s] += 1
    print(f"calls in the pass loop: {calls}")
    for r in sorted(w, key=lambda x: (x[0], int(x[1:]))):
        if r.startswith("s"):
            print(f"   {r:4} written by: " + " | ".join(f"{k} x{n}" for k, n in w[r].most_common(2)))
