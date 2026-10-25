#!/usr/bin/env python3
# path.py DIS START END [TAKEN_ADDR ...] [-v]: walk one path through an objdump -dl listing.
# Conditional branches fall through unless their address is listed (hex) as taken; j/jal x0 always jump.
# Stops after executing the instruction at END (or when END is reached). Counts instructions, loads/stores (and those
# on sp), taken branches and calls.
import re, sys
args = [a for a in sys.argv[1:] if a != "-v"]; verbose = "-v" in sys.argv
ins = {}; order = []
for l in open(args[0]):
    m = re.match(r"^\s+([0-9a-f]+):\t[0-9a-f]+\s+(\S+)\s*(.*)$", l)
    if m:
        a = int(m.group(1), 16); arg = re.sub(r"\s+#.*$", "", m.group(3)); arg = re.sub(r" <[^>]*>", "", arg)
        ins[a] = (m.group(2), arg); order.append(a)
nxt = {a: b for a, b in zip(order, order[1:])}
start, end = int(args[1], 16), int(args[2], 16)
taken = {}
for x in args[3:]:
    a, _, k = x.partition("*")
    taken[int(a, 16)] = int(k) if k else 10**9
CB = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz"}
pc = start; n = ld = st = spld = spst = tk = calls = 0; seen = 0
while True:
    op, arg = ins[pc]; n += 1; seen += 1
    if seen > 2000: print("runaway"); break
    if verbose: print(f"{pc:5x} {op:7s} {arg}")
    if op in ("lw", "lh", "lhu", "lb", "lbu"): ld += 1; spld += "(sp)" in arg
    if op in ("sw", "sh", "sb"): st += 1; spst += "(sp)" in arg
    if pc == end: break
    if op in CB:
        if taken.get(pc, 0) > 0:
            taken[pc] -= 1; tk += 1; pc = int(arg.split(",")[-1], 16); continue
    elif op == "j":
        tk += 1; pc = int(arg, 16); continue
    elif op in ("jal", "jalr"):
        calls += 1
    pc = nxt[pc]
print(f"instr {n}  loads {ld} (sp {spld})  stores {st} (sp {spst})  taken {tk}  calls {calls}")
