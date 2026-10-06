#!/usr/bin/env python3
"""List which branch-predictor entry each branch writes, for all three TRISCs, over a whole Versim VCD.

usage: extract_bpmap.py <vcd or - for stdin> <out.csv>
Output rows: trisc,branch_pc,entry,new_entry_value,cycle. A row is written when an entry changes
within 3 cycles of a predictor update for that branch.
"""
import sys

P = "TOP_01_01.tt_tensix_with_l1.tensix."
want = {(P + "trisc(2).u_trisc", "i_clk"): ("clk", None)}
for t in range(3):
    cpu = P + f"trisc({t}).u_trisc.u_trisc_control.briscv"
    bp = cpu + ".ifetch.bp"
    want[(cpu, "ex_branch_pc")] = ("pc", t)
    want[(bp, "i_ex_update_vld")] = ("upd", t)
    for k in range(16):
        want[(bp, f"bp_array({k})")] = (f"e{k}", t)
f = sys.stdin.buffer if sys.argv[1] == "-" else open(sys.argv[1], "rb")
ids, stack = {}, []
for raw in f:
    line = raw.decode("latin1").strip()
    if line.startswith("$scope"):
        stack.append(line.split()[2])
    elif line.startswith("$upscope"):
        stack.pop()
    elif line.startswith("$var"):
        tk = line.split()
        lab = want.get((".".join(stack), tk[4]))
        if lab:
            ids.setdefault(tk[3].encode("latin1"), []).append(lab)
    elif line.startswith("$enddefinitions"):
        break
missing = set(want.values()) - {l for ls in ids.values() for l in ls}
if missing:
    sys.exit(f"signals not found: {sorted(map(str, missing))}")
cur = {}
pending = {0: [], 1: [], 2: []}  # (cycle, pc)
cycle, batch = 0, []
w = open(sys.argv[2], "w")
w.write("trisc,branch_pc,entry,value,cycle\n")


def flush():
    global cycle
    rising = any(lab == ("clk", None) and v == b"1" for lab, v in batch) and cur.get(("clk", None)) == b"0"
    if rising:
        cycle += 1
        for t in range(3):
            if cur.get(("upd", t)) == b"1":
                pending[t].append((cycle, cur.get(("pc", t))))
            pending[t] = [p for p in pending[t] if cycle - p[0] <= 3]
    for lab, v in batch:
        name, t = lab
        if name.startswith("e") and t is not None and cur.get(lab) not in (None, v) and pending[t]:
            c0, pc = pending[t][0]
            w.write(f"{t},{int(pc, 2):#x},{name[1:]},{int(v, 2):#x},{cycle}\n")
            pending[t].pop(0)
        cur[lab] = v


for raw in f:
    c = raw[:1]
    if c == b"#":
        flush()
        batch = []
        continue
    if c == b"b":
        val, _, i = raw[1:].strip().partition(b" ")
    elif c and c in b"01xz":
        val, i = raw[:1], raw[1:].strip()
    else:
        continue
    for lab in ids.get(i, ()):
        batch.append((lab, val))
flush()
