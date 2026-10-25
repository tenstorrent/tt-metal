#!/usr/bin/env python3
"""walk.py DIS HEAD END --avoid 'file:line,file:lo-hi,...' [--jr ADDR=T1,T2] [--need file:line,...] [-v]
Shortest path from HEAD to END (inclusive) through the objdump -dl listing that never executes an instruction whose
source location is in the avoid set and (if --need) executes at least one in each need set. Short backward branches
(<= 16 bytes back) are polls and fall through. Prints instructions, loads (sp), stores (sp), taken branches."""
import re, sys, argparse, heapq
ap = argparse.ArgumentParser()
ap.add_argument("dis"); ap.add_argument("head"); ap.add_argument("end")
ap.add_argument("--avoid", default=""); ap.add_argument("--need", default="")
ap.add_argument("--jr", action="append", default=[]); ap.add_argument("-v", action="store_true")
ap.add_argument("--elf", default=None); ap.add_argument("--jrdbg", action="store_true")
ap.add_argument("--issues", type=int, default=0); ap.add_argument("--need-any", action="store_true")
ap.add_argument("--eq", default=""); ap.add_argument("--ne", default="")
a = ap.parse_args()
ins = {}; order = []; cur = None
for l in open(a.dis):
    m = re.match(r"^(/\S+):(\d+)", l)
    if m: cur = (m.group(1).split("/")[-1], int(m.group(2))); continue
    m = re.match(r"^\s+([0-9a-f]+):\t[0-9a-f]+\s+(\S+)\s*(.*)$", l)
    if m:
        ad = int(m.group(1), 16); arg = re.sub(r"\s+#.*$", "", m.group(3)); arg = re.sub(r" <[^>]*>", "", arg)
        ins[ad] = (m.group(2), arg, cur); order.append(ad)
nxt = {x: y for x, y in zip(order, order[1:])}
def parse_set(s):
    out = []
    for t in filter(None, s.split(",")):
        f, r = t.rsplit(":", 1)
        lo, _, hi = r.partition("-"); out.append((f, int(lo), int(hi or lo)))
    return out
avoid = parse_set(a.avoid); EQ = parse_set(a.eq); NE = parse_set(a.ne); need = [parse_set(x) for x in filter(None, a.need.split(";"))]
def inset(loc, st):
    return loc is not None and any(loc[0].startswith(f) and lo <= loc[1] <= hi for f, lo, hi in st)
jr = {}
def resolve_jr():
    import subprocess
    OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
    syms = subprocess.run([OD, "-t", a.elf], capture_output=True, text=True).stdout
    gp = None
    for l in syms.splitlines():
        if l.endswith("__global_pointer$"): gp = int(l.split()[0], 16)
    data = subprocess.run([OD, "-s", a.elf], capture_output=True, text=True).stdout
    mem = {}
    for l in data.splitlines():
        m = re.match(r"^ ([0-9a-f]{8}) ((?:[0-9a-f]{8} ?)+)", l)
        if m:
            base = int(m.group(1), 16)
            for k, w in enumerate(m.group(2).split()):
                v = int(w, 16); mem[base + 4 * k] = int.from_bytes(v.to_bytes(4, "big"), "little")
    reg = {}; slot = {}; recent = []; lic = {}
    def imm(x):
        return int(x, 0)
    tlo, thi = min(order), max(order)
    for ad in order + order:
        op, arg, loc = ins[ad]
        parts = [p.strip() for p in arg.split(",")]
        if op == "li" and len(parts) == 2:
            try: lic[parts[0]] = int(parts[1], 0)
            except ValueError: pass
        elif parts and parts[0] in lic and op not in ("sw", "sh", "sb", "j", "jr", "ret") and not op.startswith("b"): lic.pop(parts[0], None)
        if op == "lui":
            reg[parts[0]] = (imm(parts[1]) << 12) & 0xFFFFFFFF
            recent.append((ad, reg[parts[0]]))
        elif op == "addi" and len(parts) == 3:
            rs = parts[1]
            if rs == "gp" and gp is not None: reg[parts[0]] = (gp + imm(parts[2])) & 0xFFFFFFFF
            elif rs in reg: reg[parts[0]] = (reg[rs] + imm(parts[2])) & 0xFFFFFFFF
            else: reg.pop(parts[0], None)
            if parts[0] in reg: recent.append((ad, reg[parts[0]]))
        elif op == "sw" and parts[1].endswith("(sp)"):
            off = int(parts[1][:-4]); 
            if parts[0] in reg: slot[off] = reg[parts[0]]
        elif op in ("jr",):
            # look back for lw of stack slots
            vals = [v for (a2, v) in recent if 0 <= ad - a2 <= 48]
            b = ad
            for _ in range(10):
                b = prev.get(b)
                if b is None: break
                o2, a2, _ = ins[b]
                p2 = [p.strip() for p in a2.split(",")]
                if o2 == "lw" and p2[1].endswith("(sp)") and int(p2[1][:-4]) in slot:
                    vals.append(slot[int(p2[1][:-4])])
            tbl = [v for v in vals if v in mem]; bases = [v for v in vals if tlo <= v <= thi + 0x1000]
            nent = 16
            b = ad; lis = {}
            for _ in range(16):
                b = prev.get(b)
                if b is None: break
                o2, a2, _ = ins[b]
                p2 = [p.strip() for p in a2.split(",")]
                if o2 in ("bltu", "bgeu") and nent == 16: cand = p2[0]; nent = -1
                elif o2 == "li" and nent == -1 and p2[0] == cand: nent = int(p2[1], 0) + 1
            if nent < 0: nent = (lic[cand] + 1) if lic.get(cand, 99) < 16 else 5  # unknown bound: the send-type switch (5 arms)
            if tbl and bases:
                T = tbl[-1]; B = bases[-1]; ts = []
                for k in range(nent):
                    if T + 4 * k not in mem: break
                    t = (B + mem[T + 4 * k]) & 0xFFFFFFFF
                    if t in ins: ts.append(t)
                    else: break
                jr[ad] = sorted(set(ts))
        else:
            if parts and parts[0] in reg and op not in ("sw", "sh", "sb") and not op.startswith("b") and op not in ("j",):
                reg.pop(parts[0], None)
prev = {y: x for x, y in zip(order, order[1:])}
if a.elf: resolve_jr()
if a.jrdbg:
    for ad in order:
        if ins[ad][0] == "jr" and ins[ad][1].strip() != "ra" or (ins[ad][0] == "jalr" and ins[ad][1].startswith("zero")):
            print(f"jr {ad:x} {ins[ad][1]} -> {[hex(x) for x in jr.get(ad, [])]}")
    sys.exit(0)
for j in a.jr:
    k, v = j.split("="); jr[int(k, 16)] = [int(x, 16) for x in v.split(",")]
MEMOPS = {"lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb"}  # avoid sets block memory accesses and calls only
CB = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz"}
head, end = int(a.head, 16), int(a.end, 16)
# Dijkstra over (pc, needs_met mask) with cost = instructions
best = None
pq = [(0, head, 0, (), (head,))]
seen = {}
full = (1 << len(need)) - 1
def is_issue(op, arg, loc):
    # A NoC command issue: a store to NOC_CMD_CTRL (offset 0x40 within a 0x800-byte command buffer) from the NoC API.
    if op != "sw" or loc is None or not loc[0].startswith("noc_nonblocking_api"): return False
    m = re.match(r"\S+,(-?\d+)\(", arg)
    return m is not None and int(m.group(1)) % 2048 == 64
while pq:
    cost, pc, mask, stk, path = heapq.heappop(pq)
    key = (pc, mask, stk)
    if key in seen and seen[key] <= cost: continue
    seen[key] = cost
    op, arg, loc = ins[pc]
    if inset(loc, avoid) and pc != head and (op in MEMOPS or op == "jal"): continue
    if a.need_any or op in MEMOPS:
        for i, ns in enumerate(need):
            if inset(loc, ns): mask |= 1 << i
    if a.issues and is_issue(op, arg, loc):
        mask = (mask & ~(0xF << 16)) | (min(((mask >> 16) & 0xF) + 1, a.issues) << 16)
    if pc == end and not stk:
        if (mask & 0xFFFF) == full and ((mask >> 16) & 0xF) >= a.issues: best = path; break
        continue
    succ = []; nstk = stk
    if op in CB:
        t = int(arg.split(",")[-1], 16)
        if t < pc and pc - t <= 16: succ = [nxt[pc]]
        else: succ = [t, nxt[pc]]
        if op in ("beq", "beqz", "bne", "bnez") and (inset(loc, EQ) or inset(loc, NE)):
            want_eq = inset(loc, EQ)
            succ = [t] if (op in ("beq", "beqz")) == want_eq else [nxt[pc]]
    elif op == "j": succ = [int(arg, 16)]
    elif op == "jal":
        parts = arg.split(",")
        t = int(parts[-1], 16)
        if (len(parts) == 1 or parts[0] == "ra") and len(stk) < 4:
            succ = [t]; nstk = stk + (nxt[pc],)
        elif len(parts) == 2 and parts[0] == "zero": succ = [t]
        else: succ = []
    elif op in ("jr",) or (op == "jalr" and arg.startswith("zero")):
        if arg.strip() == "ra":
            succ = [stk[-1]] if stk else []; nstk = stk[:-1]
        else:
            succ = jr.get(pc, [])
    elif op == "ret":
        succ = [stk[-1]] if stk else []; nstk = stk[:-1]
    else: succ = [nxt[pc]]
    for s2 in succ:
        if s2 in ins and len(path) < 3000:
            heapq.heappush(pq, (cost + 1, s2, mask, nstk, path + (s2,)))
if best is None:
    print("no path"); sys.exit(1)
n = ld = st = spld = spst = tk = calls = iss = 0
for i, pc in enumerate(best):
    op, arg, loc = ins[pc]; n += 1
    if op in ("lw", "lh", "lhu", "lb", "lbu"): ld += 1; spld += "(sp)" in arg
    if op in ("sw", "sh", "sb"): st += 1; spst += "(sp)" in arg
    if i + 1 < len(best) and best[i + 1] != nxt.get(pc) and op not in ('jal', 'ret', 'jalr') and not (op == 'jr' and arg.strip() == 'ra'): tk += 1
    if op in ('jal', 'ret') or (op == 'jr' and arg.strip() == 'ra'): calls += 1
    iss += is_issue(op, arg, loc)
    if a.v: print(f"{pc:5x} {op:7s} {arg:28s} {loc[0] if loc else ''}:{loc[1] if loc else ''}")
print(f"instr {n}  loads {ld} (sp {spld})  stores {st} (sp {spst})  taken {tk}+1" + (f"  call/ret {calls}" if calls else "") + f"  issues {iss}")
