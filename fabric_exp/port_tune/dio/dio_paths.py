#!/usr/bin/env python3
"""dio_paths.py ELF [ELF ...]: static per-packet paths of the receiver's local delivery in receiver::run() (E1), per
send type, from the deliver dispatch (just after the `delivers` test) to the first instruction of retire.

The send type is tracked symbolically from the dispatch's own compare chain (li / beq / beqz / bltu) and jump table
(sh2add + lw + add + jr, resolved from the ELF's data and the registers' single definitions in the function). Every
other conditional branch is enumerated both ways (acyclic: a backward branch, a busy-wait, is not taken; no address
twice in a path). A call (jal to a function) is followed into the callee until its ret, counting the call and ret.
Per path: instructions, loads (sp-relative separately), stores (sp separately), conditional branches, jumps (incl.
indirect), calls, and NoC command register writes (addr2line chain), plus the sites it passes (for picking the path a
test's packets take, e.g. scatter chunk counts and the atomic flush)."""
import collections, re, struct, subprocess, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation/reconcile_device/write_gap")
from annot import annotate, func_range, sh, P

COND = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bltz", "bgez", "blez", "bgtz"}
LOADS = {"lw", "lh", "lhu", "lb", "lbu"}
STORES = {"sw", "sh", "sb"}
TYPES = {0: "write", 1: "inline write", 2: "atomic", 3: "fused write+atomic", 4: "scatter"}


def elf_word(elf, addr):
    for l in sh([P + "readelf", "-l", "-W", elf]).splitlines():
        p = l.split()
        if p and p[0] == "LOAD":
            off, va, fsz = int(p[1], 16), int(p[2], 16), int(p[4], 16)
            if va <= addr < va + fsz:
                with open(elf, "rb") as f:
                    f.seek(off + addr - va)
                    return struct.unpack("<I", f.read(4))[0]
    raise KeyError(hex(addr))


def gp_of(elf):
    for l in sh([P + "nm", elf]).splitlines():
        if l.endswith("__global_pointer$"):
            return int(l.split()[0], 16)


def parse(t):
    p = t.split(None, 1)
    return p[0], ([x.strip() for x in p[1].split(",")] if len(p) > 1 else [])


def site_of(ch):
    names = [f for f, _ in ch]
    for want in ("local_inline_write", "execute_chip_unicast_to_local_chip_impl", "local_atomic_inc",
                 "local_write<true>", "local_write<false>", "count_local_writes", "retire", "deliver_other"):
        if any(n.startswith(want) for n in names):
            return want
    return names[0] if names else "?"


class Fn:
    def __init__(self, elf, regex):
        self.elf = elf
        self.name, rows = annotate(elf, regex)
        self.rows = rows
        self.at = {a: i for i, (a, t, c) in enumerate(rows)}
        self.gp = gp_of(elf)
        self.before = len(rows)

    def reg_value(self, reg, before=None):
        """The value of a register's last constant definition (li / lui / addi / mv / gp, or a reload from a stack
        slot whose last store held a constant) before rows index `before` (by address: the preheader keeps the jump
        table's base and offset there, or the dispatch reloads them from the stack)."""
        before = self.before if before is None else before
        defs = [(i, t) for i, (a, t, c) in enumerate(self.rows[:before]) if parse(t)[1][:1] == [reg]
                and parse(t)[0] not in COND | STORES | {"j", "jr", "ret"}]
        if not defs:
            return None
        i, t = defs[-1]
        op, ar = parse(t)
        if op == "li":
            return int(ar[1], 0)
        if op == "lui":
            return int(ar[1], 0) << 12
        if op == "mv":
            return self.reg_value(ar[1], i)
        if op == "addi" and ar[1] == "gp":
            return self.gp + int(ar[2], 0)
        if op == "addi":
            base = self.reg_value(ar[1], i)
            return None if base is None else (base + int(ar[2], 0)) & 0xFFFFFFFF
        if op == "lw" and ar[1].endswith("(sp)"):
            slot = ar[1]
            st = [(k, u) for k, (a, u, c) in enumerate(self.rows[:i]) if parse(u)[0] == "sw" and parse(u)[1][1] == slot]
            return self.reg_value(parse(st[-1][1])[1][0], st[-1][0]) if st else None
        return None


def trace(fn, start, type_val, type_reg, callees, inject_at):
    """All acyclic paths from rows index `start` to the first retire instruction for one send type. The type's value
    is set in its register when the walk reaches `inject_at`, the dispatch's first compare (the unpack before it
    writes the register)."""
    out = []

    def walk(i, path, regs, seen, depth_fn):
        f, rows = depth_fn
        while True:
            if i >= len(rows):
                return
            if f is fn and i == inject_at:
                regs = dict(regs); regs[type_reg] = type_val
            a, t, ch = rows[i]
            if f is fn and site_of(ch) == "retire":
                out.append(path)
                return
            if (id(f), a) in seen:
                return
            seen = seen | {(id(f), a)}
            op, ar = parse(t)
            path = path + [(f.name, a, t, ch)]
            # constants and the type register
            if op == "li":
                regs = dict(regs); regs[ar[0]] = int(ar[1], 0)
            elif op == "lui":
                regs = dict(regs); regs[ar[0]] = int(ar[1], 0) << 12
            elif op == "addi" and isinstance(regs.get(ar[1]), int):
                regs = dict(regs); regs[ar[0]] = (regs[ar[1]] + int(ar[2], 0)) & 0xFFFFFFFF
            elif op == "mv" and ar[1] in regs:
                regs = dict(regs); regs[ar[0]] = regs[ar[1]]
            elif op in ("sh2add",) and isinstance(regs.get(ar[1]), int):
                bv = regs[ar[2]] if isinstance(regs.get(ar[2]), int) else f.reg_value(ar[2], i)
                regs = dict(regs); regs[ar[0]] = ("tbl", regs[ar[1]], bv)
            elif op == "lw" and ar[1].startswith("0(") and isinstance(regs.get(ar[1][2:-1]), tuple):
                _, idx, bv = regs[ar[1][2:-1]]
                regs = dict(regs); regs[ar[0]] = ("ent", elf_word(f.elf, bv + 4 * idx))
            elif op == "add" and isinstance(regs.get(ar[1]), tuple) and regs[ar[1]][0] == "ent":
                ov = regs[ar[2]] if isinstance(regs.get(ar[2]), int) else f.reg_value(ar[2], i)
                regs = dict(regs); regs[ar[0]] = ("tgt", (regs[ar[1]][1] + ov) & 0xFFFFFFFF)
            elif op not in COND | STORES | {"j", "jr", "ret", "jal"} and ar and ar[0] in regs:
                regs = dict(regs); del regs[ar[0]]
            if op == "ret":
                return ("ret", path, regs, seen)
            if op == "jr":
                tgt = regs.get(ar[0])
                assert isinstance(tgt, tuple) and tgt[0] == "tgt", (hex(a), t)
                i = f.at[tgt[1]]
                continue
            if op == "j":
                i = f.at[int(ar[0], 16)]
                continue
            if op == "jal":
                tgt = int(ar[-1], 16)
                cf = callees.setdefault(tgt, Fn(f.elf, re.escape(next(
                    l.split(None, 3)[3] for l in sh([P + "nm", "-C", "-S", f.elf]).splitlines()
                    if len(l.split(None, 3)) == 4 and int(l.split()[0], 16) == tgt))))
                sub = []

                def inner(j, pth, rg, sn):
                    r = walk_callee(cf, j, pth, rg, sn)
                    sub.extend(r)
                inner(0, path, regs, seen)
                for pth, rg, sn in sub:
                    walk(i + 1, pth, rg, sn, depth_fn)
                return
            if op in COND:
                tgt = int(ar[-1], 16)
                two = op in ("beqz", "bnez", "bltz", "bgez", "blez", "bgtz")
                x = regs.get(ar[0]); y = 0 if two else regs.get(ar[1])
                decided = None
                if isinstance(x, int) and isinstance(y, int):
                    decided = {"beq": x == y, "beqz": x == 0, "bne": x != y, "bnez": x != 0, "bltu": x < y,
                               "bgeu": x >= y, "blt": x < y, "bge": x >= y}.get(op)
                if tgt <= a:  # backward: a busy-wait or loop, not taken
                    i += 1
                    continue
                if decided is None:
                    walk(f.at[tgt], path, regs, seen, depth_fn)
                    i += 1
                    continue
                i = f.at[tgt] if decided else i + 1
                continue
            i += 1

    def walk_callee(cf, j, path, regs, seen):
        res = []

        def w(k, pth, rg, sn):
            r = walk_c(cf, k, pth, rg, sn)
            if r:
                res.extend(r)
        w(j, path, regs, seen)
        return res

    def walk_c(cf, k, pth, rg, sn):
        # walk the callee; collect (path, regs, seen) at each ret
        found = []

        def go(k, pth, rg, sn):
            while True:
                a, t, ch = cf.rows[k]
                if (id(cf), a) in sn:
                    return
                sn = sn | {(id(cf), a)}
                op, ar = parse(t)
                pth = pth + [(cf.name, a, t, ch)]
                if op == "li":
                    rg = dict(rg); rg[ar[0]] = int(ar[1], 0)
                elif op not in COND | STORES | {"j", "jr", "ret"} and ar and ar[0] in rg:
                    rg = dict(rg); del rg[ar[0]]
                if op == "ret":
                    found.append((pth, rg, sn)); return
                if op == "j":
                    k = cf.at[int(ar[0], 16)]; continue
                if op in COND:
                    tgt = int(ar[-1], 16)
                    two = op in ("beqz", "bnez", "bltz", "bgez", "blez", "bgtz")
                    x = rg.get(ar[0]); y = 0 if two else rg.get(ar[1])
                    if tgt <= a:
                        k += 1; continue
                    if isinstance(x, int) and isinstance(y, int):
                        d = {"beq": x == y, "beqz": x == 0, "bne": x != y, "bnez": x != 0}.get(op)
                        if d is not None:
                            k = cf.at[tgt] if d else k + 1; continue
                    go(cf.at[tgt], pth, rg, sn)
                    k += 1; continue
                k += 1
        go(k, pth, rg, sn)
        return found

    walk(start, [], {}, frozenset(), (fn, fn.rows))
    return out


def stats(path):
    c = collections.Counter()
    for fname, a, t, ch in path:
        op, ar = parse(t)
        c["insns"] += 1
        if op in LOADS:
            c["sp loads" if "(sp)" in t else "loads"] += 1
        if op in STORES:
            c["sp stores" if "(sp)" in t else "stores"] += 1
        if op in COND:
            c["branches"] += 1
        if op in ("j", "jr"):
            c["jumps"] += 1
        if op == "jr":
            c["indirect"] += 1
        if op in ("jal", "ret"):
            c["call/ret"] += 1
        if any(f.startswith("NOC_CMD_BUF_WRITE_REG") for f, _ in ch):
            c["cmd writes"] += 1
            names = " ".join(f for f, _ in ch)
            if op == "sw" and re.search(r",(64|-1984)\(", t):  # a command's go (NOC_CMD_CTRL)
                c["W" if "local_write" in names else "A" if "atomic" in names else "go other"] += 1
        if any(f.startswith("flush_write_to_noc_pipeline") for f, _ in ch) and op in COND:
            c["F"] = 1
    return c


def sites(path):
    s = []
    for fname, a, t, ch in path:
        x = site_of(ch)
        if x in ("local_write<true>", "local_write<false>", "local_atomic_inc", "local_inline_write",
                 "execute_chip_unicast_to_local_chip_impl", "count_local_writes") and (not s or s[-1][0] != x):
            s.append((x, a))
        if "flush_write_to_noc_pipeline" in " ".join(f for f, _ in ch) and (not s or s[-1][0] != "flush"):
            s.append(("flush", a))
    return " > ".join(x.replace("local_", "").replace("execute_chip_unicast_to_local_chip_impl", "impl")
                      for x, _ in s)


def main():
    for elf in sys.argv[1:]:
        fn = Fn(elf, r"receiver::run[<(]")
        rows = fn.rows
        d = next(i for i, (a, t, ch) in enumerate(rows) if ch and ch[0][0] == "delivers")
        b = next(i for i in range(d, d + 8) if parse(rows[i][1])[0] in COND)
        start = b + 1
        # the type register: the first register the dispatch compares against a constant or zero
        k = start
        type_reg = None
        while k < start + 12 and type_reg is None:
            op, ar = parse(rows[k][1])
            if op in ("beqz", "bnez"):
                type_reg = ar[0]
            elif op in ("beq", "bne", "bltu", "bgeu"):
                prev = [parse(rows[m][1]) for m in range(start, k)]
                lis = {x[1][0] for x in prev if x[0] == "li"}
                type_reg = ar[1] if ar[0] in lis else ar[0]
            k += 1
        inject_at = k - 1
        while parse(rows[inject_at][1])[0] not in COND and parse(rows[inject_at][1])[0] != "li":
            inject_at -= 1
        if parse(rows[inject_at - 1][1])[0] == "li":
            inject_at -= 1
        fn.before = start
        print(f"== {elf}\n   dispatch at {rows[start][0]:x}, type in {type_reg} from {rows[inject_at][0]:x}")
        callees = {}
        for tv, tn in TYPES.items():
            paths = trace(fn, start, tv, type_reg, callees, inject_at)
            for pth in paths:
                c = stats(pth)
                print(f"   {tn:20s} {c['insns']:3d} insns: {c['loads']:2d} ld {c['sp loads']} sp-ld {c['stores']:2d} st "
                      f"{c['sp stores']} sp-st {c['branches']:2d} br {c['jumps']} j ({c['indirect']} indirect) "
                      f"{c['call/ret']} call/ret {c['cmd writes']:2d} cmd; writes {c['W']} atomics {c['A']}"
                      f"{' flushed' if c['F'] else ''} other gos {c['go other']}")


if __name__ == "__main__":
    main()
