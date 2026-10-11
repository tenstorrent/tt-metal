#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Experiment CI legs (init-opt2): signature of the executed INIT window of every INIT measurement ELF of a Wormhole perf
build (RV32IM interpreter of deep/init0/r2/tools/ftrace.py: MMIO reads 0, Tensix words do nothing, a short backward
branch revisited with an unchanged register file falls through), from run_kernel to the INIT zone's end read (the
second wall clock read). Per ELF: sha1 of the window words without NOPs (branch and jump offsets masked), NOP count,
window length. Two builds' INIT windows are a pure NOP insertion when the sha1 matches; k = the NOP count difference.
usage: exp_initwin.py RUNNER_TEMP OUT.json.gz   (stdlib only)"""
import sys, os, json, gzip, struct, hashlib
from pathlib import Path
M32 = 0xFFFFFFFF
CLOCK_LO = 0xFFB121F0
RUN_KERNEL = "_Z10run_kernelRK13RuntimeParams"
PARAMS, STACK, RET = 0x70000000, 0x7FFF0000, 0x7EEE0000
SPIN_SPAN = 32

# instruction classes
K_ALU, K_MUL, K_DIV, K_LOAD, K_STORE, K_BR, K_JAL, K_JALR, K_TT, K_SYS, K_LUI = range(11)


def sext(x, b):
    return x - (1 << b) if x & (1 << (b - 1)) else x


def s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


class Prog:
    def __init__(self, path):
        data = Path(path).read_bytes()
        (shoff,) = struct.unpack_from("<I", data, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
        heads = [struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize) for i in range(shnum)]
        name = lambda off, tab: data[tab[4] + off: data.index(b"\0", tab[4] + off)].decode()
        self.sections, self.symbols = {}, {}
        for h in heads:
            n = name(h[0], heads[shstrndx])
            if h[2] & 2 and h[1] != 8:
                self.sections[n] = (h[3], data[h[4]: h[4] + h[5]])
            if h[1] == 2:
                for off in range(h[4], h[4] + h[5], 16):
                    st_name, st_value, st_size = struct.unpack_from("<III", data, off)
                    self.symbols[name(st_name, heads[h[6]])] = (st_value, st_size)
        self.words = {}
        for n, (a, d) in self.sections.items():
            if n.startswith(".text") or n.startswith(".init") or n.startswith(".llk") or n == ".loader_init":
                for i in range(0, len(d) - 3, 4):
                    self.words[a + i] = struct.unpack_from("<I", d, i)[0]
        self.rk = self.symbols[RUN_KERNEL]
        self.dec = {}

    def decode(self, pc):
        d = self.dec.get(pc)
        if d is not None:
            return d
        w = self.words.get(pc)
        if w is None:
            return None
        op = w & 0x7F
        rd, f3, rs1, rs2, f7 = (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31, w >> 25
        if w & 3 != 3:
            d = (K_TT, 0, 0, 0, 0, 0, 0, w)
        elif op == 0x37:
            d = (K_LUI, rd, 0, 0, w & 0xFFFFF000, 0, 0, w)
        elif op == 0x17:
            d = (K_LUI, rd, 0, 0, (pc + (w & 0xFFFFF000)) & M32, 1, 0, w)
        elif op == 0x6F:
            off = sext(((w >> 31) & 1) << 20 | ((w >> 12) & 0xFF) << 12 | ((w >> 20) & 1) << 11 | ((w >> 21) & 0x3FF) << 1, 21)
            d = (K_JAL, rd, 0, 0, (pc + off) & M32, 0, 0, w)
        elif op == 0x67:
            d = (K_JALR, rd, rs1, 0, sext(w >> 20, 12), 0, 0, w)
        elif op == 0x63:
            off = sext(((w >> 31) & 1) << 12 | ((w >> 7) & 1) << 11 | ((w >> 25) & 0x3F) << 5 | ((w >> 8) & 0xF) << 1, 13)
            d = (K_BR, 0, rs1, rs2, (pc + off) & M32, f3, 0, w)
        elif op == 0x03:
            d = (K_LOAD, rd, rs1, 0, sext(w >> 20, 12), f3, 0, w)
        elif op == 0x23:
            d = (K_STORE, 0, rs1, rs2, sext(((w >> 25) << 5) | ((w >> 7) & 31), 12), f3, 0, w)
        elif op == 0x13:
            d = (K_ALU, rd, rs1, 0, sext(w >> 20, 12), f3, (w >> 30) & 1, w)
        elif op == 0x33:
            k = K_ALU if f7 != 1 else (K_MUL if f3 < 4 else K_DIV)
            d = (k, rd, rs1, rs2, 0x33, f3, f7, w)
        else:
            d = (K_SYS, rd, rs1, 0, 0, f3, op, w)
        self.dec[pc] = d
        return d


class Mem:
    def __init__(self, prog, argbytes):
        self.m = {}
        for n, (a, d) in prog.sections.items():
            for i, b in enumerate(d):
                self.m[a + i] = b
        for i, b in enumerate(argbytes):
            self.m[PARAMS + i] = b

    @staticmethod
    def ram(a):
        return a < 0x200000 or 0xFFB00000 <= a < 0xFFB10000 or 0x70000000 <= a < 0x80000000

    def ld(self, a, n):
        if not self.ram(a):
            return 0
        m = self.m
        return sum(m.get(a + i, 0) << (8 * i) for i in range(n))

    def st(self, a, v, n):
        if self.ram(a):
            for i in range(n):
                self.m[a + i] = (v >> (8 * i)) & 0xFF




def window(elf, budget=200000):
    prog = Prog(elf)
    mem = Mem(prog, b"")
    x = [0] * 32
    x[2] = STACK; x[3] = prog.symbols.get("__global_pointer$", (0, 0))[0]; x[10] = PARAMS; x[1] = RET
    pc = prog.rk[0]
    PC = []; clk = []; seen = set(); n = 0; dec = prog.decode
    while n < budget and len(clk) < 2:
        if pc == RET:
            break
        d = dec(pc)
        if d is None:
            break
        k, rd, rs1, rs2, imm, f3, f7, w = d
        nxt = pc + 4
        if k == K_ALU:
            a = x[rs1]
            if imm == 0x33 and f7 in (0, 0x20) and (w & 0x7F) == 0x33:
                b = x[rs2]
                if f3 == 0: v = (a - b) if f7 == 0x20 else (a + b)
                elif f3 == 1: v = a << (b & 31)
                elif f3 == 2: v = int(s32(a) < s32(b))
                elif f3 == 3: v = int(a < b)
                elif f3 == 4: v = a ^ b
                elif f3 == 5: v = (s32(a) >> (b & 31)) if f7 == 0x20 else (a >> (b & 31))
                elif f3 == 6: v = a | b
                else: v = a & b
            else:
                i = imm; sh = (w >> 20) & 31
                if f3 == 0: v = a + i
                elif f3 == 2: v = int(s32(a) < i)
                elif f3 == 3: v = int(a < (i & M32))
                elif f3 == 4: v = a ^ (i & M32)
                elif f3 == 6: v = a | (i & M32)
                elif f3 == 7: v = a & (i & M32)
                elif f3 == 1: v = a << sh
                else: v = (s32(a) >> sh) if f7 else (a >> sh)
            if rd: x[rd] = v & M32
        elif k == K_LUI:
            if rd: x[rd] = imm
        elif k == K_LOAD:
            ea = (x[rs1] + imm) & M32
            nb = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}.get(f3, 4)
            v = mem.ld(ea, nb)
            if f3 == 0 and v & 0x80: v -= 0x100
            if f3 == 1 and v & 0x8000: v -= 0x10000
            if rd: x[rd] = v & M32
            if ea == CLOCK_LO:
                clk.append(n)
        elif k == K_STORE:
            ea = (x[rs1] + imm) & M32
            mem.st(ea, x[rs2], {0: 1, 1: 2, 2: 4}.get(f3, 4))
        elif k == K_BR:
            a, b = x[rs1], x[rs2]
            if f3 == 0: c = a == b
            elif f3 == 1: c = a != b
            elif f3 == 4: c = s32(a) < s32(b)
            elif f3 == 5: c = s32(a) >= s32(b)
            elif f3 == 6: c = a < b
            else: c = a >= b
            if c and imm <= pc and pc - imm <= SPIN_SPAN:
                key = (pc, tuple(x))
                if key in seen: c = False
                seen.add(key)
            if c:
                nxt = imm
        elif k == K_JAL:
            if rd: x[rd] = pc + 4
            nxt = imm
        elif k == K_JALR:
            t = (x[rs1] + imm) & ~1 & M32
            if rd: x[rd] = pc + 4
            nxt = t
        elif k in (K_MUL, K_DIV):
            a, b = x[rs1], x[rs2]
            sa, sb = s32(a), s32(b)
            if f3 == 0: v = a * b
            elif f3 == 1: v = (sa * sb) >> 32
            elif f3 == 2: v = (sa * b) >> 32
            elif f3 == 3: v = (a * b) >> 32
            elif f3 == 4: v = int(sa / sb) if sb else -1
            elif f3 == 5: v = a // b if b else M32
            elif f3 == 6: v = (abs(sa) % abs(sb)) * (1 if sa >= 0 else -1) if sb else sa
            else: v = a % b if b else a
            if rd: x[rd] = v & M32
        elif k == K_SYS:
            if f7 == 0x73 and f3 and rd:
                x[rd] = 0
        PC.append(pc)
        pc = nxt
        n += 1
    if len(clk) < 2:
        return None
    words = []
    for p in PC[clk[0]:clk[1] + 1]:
        w = prog.words.get(p)
        op = w & 0x7F if w is not None else None
        if op == 0x63:
            w &= 0x01FFF07F
        elif op in (0x6F, 0x17):
            w &= 0xFFF
        words.append(w)
    core = [w for w in words if w != 0x13]
    sig = hashlib.sha1(json.dumps(core).encode()).hexdigest()[:20]
    return {"sig": sig, "nop": len(words) - len(core), "n": len(words), "pc0": PC[clk[0]]}


def _one(e):
    try:
        return e, window(e)
    except Exception as ex:  # never fail a leg on the diagnostic
        return e, {"err": repr(ex)[-200:]}


def main(rt, out):
    from multiprocessing import Pool

    root = Path(rt) / "tt-llk-build" / "sources"
    elfs = [str(e) for e in sorted(root.glob("*/*/init_elf/*.elf"))]
    with Pool(os.cpu_count() or 4) as pool:
        res = {os.path.relpath(e, root): w for e, w in pool.imap_unordered(_one, elfs, chunksize=16)}
    json.dump(res, gzip.open(out, "wt"))
    print(f"exp_initwin: {len(res)} INIT ELFs -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
