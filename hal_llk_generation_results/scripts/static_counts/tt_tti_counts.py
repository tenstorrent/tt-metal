#!/usr/bin/env python3
"""Static TT / TTI counts of the original and HAL kernels of one or more HAL runs.

Usage:
  tt_tti_counts.py <run_dir> [<run_dir> ...] [--md OUT.md] [--json OUT.json] [--tsv STORES.tsv]
                   [--elfray PATH] [--objdump PATH] [--orig TEST.cpp] [--hal TEST.cpp]

For every run, side (original / HAL), kernel (unpack / math / pack) and build it reports:

  text bytes    size of the .text section (from the ELF section header)
  TTI           lines of `elfray --section .text` whose mnemonic starts with "tt"
  TT elfray     store lines that elfray marks with "; tensix issue" (a lower bound)
  TT linear     buffer stores found by linear register tracking (same rule as the harness
                elfray_tt_tti.py, count_buffer_stores)
  TT all paths  stores whose address lies in 0xFFE40000-0xFFE4FFFF on every path, by a forward
                dataflow analysis over each function's control-flow graph (see "Analysis" below)
  unresolved    stores whose address the analysis cannot settle (unknown, or in the buffer
                window on some paths only); they are counted and listed, never dropped

Inputs per run: <run>/codesize_tmp/tt-llk-build/sources/<test>/<variant>/{build.h,elf/<kernel>.elf}.
objdump defaults to <run>/ws/tt_metal/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-objdump,
elfray to $ELFRAY or ../elfray next to this script. Builds of the two sides are paired by the
compile-time parameters in build.h.

Analysis (TT all paths). Values are sets of up to 16 items: an unsigned interval [lo, hi], a
stack address (sp at function entry + offset) or "opcode only" (top byte known, low 24 bits
run-time). Calls keep s0-s11/sp/gp/tp and clobber the other registers; arguments are joined over
all call sites (and tail calls). gp comes from __global_pointer$. A word loaded from a constant
address in an initialised data section takes its initial value from the ELF unless some store
may write that word. Assumptions: callees follow the ABI; memory changes only through visible
stores; stack slots change only through stack-derived pointers.
Opcode of a TT store: the top byte of the stored value. "proven" when the value set fixes it,
"assumed" when a constant opcode word is combined (add / or) with a run-time part that is
assumed to stay below bit 24 (the same assumption elfray makes when it names the opcode).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

KERNELS = ["unpack", "math", "pack"]
SIDES = ["original", "hal"]
DEFAULT_TESTS = {"original": "eltwise_binary_test.cpp", "hal": "eltwise_binary_hal_test.cpp"}
BUF_LO, BUF_HI = 0xFFE40000, 0xFFE4FFFF
M32 = 0xFFFFFFFF
LIMIT = 16
WIDEN_AFTER = 40

# --------------------------------------------------------------------------------------------
# ELF reader (32-bit little-endian)
# --------------------------------------------------------------------------------------------
SHF_WRITE, SHF_ALLOC, SHF_EXEC = 1, 2, 4
SHT_PROGBITS, SHT_SYMTAB, SHT_NOBITS = 1, 2, 8


class Elf:
    def __init__(self, path):
        self.path = path
        self.data = open(path, "rb").read()
        d = self.data
        if d[:4] != b"\x7fELF" or d[4] != 1 or d[5] != 1:
            raise ValueError(f"{path}: not a 32-bit little-endian ELF")
        shoff, = struct.unpack_from("<I", d, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x2E)
        raw = [struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize) for i in range(shnum)]
        strtab = raw[shstrndx]

        def name_at(tab, off):
            s = tab[4] + off
            return d[s:d.index(b"\0", s)].decode()
        self.sections = []
        for r in raw:
            self.sections.append(dict(name=name_at(strtab, r[0]), type=r[1], flags=r[2], addr=r[3],
                                      offset=r[4], size=r[5], link=r[6], entsize=r[9]))
        self.symbols = {}
        for s in self.sections:
            if s["type"] != SHT_SYMTAB:
                continue
            st = raw[s["link"]]
            for i in range(s["size"] // 16):
                nm, val, size, info, other, shndx = struct.unpack_from("<IIIBBH", d, s["offset"] + 16 * i)
                if nm:
                    self.symbols.setdefault(name_at(st, nm), val)

    def section(self, name):
        for s in self.sections:
            if s["name"] == name:
                return s
        return None

    def init_words(self):
        """Initial 32-bit words of allocated, non-executable sections with contents."""
        out = {}
        for s in self.sections:
            if s["type"] == SHT_PROGBITS and s["flags"] & SHF_ALLOC and not s["flags"] & SHF_EXEC:
                for o in range(0, s["size"] - 3, 4):
                    out[s["addr"] + o], = struct.unpack_from("<I", self.data, s["offset"] + o)
        return out

    def load_md5(self):
        """md5 over the contents of all allocated sections (code and data, no debug info)."""
        h = hashlib.md5()
        for s in self.sections:
            if s["flags"] & SHF_ALLOC and s["type"] != SHT_NOBITS:
                h.update(s["name"].encode())
                h.update(self.data[s["offset"]:s["offset"] + s["size"]])
        return h.hexdigest()


# --------------------------------------------------------------------------------------------
# elfray listing parser
# --------------------------------------------------------------------------------------------
LINE_RE = re.compile(r"^\s+0x([0-9a-f]+)\s+(\S+)(?:\t(.*))?$")
TARGET_RE = re.compile(r"(?:^|,)([0-9a-f]+) <(.*)>$")
MEM_RE = re.compile(r"^(-?\w+)\((\w+)\)$")
ISSUE_RE = re.compile(r"; tensix issue (op 0x[0-9a-fA-F]+|[A-Za-z0-9_?]+)")


class Insn:
    __slots__ = ("idx", "line", "addr", "mnem", "ops", "target", "note", "func", "raw",
                 "vin", "reached")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def parse_listing(text):
    """Returns (lines, funcs); funcs = list of (name, [Insn])."""
    lines = text.splitlines()
    funcs, cur, idx = [], None, 0
    for n, raw in enumerate(lines, 1):
        m = LINE_RE.match(raw)
        if not m:
            if n > 3 and raw.strip():
                cur = (raw.strip(), [])
                funcs.append(cur)
            continue
        rest = m.group(3) or ""
        note = ""
        if "   ; " in rest:
            rest, note = rest.split("   ; ", 1)
        if " # " in rest:
            rest = rest.split(" # ", 1)[0]
        target = None
        tm = TARGET_RE.search(rest)
        if tm:
            target = int(tm.group(1), 16)
            rest = rest[:tm.start()]
        ops = [o.strip() for o in rest.split(",") if o.strip()]
        cur[1].append(Insn(idx=idx, line=n, addr=int(m.group(1), 16), mnem=m.group(2), ops=ops,
                           target=target, note=note, func=cur[0], raw=raw))
        idx += 1
    return lines, funcs


def count_buffer_stores_linear(lines):
    """Harness rule (elfray_tt_tti.py count_buffer_stores): line numbers of buffer stores found by
    tracking registers that hold 0xffe40000 in address order, ignoring control flow."""
    ptr, hits = set(), set()
    for n, line in enumerate(lines, 1):
        if not LINE_RE.match(line):
            if line.strip().endswith(":"):
                ptr = set()
            continue
        body = line.split(";", 1)[0]
        parts = body.split(None, 2)
        mnem = parts[1]
        ops = [o.strip() for o in (parts[2].split("#", 1)[0] if len(parts) > 2 else "").split(",")]
        if mnem in ("sw", "sh", "sb"):
            m = re.match(r"(-?\w*)\((\w+)\)", ops[-1])
            if (m and m.group(1) in ("", "0") and m.group(2) in ptr) or "<__instrn_buffer>" in line:
                hits.add(n)
            continue
        if not ops or not ops[0]:
            continue
        dst = ops[0]
        if mnem == "lui" and len(ops) > 1 and ops[1] == "0xffe40":
            ptr.add(dst)
        elif mnem == "mv" and len(ops) > 1 and ops[1] in ptr:
            ptr.add(dst)
        elif "<__instrn_buffer>" in line and mnem in ("addi", "lw"):
            ptr.add(dst)
        elif not mnem.startswith(("tt", "b", "j")) and mnem not in ("ret", "fence"):
            ptr.discard(dst)
    return hits


# --------------------------------------------------------------------------------------------
# Abstract values: NAC, or frozenset of items
#   ('c', lo, hi)  unsigned interval, lo <= hi
#   ('s', off)     sp at function entry + off (off None: somewhere in the stack)
#   ('f', op)      top byte = op, low 24 bits run-time (assumption, see module docstring)
# --------------------------------------------------------------------------------------------
NAC = "NAC"


def const(c):
    c &= M32
    return frozenset([("c", c, c)])


def rng(lo, hi):
    return frozenset([("c", lo, hi)])


ZERO_ONE = rng(0, 1)


def s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def single(av):
    """The constant if av is exactly one value."""
    if av is not NAC and len(av) == 1:
        (it,) = av
        if it[0] == "c" and it[1] == it[2]:
            return it[1]
    return None


def top_byte_of(av):
    """Common top byte of every item, else None."""
    if av is NAC:
        return None
    tops = set()
    for it in av:
        if it[0] == "c":
            tops.update(range(it[1] >> 24, (it[2] >> 24) + 1))
        elif it[0] == "f":
            tops.add(it[1])
        else:
            return None
        if len(tops) > 1:
            return None
    return tops.pop() if tops else None


def widen(items):
    items = frozenset(items)
    if len(items) <= LIMIT:
        return items
    if all(it[0] == "s" for it in items):
        return frozenset([("s", None)])
    if all(it[0] == "c" for it in items):
        lo = min(it[1] for it in items)
        hi = max(it[2] for it in items)
        return rng(lo, hi)
    t = top_byte_of(items)
    if t and t != 0xFF:          # opcode words with run-time parts, all with the same opcode byte
        return frozenset([("f", t)])
    return NAC


def hard_widen(av):
    """Used on loop heads after many updates, to guarantee termination."""
    if av is NAC:
        return NAC
    if all(it[0] == "s" for it in av):
        return frozenset([("s", None)]) if len(av) > 1 else av
    t = top_byte_of(av)
    if t:
        return frozenset([("f", t)])
    return NAC


def join_av(a, b):
    if a is NAC or b is NAC:
        return NAC
    if a == b:
        return a
    return widen(a | b)


def _int_add(lo1, hi1, lo2, hi2):
    lo, hi = lo1 + lo2, hi1 + hi2
    if lo >> 32 == hi >> 32:
        return ("c", lo & M32, hi & M32)
    return None


def add_av(a, b):
    if a is NAC and b is NAC:
        return NAC
    if a is NAC or b is NAC:
        x = b if a is NAC else a
        return _with_runtime(x)
    out = set()
    for ia in a:
        for ib in b:
            r = _add_item(ia, ib)
            if r is None:
                return NAC
            out.add(r)
    return widen(out)


def _add_item(ia, ib):
    ka, kb = ia[0], ib[0]
    if ka == "c" and kb == "c":
        if ia[1] == ia[2] and ib[1] == ib[2]:
            v = (ia[1] + ib[1]) & M32
            return ("c", v, v)
        r = _int_add(ia[1], ia[2], ib[1], ib[2])
        # one side a signed constant (e.g. addi -4): treat as signed offset
        for x, y in ((ia, ib), (ib, ia)):
            if y[1] == y[2] and y[1] & 0x80000000:
                k = s32(y[1])
                lo, hi = x[1] + k, x[2] + k
                r = ("c", lo & M32, hi & M32) if lo >> 32 == hi >> 32 else None
                break
        if r is not None and r[1] >> 24 != r[2] >> 24:
            # opcode word + bounded run-time part whose carry could reach the top byte:
            # same assumption as for an unbounded run-time part (fields stay below bit 24)
            for x, y in ((ia, ib), (ib, ia)):
                if x[1] == x[2] and 0 < x[1] >> 24 < 0xFF and y[2] < (1 << 24):
                    return ("f", x[1] >> 24)
        return r
    if ka == "s" and kb == "c" and ib[1] == ib[2]:
        return ("s", None if ia[1] is None else ia[1] + s32(ib[1]))
    if kb == "s" and ka == "c" and ia[1] == ia[2]:
        return ("s", None if ib[1] is None else ib[1] + s32(ia[1]))
    # opcode word + run-time part, plus a constant of magnitude < 2^24 (also negative)
    if ka == "f" and kb == "c" and (ib[2] < (1 << 24) or (ib[1] == ib[2] and s32(ib[1]) > -(1 << 24))):
        return ia
    if kb == "f" and ka == "c" and (ia[2] < (1 << 24) or (ia[1] == ia[2] and s32(ia[1]) > -(1 << 24))):
        return ib
    return None


def _with_runtime(x):
    """x combined (add / or) with a run-time value: opcode-only if x is an opcode word."""
    if x is NAC:
        return NAC
    t = top_byte_of(x)
    if t and t != 0xFF:          # 0xFF..: negative immediates and high addresses, not opcode words
        return frozenset([("f", t)])
    return NAC


def or_av(a, b):
    if a is NAC or b is NAC:
        x = b if a is NAC else a
        return _with_runtime(x) if (a is NAC) != (b is NAC) else NAC
    out = set()
    for ia in a:
        for ib in b:
            if ia[0] == "c" and ib[0] == "c":
                if ia[1] == ia[2] and ib[1] == ib[2]:
                    v = ia[1] | ib[1]
                    out.add(("c", v, v))
                else:
                    # max(a, b) <= a | b <= a + b, and a | b < 2^bitlen(max)
                    hi = min(M32, ia[2] + ib[2], (1 << max(ia[2], ib[2]).bit_length()) - 1)
                    for x, y in ((ia, ib), (ib, ia)):
                        if x[1] == x[2]:
                            hi = min(hi, x[1] | ((1 << y[2].bit_length()) - 1),
                                     y[2] | ((1 << x[1].bit_length()) - 1))
                    out.add(("c", max(ia[1], ib[1]), hi))
            elif ia[0] == "f" and ib[0] == "c" and ib[2] < (1 << 24):
                out.add(ia)
            elif ib[0] == "f" and ia[0] == "c" and ia[2] < (1 << 24):
                out.add(ib)
            else:
                return NAC
    return widen(out)


def and_av(a, b):
    ma, mb = single(a), single(b)
    if ma is not None and mb is not None:
        return const(ma & mb)
    for x, m in ((a, mb), (b, ma)):
        if m is not None:
            hi = m
            if x is not NAC and all(it[0] == "c" for it in x):
                hi = min(m, max(it[2] for it in x))
            return rng(0, hi)
    if a is not NAC and b is not NAC and all(it[0] == "c" for it in a | b):
        return rng(0, min(max(it[2] for it in a), max(it[2] for it in b)))
    return NAC


def map_c(a, f):
    """Apply f to every constant of a small value set (constants only)."""
    if a is NAC:
        return NAC
    out = set()
    for it in a:
        if it[0] != "c" or it[2] - it[1] > LIMIT:
            return None
        for v in range(it[1], it[2] + 1):
            r = f(v)
            if r is None:
                return NAC
            out.add(("c", r & M32, r & M32))
            if len(out) > LIMIT:
                return None
    return frozenset(out)


def lift2(f, a, b):
    if a is NAC or b is NAC:
        return NAC
    out = set()
    for ia in a:
        for ib in b:
            if ia[0] != "c" or ib[0] != "c" or ia[1] != ia[2] or ib[1] != ib[2]:
                return None
            r = f(ia[1], ib[1])
            if r is None:
                return NAC
            out.add(("c", r & M32, r & M32))
    return widen(out)


def all_c(a):
    return a is not NAC and all(it[0] == "c" for it in a)


def hull(a):
    return min(it[1] for it in a), max(it[2] for it in a)


def sll_av(a, k):
    r = lift2(lambda x, _y: x << k, a, const(0))
    if r is not None:
        return r
    if all_c(a):
        lo, hi = hull(a)
        if hi << k <= M32:
            return rng(lo << k, hi << k)
    return NAC


def srl_av(a, k):
    r = lift2(lambda x, _y: x >> k, a, const(0))
    if r is not None:
        return r
    if all_c(a):
        lo, hi = hull(a)
        return rng(lo >> k, hi >> k)
    return rng(0, M32 >> k)


def sub_av(a, b):
    r = lift2(lambda x, y: x - y, a, b)
    if r is not None:
        return r
    if a is not NAC and b is not NAC:
        sb = single(b)
        if sb is not None:
            return add_av(a, const(-sb))
        if all_c(a) and all_c(b):
            lo1, hi1 = hull(a)
            lo2, hi2 = hull(b)
            if lo1 - hi2 >= 0:
                return rng(lo1 - hi2, hi1 - lo2)
    return NAC


def mul_av(a, b):
    r = lift2(lambda x, y: x * y, a, b)
    if r is not None:
        return r
    if all_c(a) and all_c(b):
        lo1, hi1 = hull(a)
        lo2, hi2 = hull(b)
        if hi1 * hi2 <= M32:
            return rng(lo1 * lo2, hi1 * hi2)
    return NAC


def generic2(f, a, b):
    r = lift2(f, a, b)
    return NAC if r is None else r


def generic1(f, a):
    r = map_c(a, f)
    return NAC if r is None else r


# --------------------------------------------------------------------------------------------
# Dataflow analysis
# --------------------------------------------------------------------------------------------
CALLER_SAVED = ["ra"] + [f"t{i}" for i in range(7)] + [f"a{i}" for i in range(8)]
ARGS = [f"a{i}" for i in range(8)]
BRANCH2 = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "bgt", "ble", "bgtu", "bleu"}
BRANCH1 = {"beqz", "bnez", "blez", "bgez", "bltz", "bgtz"}
LOADS = {"lw": 4, "lh": 2, "lhu": 2, "lb": 1, "lbu": 1}
STORES = {"sw": 4, "sh": 2, "sb": 1}
NO_DEST = {"fence", "fence.i", "ebreak", "ecall", "nop", "wfi", "unimp", "c.unimp", ".insn", ".word"}

CMP2 = {"slt": lambda x, y: int(s32(x) < s32(y)), "sltu": lambda x, y: int(x < y)}
ALU2 = {
    "xor": lambda x, y: x ^ y, "sra": lambda x, y: s32(x) >> (y & 31),
    "divu": lambda x, y: (x // y) if y else M32, "remu": lambda x, y: (x % y) if y else x,
    "div": lambda x, y: int(s32(x) / s32(y)) if y else M32,
    "rem": lambda x, y: s32(x) - int(s32(x) / s32(y)) * s32(y) if y else x,
    "mulhu": lambda x, y: (x * y) >> 32, "minu": min, "maxu": max,
    "min": lambda x, y: x if s32(x) < s32(y) else y, "max": lambda x, y: x if s32(x) > s32(y) else y,
    "andn": lambda x, y: x & ~y, "orn": lambda x, y: x | ~y, "xnor": lambda x, y: ~(x ^ y),
    "bclr": lambda x, y: x & ~(1 << (y & 31)), "bset": lambda x, y: x | (1 << (y & 31)),
    "binv": lambda x, y: x ^ (1 << (y & 31)), "bext": lambda x, y: (x >> (y & 31)) & 1,
    "rol": lambda x, y: ((x << (y & 31)) | (x >> (32 - (y & 31)))) if y & 31 else x,
    "ror": lambda x, y: ((x >> (y & 31)) | (x << (32 - (y & 31)))) if y & 31 else x,
    "czero.eqz": lambda x, y: 0 if y == 0 else x, "czero.nez": lambda x, y: 0 if y != 0 else x,
}
ALU1 = {
    "neg": lambda x: -x, "not": lambda x: ~x,
    "sext.b": lambda x: s32((x & 0xFF) << 24) >> 24, "sext.h": lambda x: s32((x & 0xFFFF) << 16) >> 16,
    "rev8": lambda x: int.from_bytes((x & M32).to_bytes(4, "little"), "big"),
}
CMP1 = {"seqz", "snez", "sltz", "sgtz"}
IMM_MAP = {"addi": "add", "andi": "and", "ori": "or", "xori": "xor", "slli": "sll", "srli": "srl",
           "srai": "sra", "slti": "slt", "sltiu": "sltu", "bclri": "bclr", "bseti": "bset",
           "binvi": "binv", "bexti": "bext", "rori": "ror"}


class State:
    __slots__ = ("v", "slots")

    def __init__(self, v, slots):
        self.v, self.slots = v, slots

    def copy(self):
        return State(dict(self.v), dict(self.slots))

    def get(self, r):
        if r == "zero":
            return const(0)
        return self.v.get(r, NAC)


def join_state(a, b, hard=False):
    if a is None:
        return b.copy()
    v = {}
    for r in set(a.v) | set(b.v):
        x, y = a.v.get(r, NAC), b.v.get(r, NAC)
        j = join_av(x, y)
        if hard and j != x:
            j = hard_widen(j)
        v[r] = j
    slots = {}
    for k in set(a.slots) & set(b.slots):
        x = a.slots[k]
        j = join_av(x, b.slots[k])
        if hard and j != x:
            j = hard_widen(j)
        slots[k] = j
    return State(v, slots)


def sp_off(av):
    if av is not NAC and len(av) == 1:
        (it,) = av
        if it[0] == "s":
            return it[1]
    return None


def has_sp(av):
    return av is not NAC and any(it[0] == "s" for it in av)


def rebase(av, k):
    """Caller value -> callee view: stack offsets relative to the callee's entry sp."""
    if av is NAC or not has_sp(av):
        return av
    if k is None:
        return NAC
    return frozenset(it if it[0] != "s" or it[1] is None else ("s", it[1] - k) for it in av)


class Ctx:
    def __init__(self, gp, data):
        self.gp = gp
        self.data = data          # addr -> initial word (trusted)
        self.unknown = set()


def mem_addr(st, op):
    mm = MEM_RE.match(op)
    return add_av(st.get(mm.group(2)), const(int(mm.group(1), 0)))


def transfer(ins, st, ctx, record=False):
    """Applies ins to st in place. Returns ('call', target, args) for direct calls."""
    m, ops = ins.mnem, ins.ops
    if record:
        ins.vin = {}
        for o in ops:
            mm = MEM_RE.match(o)
            r = mm.group(2) if mm else o
            if re.match(r"^(zero|ra|sp|gp|tp|[ast]\d+)$", r):
                ins.vin[r] = st.get(r)
        ins.reached = True

    def setr(r, av):
        if r != "zero":
            st.v[r] = av

    if m in STORES:
        a = mem_addr(st, ops[1])
        off = sp_off(a)
        if off is not None:
            k = off & ~3
            st.slots[k] = st.get(ops[0]) if (m == "sw" and off == k) else NAC
        elif has_sp(a):
            st.slots.clear()
        return None
    if m in LOADS:
        a = mem_addr(st, ops[1])
        off = sp_off(a)
        val = NAC
        if off is not None and m == "lw":
            val = st.slots.get(off, NAC)
        elif m == "lw" and single(a) is not None and single(a) in ctx.data:
            val = const(ctx.data[single(a)])
        elif m == "lbu":
            val = rng(0, 0xFF)
        elif m == "lhu":
            val = rng(0, 0xFFFF)
        setr(ops[0], val)
        return None
    if m.startswith("tt") or m in NO_DEST or m in BRANCH1 or m in BRANCH2 or m in ("j", "ret", "jr"):
        return None
    if m in ("jal", "jalr"):
        args = {r: rebase(st.get(r), sp_off(st.get("sp"))) for r in ARGS}
        if any(has_sp(st.get(r)) for r in ARGS):
            st.slots.clear()
        for r in CALLER_SAVED:
            st.v[r] = NAC
        if m == "jal" and len(ops) <= 1 and ins.target is not None:
            return ("call", ins.target, args)
        ctx.unknown.add("indirect call " + m)
        return None
    if not ops:
        ctx.unknown.add(m)
        return None
    d = ops[0]
    g = st.get
    if m == "lui":
        setr(d, const(int(ops[1], 0) << 12))
    elif m == "auipc":
        setr(d, const(ins.addr + (int(ops[1], 0) << 12)))
    elif m == "li":
        setr(d, const(int(ops[1], 0)))
    elif m == "mv":
        setr(d, g(ops[1]))
    elif m in IMM_MAP or m in ("add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu",
                               "mul") or m in ALU2 or m in ("sh1add", "sh2add", "sh3add"):
        base = IMM_MAP.get(m, m)
        if m in IMM_MAP:
            x, y = g(ops[1]), const(int(ops[2], 0))
            yk = int(ops[2], 0)
        else:
            x, y = g(ops[1]), g(ops[2])
            yk = single(y)
        if base == "add":
            setr(d, add_av(x, y))
        elif base in ("sh1add", "sh2add", "sh3add"):
            setr(d, add_av(sll_av(x, int(base[2])), y))
        elif base == "sub":
            setr(d, sub_av(x, y))
        elif base == "and":
            setr(d, and_av(x, y))
        elif base == "or":
            setr(d, or_av(x, y))
        elif base == "sll":
            setr(d, sll_av(x, yk & 31) if yk is not None else NAC)
        elif base == "srl":
            setr(d, srl_av(x, yk & 31) if yk is not None else rng(0, M32))
        elif base == "mul":
            setr(d, mul_av(x, y))
        elif base in CMP2:
            r = lift2(CMP2[base], x, y)
            setr(d, ZERO_ONE if r is None or r is NAC else r)
        elif base in ALU2:
            setr(d, generic2(ALU2[base], x, y))
        else:
            setr(d, NAC)
    elif m in CMP1:
        f = {"seqz": lambda x: int(x == 0), "snez": lambda x: int(x != 0),
             "sltz": lambda x: int(s32(x) < 0), "sgtz": lambda x: int(s32(x) > 0)}[m]
        r = map_c(g(ops[1]), f)
        setr(d, ZERO_ONE if r is None or r is NAC else r)
    elif m == "zext.b":
        setr(d, and_av(g(ops[1]), const(0xFF)))
    elif m == "zext.h":
        setr(d, and_av(g(ops[1]), const(0xFFFF)))
    elif m in ALU1:
        setr(d, generic1(ALU1[m], g(ops[1])))
    elif m in ("clz", "ctz", "cpop"):
        setr(d, rng(0, 32))
    else:
        setr(d, NAC)
        ctx.unknown.add(m)
    return None


def blocks_of(insns):
    pos = {ins.addr: i for i, ins in enumerate(insns)}
    leaders = {0}
    for i, ins in enumerate(insns):
        m = ins.mnem
        if m in BRANCH1 or m in BRANCH2 or m in ("j", "ret", "jr"):
            if i + 1 < len(insns):
                leaders.add(i + 1)
            if ins.target in pos:
                leaders.add(pos[ins.target])
    starts = sorted(leaders)
    bid, blocks = {}, []
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else len(insns)
        blocks.append(list(range(s, e)))
        bid[s] = b
    succ = []
    for b, idxs in enumerate(blocks):
        last = insns[idxs[-1]]
        m = last.mnem
        s = []
        if m in BRANCH1 or m in BRANCH2:
            if last.target in pos:
                s.append(bid[pos[last.target]])
            if b + 1 < len(blocks):
                s.append(b + 1)
        elif m == "j":
            if last.target in pos:
                s.append(bid[pos[last.target]])
        elif m in ("ret", "jr"):
            pass
        elif b + 1 < len(blocks):
            s.append(b + 1)
        succ.append(s)
    return blocks, succ


def entry_state(args, gp):
    v = {"sp": frozenset([("s", 0)])}
    if gp is not None:
        v["gp"] = const(gp)
    for r in ARGS:
        v[r] = args.get(r, NAC) if args else NAC
    return State(v, {})


def run_dataflow(funcs, ctx):
    by_addr = {f[1][0].addr: f for f in funcs if f[1]}
    callers = {f[0]: {} for f in funcs}
    for _round in range(10):
        changed = False
        for name, insns in funcs:
            if not insns:
                continue
            for ins in insns:
                ins.vin, ins.reached = None, False
            blocks, succ = blocks_of(insns)
            cs = callers[name]
            args = None
            if cs:
                args = {}
                for r in ARGS:
                    av = None
                    for a in cs.values():
                        av = a[r] if av is None else join_av(av, a[r])
                    args[r] = av
            ins_state = [None] * len(blocks)
            visits = [0] * len(blocks)
            ins_state[0] = entry_state(args, ctx.gp)
            work = [0]
            while work:
                b = work.pop()
                st = ins_state[b].copy()
                for i in blocks[b]:
                    transfer(insns[i], st, ctx)
                for s in succ[b]:
                    visits[s] += 1
                    new = join_state(ins_state[s], st, hard=visits[s] > WIDEN_AFTER)
                    old = ins_state[s]
                    if old is None or new.v != old.v or new.slots != old.slots:
                        ins_state[s] = new
                        if s not in work:
                            work.append(s)
            for b in range(len(blocks)):
                if ins_state[b] is None:
                    continue
                st = ins_state[b].copy()
                for i in blocks[b]:
                    ins = insns[i]
                    r = transfer(ins, st, ctx, record=True)
                    if r and r[0] == "call" and r[1] in by_addr:
                        callee = by_addr[r[1]][0]
                        if callers[callee].get(ins.idx) != r[2]:
                            callers[callee][ins.idx] = r[2]
                            changed = True
                last = insns[blocks[b][-1]]
                if last.mnem == "j" and last.target in by_addr and last.target != insns[0].addr:
                    callee = by_addr[last.target][0]
                    a = {r: rebase(st.get(r), sp_off(st.get("sp"))) for r in ARGS}
                    if callers[callee].get(last.idx) != a:
                        callers[callee][last.idx] = a
                        changed = True
        if not changed:
            break


def classify_addr(av):
    """'must' (buffer on every path), 'may' (some paths), 'no', or 'unknown'."""
    if av is NAC:
        return "unknown"
    hits = []
    for it in av:
        if it[0] == "s":
            hits.append(False)
        elif it[0] == "f":
            return "unknown"
        else:
            if BUF_LO <= it[1] and it[2] <= BUF_HI:
                hits.append(True)
            elif it[2] < BUF_LO or it[1] > BUF_HI:
                hits.append(False)
            else:
                return "may"
    if all(hits):
        return "must"
    return "may" if any(hits) else "no"


def value_opcodes(av):
    """(sorted opcode list or None, 'proven' | 'assumed' | 'several' | 'unknown').
    'several': different paths store different opcodes (each one known)."""
    if av is NAC or any(it[0] == "s" for it in av):
        return None, "unknown"
    tops = set()
    for it in av:
        if it[0] == "f":
            tops.add(it[1])
        elif it[1] >> 24 == it[2] >> 24:
            tops.add(it[1] >> 24)
        else:
            return None, "unknown"
    if len(tops) > 1:
        return sorted(tops), "several"
    return sorted(tops), ("assumed" if any(it[0] == "f" for it in av) else "proven")


def fmt_av(av):
    if av is NAC:
        return "?"
    out = []
    for it in sorted(av, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
        if it[0] == "c":
            out.append(hex(it[1]) if it[1] == it[2] else f"{it[1]:#x}..{it[2]:#x}")
        elif it[0] == "s":
            out.append("sp+?" if it[1] is None else f"sp{it[1]:+d}")
        else:
            out.append(f"op{it[1]:#04x}+rt")
    return "{" + ",".join(out) + "}"


def analyze_listing(text, elf):
    """Runs the dataflow analysis; returns (lines, funcs, info)."""
    lines, funcs = parse_listing(text)
    gp = elf.symbols.get("__global_pointer$")
    data = elf.init_words()
    excluded = set()
    for _ in range(5):
        ctx = Ctx(gp, {a: v for a, v in data.items() if a not in excluded})
        run_dataflow(funcs, ctx)
        written = set()
        for _n, insns in funcs:
            for ins in insns:
                if ins.mnem in STORES and ins.reached:
                    a = mem_addr_rec(ins)
                    if a is NAC:
                        continue
                    for it in a:
                        if it[0] == "c":
                            for w in data:
                                if it[1] - 3 <= w <= it[2]:
                                    written.add(w)
        if written <= excluded:
            break
        excluded |= written
    return lines, funcs, {"unmodelled": sorted(ctx.unknown), "gp": gp,
                          "data_words_trusted": len(data) - len(excluded)}


def mem_addr_rec(ins):
    mm = MEM_RE.match(ins.ops[1])
    base = ins.vin.get(mm.group(2), NAC) if mm.group(2) != "zero" else const(0)
    return add_av(base, const(int(mm.group(1), 0)))


def store_value_rec(ins):
    r = ins.ops[0]
    return const(0) if r == "zero" else ins.vin.get(r, NAC)


# --------------------------------------------------------------------------------------------
# Opcode names (from the run's objdump: decode every opcode byte as a .ttinsn word)
# --------------------------------------------------------------------------------------------
def opcode_names(objdump, elf_path):
    """{opcode byte: ISA name} by disassembling a patched copy of an ELF whose .text holds
    rotl2(op << 24) for every op (the .ttinsn encoding used by the toolchain)."""
    elf = Elf(elf_path)
    text = elf.section(".text")
    names = {}
    per = text["size"] // 4
    tmp = tempfile.mkdtemp(prefix="tt_tti_ops_")
    try:
        for start in range(0, 256, per):
            buf = bytearray(elf.data)
            ops = list(range(start, min(256, start + per)))
            for i, op in enumerate(ops):
                w = op << 24
                enc = ((w << 2) | (w >> 30)) & M32
                struct.pack_into("<I", buf, text["offset"] + 4 * i, enc)
            p = os.path.join(tmp, "ops.elf")
            open(p, "wb").write(buf)
            out = subprocess.run([objdump, "-d", "-j", ".text", p], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and parts[0].strip().endswith(":"):
                    try:
                        a = int(parts[0].strip()[:-1], 16)
                    except ValueError:
                        continue
                    i = (a - text["addr"]) // 4
                    if 0 <= i < len(ops) and (a - text["addr"]) % 4 == 0:
                        mn = parts[2].split()[0] if parts[2].strip() else ""
                        if mn.startswith("tt"):
                            names[ops[i]] = mn[2:].upper()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return names


def op_name(names, op):
    return names.get(op, f"OP_{op:#04x}")


# --------------------------------------------------------------------------------------------
# Per-ELF analysis
# --------------------------------------------------------------------------------------------
def elfray_listing(elfray, objdump, elf_path):
    env = dict(os.environ, ELFRAY_OBJDUMP=objdump)
    return subprocess.run([elfray, "--section", ".text", "--no-color", elf_path], env=env,
                          capture_output=True, text=True, check=True).stdout


def analyze_elf(elf_path, elfray, objdump, names):
    elf = Elf(elf_path)
    text = elfray_listing(elfray, objdump, elf_path)
    lines, funcs, info = analyze_listing(text, elf)
    linear = count_buffer_stores_linear(lines)
    res = collections.Counter()
    tti_by, tt_by = collections.Counter(), collections.Counter()
    elfray_by = collections.Counter()
    stores, insn_rows = [], []
    for fname, insns in funcs:
        for ins in insns:
            row = {"addr": ins.addr, "func": fname, "mnem": ins.mnem, "target": ins.target,
                   "tti": False, "tt": False, "op": None}
            res["insns"] += 1
            if ins.mnem.startswith("tt"):
                res["tti"] += 1
                tti_by[ins.mnem[2:].upper()] += 1
                row["tti"] = True
                row["op"] = ins.mnem[2:].upper()
            elif ins.mnem in (".insn", ".word"):
                res["insn_word"] += 1
            mk = ISSUE_RE.search(ins.raw)
            if mk:
                res["marks_any_line"] += 1
            if ins.mnem not in STORES:
                insn_rows.append(row)
                continue
            res["stores"] += 1
            marked = bool(mk) and ins.mnem in STORES
            res["tt_elfray"] += marked
            ename = ""
            if marked:
                ename = mk.group(1)
                if ename.startswith("op "):
                    ename = f"OP_{int(ename[3:], 16):#04x}"
                elfray_by[ename] += 1
            lin = ins.line in linear
            res["tt_linear"] += lin
            if not ins.reached:
                cls, addr, val = "unreachable", NAC, NAC
            else:
                addr, val = mem_addr_rec(ins), store_value_rec(ins)
                cls = classify_addr(addr)
            op, opst = (None, "")
            if cls == "must":
                res["tt_all_paths"] += 1
                ops_, opst = value_opcodes(val)
                op = "|".join(op_name(names, o) for o in ops_) if ops_ else None
                res["tt_op_" + opst] += 1
                tt_by[op or "?"] += 1
                if ins.mnem != "sw":
                    res["tt_not_sw"] += 1
                if marked and op is not None and ename not in op.split("|"):
                    res["tt_op_differs_from_elfray"] += 1
                row["tt"] = True
                row["op"] = op or "?"
            elif cls in ("may", "unknown"):
                res["unresolved"] += 1
            elif cls == "unreachable":
                res["unreachable_stores"] += 1
            if (marked or lin) and cls != "must":
                res["marked_or_linear_not_proven"] += 1
            insn_rows.append(row)
            if cls != "no" or marked or lin:
                stores.append(dict(line=ins.line, addr=hex(ins.addr), func=fname, insn=f"{ins.mnem} {','.join(ins.ops)}",
                                   cls=cls, elfray=ename, linear=lin,
                                   opcode=op or "", opcode_status=opst,
                                   addrset=fmt_av(addr), valset=fmt_av(val)))
    tsec = elf.section(".text")
    init = elf.section(".init")
    return {
        "text_bytes": tsec["size"] if tsec else 0,
        "init_bytes": init["size"] if init else 0,
        "tti": res["tti"],
        "tt_elfray": res["tt_elfray"],
        "tt_linear": res["tt_linear"],
        "tt_all_paths": res["tt_all_paths"],
        "unresolved": res["unresolved"],
        "detail": {k: v for k, v in sorted(res.items())
                   if k not in ("tti", "tt_elfray", "tt_linear", "tt_all_paths", "unresolved")},
        "tti_by_opcode": dict(sorted(tti_by.items())),
        "tt_all_paths_by_opcode": dict(sorted(tt_by.items())),
        "tt_elfray_by_opcode": dict(sorted(elfray_by.items())),
        "load_md5": elf.load_md5(),
        "analysis": info,
    }, stores, insn_rows


# --------------------------------------------------------------------------------------------
# Runs, builds, pairing
# --------------------------------------------------------------------------------------------
PARAM_RE = re.compile(r"^\s*constexpr\s+(?:[\w:<>]+\s+)?(\w+)\s*=\s*([^;]+);")
DEFINE_RE = re.compile(r"^\s*#define\s+(\w+)(?:\s+(.*))?$")
SKIP_DEFINES = {"RUNTIME_PARAMETERS"}


def build_params(path):
    p = {}
    if not os.path.exists(path):
        return p
    for line in open(path):
        m = PARAM_RE.match(line)
        if m:
            p[m.group(1)] = re.sub(r"\b\w+::", "", m.group(2).strip())
            continue
        m = DEFINE_RE.match(line)
        if m and m.group(1) not in SKIP_DEFINES:
            p[m.group(1)] = (m.group(2) or "").strip()
    return p


def build_label(p):
    keep = []
    for k, v in sorted(p.items()):
        keep.append(f"{k}={v}")
    return "; ".join(keep)


def short_label(p, varying):
    return " ".join(f"{k}={p.get(k, '-')}" for k in varying)


def find_tool(run, name, override):
    if override:
        return override
    p = os.path.join(run, "ws", "tt_metal", "tt-llk", "tests", "sfpi", "compiler", "bin", name)
    return p if os.path.exists(p) else shutil.which(name)


def analyze_run(run, args, log):
    objdump = find_tool(run, "riscv-tt-elf-objdump", args.objdump or os.environ.get("ELFRAY_OBJDUMP"))
    if not objdump:
        raise SystemExit(f"{run}: no objdump found; pass --objdump")
    src = os.path.join(run, "codesize_tmp", "tt-llk-build", "sources")
    tests = {"original": args.orig, "hal": args.hal}
    sides = {}
    all_elfs = []
    for side in SIDES:
        base = os.path.join(src, tests[side])
        variants = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))) \
            if os.path.isdir(base) else []
        sides[side] = [(v, build_params(os.path.join(base, v, "build.h"))) for v in variants]
        all_elfs += [os.path.join(base, v, "elf", f"{k}.elf") for v in variants for k in KERNELS]
    all_elfs = [e for e in all_elfs if os.path.exists(e)]
    biggest = max(all_elfs, key=lambda e: Elf(e).section(".text")["size"])
    names = opcode_names(objdump, biggest)
    # parameters that differ between builds
    allp = [p for side in SIDES for _v, p in sides[side]]
    keys = sorted({k for p in allp for k in p})
    varying = [k for k in keys if len({p.get(k) for p in allp}) > 1]
    out = {"run_dir": os.path.abspath(run), "label": os.path.basename(os.path.normpath(run)),
           "objdump": objdump, "tests": tests, "varying_parameters": varying,
           "opcode_names": {f"{k:#04x}": v for k, v in sorted(names.items())}, "sides": {}}
    store_rows, insn_rows = [], {}
    for side in SIDES:
        builds = []
        for v, p in sides[side]:
            b = {"variant": v, "params": p, "key": build_label(p), "label": short_label(p, varying), "kernels": {}}
            for k in KERNELS:
                elf = os.path.join(src, tests[side], v, "elf", f"{k}.elf")
                if not os.path.exists(elf):
                    continue
                log(f"  {side:8s} {v[:12]} {k}")
                r, stores, rows = analyze_elf(elf, args.elfray, objdump, names)
                b["kernels"][k] = r
                for s in stores:
                    store_rows.append(dict(run=out["label"], side=side, variant=v[:12], kernel=k, **s))
                insn_rows[(side, v, k)] = rows
            builds.append(b)
        out["sides"][side] = {"test": tests[side], "builds": builds}
    # pairing
    pairs = []
    hal_by_key = collections.defaultdict(list)
    for b in out["sides"]["hal"]["builds"]:
        hal_by_key[b["key"]].append(b["variant"])
    for b in out["sides"]["original"]["builds"]:
        h = hal_by_key.get(b["key"], [])
        pairs.append({"original": b["variant"], "hal": h[0] if len(h) == 1 else None, "label": b["label"]})
    out["pairs"] = pairs
    out["summary"] = summarize(out)
    return out, store_rows, insn_rows


COLS = ["text_bytes", "tti", "tt_elfray", "tt_linear", "tt_all_paths", "unresolved"]


def summarize(run):
    s = {}
    for side in SIDES:
        builds = run["sides"][side]["builds"]
        s[side] = {}
        for k in KERNELS + ["sum"]:
            rows = []
            for b in builds:
                if k == "sum":
                    if all(kk in b["kernels"] for kk in KERNELS):
                        rows.append({c: sum(b["kernels"][kk][c] for kk in KERNELS) for c in COLS})
                elif k in b["kernels"]:
                    rows.append(b["kernels"][k])
            n = len(rows)
            s[side][k] = {"builds": n,
                          "avg": {c: (sum(r[c] for r in rows) / n if n else None) for c in COLS},
                          "total": {c: sum(r[c] for r in rows) for c in COLS}}
    return s


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return str(x)


def markdown(runs):
    L = ["# TT / TTI counts (static, per ELF)", "",
         "Columns: text bytes = `.text` size; TTI = `tt*` lines in elfray `.text` output; TT elfray = stores elfray "
         "marks `tensix issue` (lower bound); TT linear = harness linear register tracking; TT all paths = stores whose "
         "address is in `0xFFE40000`-`0xFFE4FFFF` on every path (dataflow analysis); unresolved = stores whose address "
         "could not be settled.", ""]
    names = [r["label"] for r in runs]
    md5s = collections.defaultdict(set)
    for r in runs:
        for side in SIDES:
            for b in r["sides"][side]["builds"]:
                for k, kr in b["kernels"].items():
                    md5s[(side, b["key"], k)].add(kr["load_md5"])
    if len(runs) > 1:
        for side in SIDES:
            same = sum(1 for (s, _b, _k), v in md5s.items() if s == side and len(v) == 1)
            tot = sum(1 for (s, _b, _k) in md5s if s == side)
            L.append(f"- {side}: {same} of {tot} (build, kernel) ELFs have identical loadable sections in all "
                     f"runs that contain them ({', '.join(names)}).")
        L.append("")
    for r in runs:
        L += [f"## {r['label']}", "", f"`{r['run_dir']}`", "",
              f"Builds: original {r['summary']['original']['sum']['builds']}, HAL {r['summary']['hal']['sum']['builds']}. "
              f"Varying build parameters: {', '.join(r['varying_parameters']) or '-'}.", "",
              "### Average per build", "",
              "| Side | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved | TTI + TT all paths |",
              "|---|---|---|---|---|---|---|---|---|"]
        for side in SIDES:
            for k in KERNELS + ["sum"]:
                a = r["summary"][side][k]["avg"]
                tti_tt = None if a["tti"] is None else a["tti"] + a["tt_all_paths"]
                L.append(f"| {side} | {k} | " + " | ".join(fmt(a[c]) for c in COLS) + f" | {fmt(tti_tt)} |")
        L += ["", "### Total over all builds", "",
              "| Side | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved |",
              "|---|---|---|---|---|---|---|---|"]
        for side in SIDES:
            for k in KERNELS + ["sum"]:
                t = r["summary"][side][k]["total"]
                L.append(f"| {side} | {k} | " + " | ".join(fmt(t[c]) for c in COLS) + " |")
        L += ["", "### Per build", "",
              "| Side | Build | Variant | Kernel | text bytes | TTI | TT elfray | TT linear | TT all paths | unresolved |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for side in SIDES:
            for b in r["sides"][side]["builds"]:
                for k in KERNELS:
                    if k in b["kernels"]:
                        kr = b["kernels"][k]
                        L.append(f"| {side} | {b['label']} | `{b['variant'][:12]}` | {k} | "
                                 + " | ".join(fmt(kr[c]) for c in COLS) + " |")
        unp = sorted(r["pairs"], key=lambda p: p["label"])
        L += ["", "Build pairs (same build.h parameters): " +
              "; ".join(f"`{p['original'][:12]}` / `{(p['hal'] or '-')[:12]}`" for p in unp), ""]
        notes = collections.Counter()
        unm = set()
        for side in SIDES:
            for b in r["sides"][side]["builds"]:
                for kr in b["kernels"].values():
                    for kk, v in kr["detail"].items():
                        if kk.startswith(("tt_op_", "tt_not_sw", "marked_or", "unreachable")):
                            notes[f"{side} {kk}"] += v
                    unm.update(kr["analysis"]["unmodelled"])
        L += ["Checks: " + ", ".join(f"{k} = {v}" for k, v in sorted(notes.items())) + ".",
              f"Mnemonics without a value model (result treated as unknown): {', '.join(sorted(unm)) or 'none'}.", ""]
    return "\n".join(L) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--tsv", help="one row per store that is marked, linear, or not provably outside the buffer")
    ap.add_argument("--elfray", default=os.environ.get("ELFRAY", os.path.join(os.path.dirname(here), "elfray")))
    ap.add_argument("--objdump")
    ap.add_argument("--orig", default=DEFAULT_TESTS["original"])
    ap.add_argument("--hal", default=DEFAULT_TESTS["hal"])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    log = (lambda s: None) if a.quiet else (lambda s: print(s, file=sys.stderr))
    runs, stores = [], []
    for run in a.runs:
        log(f"run {run}")
        r, st, _rows = analyze_run(run, a, log)
        runs.append(r)
        stores += st
    md = markdown(runs)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump({"runs": runs}, open(a.json, "w"), indent=1)
    if a.tsv:
        cols = ["run", "side", "variant", "kernel", "line", "addr", "func", "insn", "cls", "elfray", "linear",
                "opcode", "opcode_status", "addrset", "valset"]
        with open(a.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for s in stores:
                f.write("\t".join(str(s[c]) for c in cols) + "\n")
    if not a.md:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
