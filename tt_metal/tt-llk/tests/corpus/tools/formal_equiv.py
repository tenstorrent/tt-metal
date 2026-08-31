#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# laneJO: SMT-based translation validation for SFPU kernels.
#
# Proves (z3 UNSAT) or refutes (z3 SAT + counterexample) BIT-EXACT
# equivalence between the compiled semantic kernel and the hand kernel of a
# corpus board row, over ALL inputs, at the Dst-state level.
#
# Pipeline
#   1. Both legs run once on the PINNED craq-sim (nkapre/sfpi 1c47e9cd) built
#      with the laneJO TTSIM_TRACE_SFPU_STREAM hook (value-observation only):
#      one `SFPUJO I` line per executed SFPU instruction (with the resolved
#      Dst row / mod0 captured pre-execution), one `SFPUJO M` line per retired
#      SFPLOADMACRO event, `SFPUJO C` on deferred-CC application, and a
#      `SFPUJO V` validation snapshot (lanes 0/1 of LReg0..7 + LReg16 + CC)
#      after each of those.
#   2. VALIDATION GATE: this tool re-executes each trace CONCRETELY with its
#      own transcription of the pinned simulator's per-instruction semantics
#      (src/tensix.cpp + src/fma.cpp, TT_VERSION==1/BH) and demands that
#      EVERY intermediate V snapshot reproduces bitwise, for lane 0 and
#      lane 1 independently.  A row whose semantics do not replay is
#      SEMANTICS-UNVALIDATED and gets no verdict.
#   3. SYMBOLIC RUN: the same executor runs over z3 QF_BV terms; every Dst
#      cell first read becomes a fresh 16-bit symbol shared between the two
#      legs.  The final Dst state of each leg is a map cell -> BV expression.
#   4. QUERY: per output cell, assert sem_expr != hand_expr.  UNSAT for every
#      (deduplicated) cell = PROVEN-EQUIV-ALL-INPUTS.  SAT = DIVERGENT with a
#      witness input assignment (to be checked back on the simulator).
#
# The per-opcode semantics below are TRANSCRIBED from the pinned simulator;
# each function cites its source line range (craq-sim @ 1c47e9cd).
# Lane model: lanes are value-independent for the target rows; the executor
# models lanes 0 and 1 exactly and refuses cross-lane instructions
# (SFPTRANSP, SFPSHFT2 modes 1..4) and nondeterminism (PRNG moves,
# stochastic rounding) with named SCOPE refusals.

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict

try:
    import z3
except ImportError:
    z3 = None

DST_ROWS = 1024

# ---------------------------------------------------------------------------
# Backends: concrete (python ints) and symbolic (z3 BV).
# All 32-bit values are backend terms; bools are python bool or z3 BoolRef.
# ---------------------------------------------------------------------------


class Concrete:
    symbolic = False

    @staticmethod
    def const(v, width=32):
        return v & ((1 << width) - 1)

    @staticmethod
    def is_const(v):
        return True

    @staticmethod
    def const_val(v):
        return v

    @staticmethod
    def add(a, b):
        return (a + b) & 0xFFFFFFFF

    @staticmethod
    def sub(a, b):
        return (a - b) & 0xFFFFFFFF

    @staticmethod
    def band(a, b):
        return a & b

    @staticmethod
    def bor(a, b):
        return a | b

    @staticmethod
    def bxor(a, b):
        return a ^ b

    @staticmethod
    def bnot(a):
        return (~a) & 0xFFFFFFFF

    @staticmethod
    def shl(a, s):
        return (a << s) & 0xFFFFFFFF if s < 32 else 0

    @staticmethod
    def lshr(a, s):
        return a >> s if s < 32 else 0

    @staticmethod
    def ashr(a, s):
        sa = a - (1 << 32) if a & 0x80000000 else a
        return (sa >> min(s, 31)) & 0xFFFFFFFF

    # variable-amount shifts (amount is a term, masked by caller)
    shl_v = None  # filled below
    lshr_v = None
    ashr_v = None

    @staticmethod
    def mul64(a, b):
        return (a * b) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def add64(a, b):
        return (a + b) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def shl64(a, s):
        return (a << s) & 0xFFFFFFFFFFFFFFFF if s < 64 else 0

    @staticmethod
    def lshr64(a, s):
        return a >> s if s < 64 else 0

    @staticmethod
    def trunc32(a):
        return a & 0xFFFFFFFF

    @staticmethod
    def eq(a, b):
        return a == b

    @staticmethod
    def ne(a, b):
        return a != b

    @staticmethod
    def ult(a, b):
        return a < b

    @staticmethod
    def ule(a, b):
        return a <= b

    @staticmethod
    def slt(a, b):
        def s(x):
            return x - (1 << 32) if x & 0x80000000 else x

        return s(a) < s(b)

    @staticmethod
    def sle(a, b):
        def s(x):
            return x - (1 << 32) if x & 0x80000000 else x

        return s(a) <= s(b)

    @staticmethod
    def ite(c, a, b):
        return a if c else b

    @staticmethod
    def bool_and(*xs):
        return all(xs)

    @staticmethod
    def bool_or(*xs):
        return any(xs)

    @staticmethod
    def bool_not(x):
        return not x

    @staticmethod
    def clz32(a):
        # count leading zeros of a 32-bit value; a==0 -> 32
        if a == 0:
            return 32
        return 32 - a.bit_length()

    @staticmethod
    def clz32_t(a):
        return Concrete.clz32(a)


def _conc_shl_v(a, s):
    s &= 0xFFFFFFFF
    return (a << s) & 0xFFFFFFFF if s < 32 else 0


def _conc_lshr_v(a, s):
    s &= 0xFFFFFFFF
    return a >> s if s < 32 else 0


def _conc_ashr_v(a, s):
    return Concrete.ashr(a, s & 0xFFFFFFFF if (s & 0xFFFFFFFF) < 32 else 31)


Concrete.shl_v = staticmethod(_conc_shl_v)
Concrete.lshr_v = staticmethod(_conc_lshr_v)
Concrete.ashr_v = staticmethod(_conc_ashr_v)


class Symbolic:
    """z3 QF_BV backend.  32-bit terms are z3 BitVec(32); bools are BoolRef
    or python bool (kept python where provably concrete)."""

    symbolic = True

    @staticmethod
    def const(v, width=32):
        return z3.BitVecVal(v & ((1 << width) - 1), width)

    @staticmethod
    def is_const(v):
        if isinstance(v, int):
            return True
        return z3.is_bv_value(z3.simplify(v)) if z3.is_bv(v) else False

    @staticmethod
    def const_val(v):
        if isinstance(v, int):
            return v
        return z3.simplify(v).as_long()

    @staticmethod
    def _bv(v, width=32):
        if isinstance(v, int):
            return z3.BitVecVal(v & ((1 << width) - 1), width)
        return v

    @classmethod
    def add(cls, a, b):
        return cls._bv(a) + cls._bv(b)

    @classmethod
    def sub(cls, a, b):
        return cls._bv(a) - cls._bv(b)

    @classmethod
    def band(cls, a, b):
        return cls._bv(a) & cls._bv(b)

    @classmethod
    def bor(cls, a, b):
        return cls._bv(a) | cls._bv(b)

    @classmethod
    def bxor(cls, a, b):
        return cls._bv(a) ^ cls._bv(b)

    @classmethod
    def bnot(cls, a):
        return ~cls._bv(a)

    @classmethod
    def shl(cls, a, s):
        return cls._bv(a) << s if s < 32 else cls.const(0)

    @classmethod
    def lshr(cls, a, s):
        return z3.LShR(cls._bv(a), s) if s < 32 else cls.const(0)

    @classmethod
    def ashr(cls, a, s):
        return cls._bv(a) >> min(s, 31)

    @classmethod
    def shl_v(cls, a, s):
        s = cls._bv(s)
        return z3.If(z3.ULT(s, 32), cls._bv(a) << s, cls.const(0))

    @classmethod
    def lshr_v(cls, a, s):
        s = cls._bv(s)
        return z3.If(z3.ULT(s, 32), z3.LShR(cls._bv(a), s), cls.const(0))

    @classmethod
    def ashr_v(cls, a, s):
        s = cls._bv(s)
        return cls._bv(a) >> z3.If(z3.ULT(s, 32), s, cls.const(31))

    @classmethod
    def mul64(cls, a, b):
        return z3.ZeroExt(32, cls._bv(a)) * z3.ZeroExt(32, cls._bv(b))

    @classmethod
    def add64(cls, a, b):
        return cls._bv(a, 64) + cls._bv(b, 64)

    @classmethod
    def shl64(cls, a, s):
        return cls._bv(a, 64) << s if s < 64 else z3.BitVecVal(0, 64)

    @classmethod
    def lshr64(cls, a, s):
        return z3.LShR(cls._bv(a, 64), s) if s < 64 else z3.BitVecVal(0, 64)

    @staticmethod
    def trunc32(a):
        if isinstance(a, int):
            return a & 0xFFFFFFFF
        return z3.Extract(31, 0, a)

    @classmethod
    def eq(cls, a, b):
        return cls._bv(a) == cls._bv(b)

    @classmethod
    def ne(cls, a, b):
        return cls._bv(a) != cls._bv(b)

    @classmethod
    def ult(cls, a, b):
        return z3.ULT(cls._bv(a), cls._bv(b))

    @classmethod
    def ule(cls, a, b):
        return z3.ULE(cls._bv(a), cls._bv(b))

    @classmethod
    def slt(cls, a, b):
        return cls._bv(a) < cls._bv(b)

    @classmethod
    def sle(cls, a, b):
        return cls._bv(a) <= cls._bv(b)

    @classmethod
    def ite(cls, c, a, b):
        if isinstance(c, bool):
            return a if c else b
        a = cls._bv(a) if not isinstance(a, bool) else a
        b = cls._bv(b) if not isinstance(b, bool) else b
        return z3.If(c, a, b)

    @staticmethod
    def bool_and(*xs):
        cs = [x for x in xs if not isinstance(x, bool)]
        if any(x is False for x in xs if isinstance(x, bool)):
            return False
        if not cs:
            return True
        return z3.And(*cs) if len(cs) > 1 else cs[0]

    @staticmethod
    def bool_or(*xs):
        cs = [x for x in xs if not isinstance(x, bool)]
        if any(x is True for x in xs if isinstance(x, bool)):
            return True
        if not cs:
            return False
        return z3.Or(*cs) if len(cs) > 1 else cs[0]

    @staticmethod
    def bool_not(x):
        if isinstance(x, bool):
            return not x
        return z3.Not(x)

    @classmethod
    def clz32_t(cls, a):
        """clz as a BV term (0..32) via binary cascade."""
        a = cls._bv(a)
        n = cls.const(0)
        x = a
        for width, mask in (
            (16, 0xFFFF0000),
            (8, 0xFF000000),
            (4, 0xF0000000),
            (2, 0xC0000000),
            (1, 0x80000000),
        ):
            cond = (x & cls.const(mask)) == cls.const(0)
            n = z3.If(cond, n + cls.const(width), n)
            x = z3.If(cond, x << width, x)
        return z3.If(a == cls.const(0), cls.const(32), n)


# ---------------------------------------------------------------------------
# fma_model_bh — transcribed from craq-sim src/fma.cpp:102-189 (@1c47e9cd).
# Bit-exact BH SFPU FMA: denormal-flush on inputs and output, product
# realigned to 27 bits with sticky, semi-sticky alignment shifts, RTNE.
# ---------------------------------------------------------------------------


def fma_model_bh(B, x, y, z):
    c = B.const

    def unpack(v):
        e = B.band(B.lshr(v, 23), c(255))
        m = B.bxor(B.band(v, c(0x7FFFFF)), c(0x800000))
        m = B.ite(B.eq(e, c(0)), c(0), m)  # flush denormals
        return e, m

    xe, xm = unpack(x)
    ye, ym = unpack(y)
    ze, zm0 = unpack(z)
    z_sign = B.band(z, c(0x80000000))
    p_sign = B.band(B.bxor(x, y), c(0x80000000))

    # p = x * y (48-bit product), then 3 extra G/R/S bits
    pm64 = B.shl64(B.mul64(xm, ym), 3)
    # p_e = x_e + y_e - 23 - 127 + 23 (after realign)  [kept as int32 term]
    p_e = B.sub(B.add(xe, ye), c(127))
    zm = B.shl(zm0, 3)

    # realign p to 27+1 bits with sticky
    pm_hi = B.trunc32(B.lshr64(pm64, 23))
    pm_sticky = B.ite(B.ne(B.band(B.trunc32(pm64), c(0x7FFFFF)), c(0)), c(1), c(0))
    pm = B.bor(pm_hi, pm_sticky)

    # special-case predicates (evaluated on post-shift zm as in the C)
    x_nan_ish = B.bool_and(
        B.eq(xe, c(255)), B.bool_or(B.ne(xm, c(0x800000)), B.eq(ym, c(0)))
    )
    y_nan_ish = B.bool_and(
        B.eq(ye, c(255)), B.bool_or(B.ne(ym, c(0x800000)), B.eq(xm, c(0)))
    )
    z_nan = B.bool_and(B.eq(ze, c(255)), B.ne(zm, c(0x4000000)))
    zp_inf_conflict = B.bool_and(
        B.eq(ze, c(255)),
        B.bool_or(B.eq(xe, c(255)), B.eq(ye, c(255))),
        B.ne(z_sign, p_sign),
    )
    special = B.bool_or(
        B.eq(xe, c(255)), B.eq(ye, c(255)), B.sle(c(255), p_e), B.eq(ze, c(255))
    )
    nan_out = B.bool_or(x_nan_ish, y_nan_ish, z_nan, zp_inf_conflict)
    special_result = B.ite(
        nan_out,
        c(0x7FC00000),
        B.ite(B.eq(ze, c(255)), z, B.bor(p_sign, c(0x7F800000))),
    )

    # shortcut if p == 0 or the multiply on its own would underflow
    p_zero = B.bool_or(B.eq(pm, c(0)), B.slt(p_e, c(0)))
    p_zero_result = B.ite(B.ne(zm, c(0)), z, B.band(z_sign, p_sign))

    # r = z + p with semi-sticky alignment shifts
    r_e = B.ite(B.slt(p_e, ze), ze, p_e)  # max(p_e, z_e)

    def semi_sticky(var, amount):
        # amount >= 0 (term).  if amount >= 32: 0 else v>>=a; if v: v |= (v<<a != orig)
        shifted = B.lshr_v(var, amount)
        back = B.shl_v(shifted, amount)
        sticky = B.ite(B.ne(back, var), c(1), c(0))
        with_sticky = B.bor(shifted, sticky)
        res = B.ite(B.eq(shifted, c(0)), shifted, with_sticky)
        return B.ite(B.sle(c(32), amount), c(0), res)

    pm_al = B.ite(B.slt(p_e, r_e), semi_sticky(pm, B.sub(r_e, p_e)), pm)
    zm_al = B.ite(B.slt(ze, r_e), semi_sticky(zm, B.sub(r_e, ze)), zm)
    r_sign = B.ite(B.ule(zm_al, pm_al), p_sign, z_sign)
    signs_differ = B.ne(p_sign, z_sign)
    zm2 = B.ite(B.ne(z_sign, r_sign), B.bnot(zm_al), zm_al)
    pm2 = B.ite(B.ne(p_sign, r_sign), B.bnot(pm_al), pm_al)
    r_m = B.add(B.add(zm2, pm2), B.ite(signs_differ, c(1), c(0)))

    r_zero_result = B.band(z_sign, p_sign)

    # normalise to 5 zero bits, 1 one bit, 26 fractional bits
    n = B.sub(c(5), B.clz32_t(r_m))
    r_e2 = B.add(r_e, n)
    overflow = B.sle(c(255), r_e2)
    under = B.sle(r_e2, c(0))
    n2 = B.ite(under, B.add(n, c(1)), n)
    r_e3 = B.ite(under, c(0), r_e2)
    neg_n = B.sub(c(0), n2)
    shifted_left = B.shl_v(r_m, neg_n)
    # right shift with the C's (n|1) semi-mask sticky
    sticky_mask = B.band(B.bor(n2, c(1)), c(0xFFFFFFFF))
    shifted_right = B.bor(
        B.lshr_v(r_m, n2), B.ite(B.ne(B.band(r_m, sticky_mask), c(0)), c(1), c(0))
    )
    r_m2 = B.ite(B.sle(n2, c(0)), shifted_left, shifted_right)

    r = B.add(B.shl(r_e3, 23), B.band(B.lshr(r_m2, 3), c(0x7FFFFF)))
    round_up = B.ult(c(4), B.add(B.band(r_m2, c(7)), B.band(r, c(1))))
    r = B.ite(round_up, B.add(r, c(1)), r)
    # flush denormals after rounding
    r = B.ite(B.eq(B.lshr(r, 23), c(0)), c(0), r)
    normal_result = B.bor(r_sign, r)
    normal_result = B.ite(B.eq(r_m, c(0)), r_zero_result, normal_result)
    normal_result = B.ite(overflow, B.bor(r_sign, c(0x7F800000)), normal_result)

    return B.ite(special, special_result, B.ite(p_zero, p_zero_result, normal_result))


# ---------------------------------------------------------------------------
# approx_recip / approx_exp — craq-sim src/tensix.cpp:10281-10362.
# ---------------------------------------------------------------------------

_ARECIP_LUT = [
    127,
    125,
    123,
    121,
    119,
    117,
    116,
    114,
    112,
    110,
    109,
    107,
    105,
    104,
    102,
    100,
    99,
    97,
    96,
    94,
    93,
    91,
    90,
    88,
    87,
    85,
    84,
    83,
    81,
    80,
    79,
    77,
    76,
    75,
    74,
    72,
    71,
    70,
    69,
    68,
    66,
    65,
    64,
    63,
    62,
    61,
    60,
    59,
    58,
    57,
    56,
    55,
    54,
    53,
    52,
    51,
    50,
    49,
    48,
    47,
    46,
    45,
    44,
    43,
    42,
    41,
    40,
    40,
    39,
    38,
    37,
    36,
    35,
    35,
    34,
    33,
    32,
    31,
    31,
    30,
    29,
    28,
    28,
    27,
    26,
    25,
    25,
    24,
    23,
    23,
    22,
    21,
    21,
    20,
    19,
    19,
    18,
    17,
    17,
    16,
    15,
    15,
    14,
    14,
    13,
    12,
    12,
    11,
    11,
    10,
    9,
    9,
    8,
    8,
    7,
    7,
    6,
    5,
    5,
    4,
    4,
    3,
    3,
    2,
    2,
    1,
    1,
    0,
]


def _lut_lookup(B, table, idx, idx_bits):
    """Balanced ite tree lookup of a python int table by a BV index term."""
    if B.is_const(idx):
        return B.const(table[B.const_val(idx) & ((1 << idx_bits) - 1)])

    def build(lo, hi, bit):
        if lo == hi:
            return B.const(table[lo])
        mid = (lo + hi) // 2
        return B.ite(
            B.ule(idx, B.const(mid)), build(lo, mid, bit), build(mid + 1, hi, bit)
        )

    return build(0, (1 << idx_bits) - 1, idx_bits)


def approx_recip(B, x):
    c = B.const
    lut_v = _lut_lookup(B, _ARECIP_LUT, B.band(B.lshr(x, 16), c(0x7F)), 7)
    mid = B.bor(B.shl(B.sub(c(253), B.lshr(x, 23)), 23), B.shl(lut_v, 16))
    return B.ite(
        B.ult(x, c(0x800000)),
        c(0x7F800000),
        B.ite(B.ult(x, c(0x7E800000)), mid, c(0)),
    )


# ---------------------------------------------------------------------------
# Field decode from the generated ISA JSON (craq-sim data/bh/tensix_isa.json).
# ---------------------------------------------------------------------------

DEFAULT_ISA_JSON = os.path.expanduser("~/sfpi-uplift/craq-sim/data/bh/tensix_isa.json")


class Isa:
    def __init__(self, path=DEFAULT_ISA_JSON):
        raw = json.load(open(path))
        self.by_opcode = {}
        for name, spec in raw.items():
            if not isinstance(spec, dict) or "opcode" not in spec:
                continue
            self.by_opcode[spec["opcode"]] = (name, spec.get("args", {}))

    def decode(self, inst):
        opcode = (inst >> 24) & 0xFF
        name, args = self.by_opcode.get(opcode, ("UNKNOWN_%02x" % opcode, {}))
        fields = {}
        for fname, span in args.items():
            hi, lo = (int(t) for t in span.split(":"))
            fields[fname] = (inst >> lo) & ((1 << (hi - lo + 1)) - 1)
        return opcode, name, fields


# ---------------------------------------------------------------------------
# Dst model.  Cells are 16-bit datums addressed (adj_row, col) exactly like
# p_tensix->dst.  A 32-bit access reads/writes (adj_row, col) hi and
# (adj_row+8, col) lo (tensix.cpp:3508-3560).  First read of an unknown cell
# binds it: concrete mode adopts the value implied by the trace; symbolic
# mode creates a fresh 16-bit symbol (shared across legs via `symbols`).
# ---------------------------------------------------------------------------


class ScopeRefusal(Exception):
    pass


class ValidationError(Exception):
    pass


def dst_remap_row(row, remap):
    if remap:
        row = (row & 0x3C7) ^ ((row & 0x030) >> 1) ^ ((row & 0x008) << 2)
    return row


def dst32b_adjust_row(row, remap, swizzle):
    row = dst_remap_row(row, remap)
    if swizzle:
        row = (row & 0x3F3) ^ ((row & 0x018) >> 1) ^ ((row & 0x004) << 1)
    return ((row & 0x1F8) << 1) | (row & 0x207)


def dst16b_adjust_row(row):
    return row


class Dst:
    """Dst memory model with GENERATIONS: the FPU (datacopy) rewrites the
    input tiles between kernel invocations of the same test; the concrete
    validation run detects each rewrite (traced load value != modeled cell)
    and journals it; the symbolic run replays the journal, bumping the
    cell's generation (a fresh input symbol) at the same instruction and
    capturing any SFPU-written value it overwrites as an OUTPUT."""

    def __init__(self, B, symbols=None, name=""):
        self.B = B
        self.cells = {}  # (adj_row, col) -> 16-bit term
        self.written = set()
        self.gen = {}  # (adj_row, col) -> generation int
        self.outputs = {}  # ((adj_row, col), gen) -> term (captured at invalidation)
        self.symbols = symbols if symbols is not None else {}
        self.tags = {}  # (adj_row, col) -> "32hi"/"32lo"/"16" datum width tag
        self.name = name

    def _fresh(self, key):
        g = self.gen.get(key, 0)
        skey = (key[0], key[1], g)
        if skey not in self.symbols:
            self.symbols[skey] = z3.BitVec("d_%d_%d_g%d" % skey, 16)
        return z3.ZeroExt(16, self.symbols[skey])

    def read16(self, adj_row, col):
        key = (adj_row, col)
        if key not in self.cells:
            if not self.B.symbolic:
                return None
            self.cells[key] = self._fresh(key)
        return self.cells[key]

    def write16(self, adj_row, col, val16):
        self.cells[(adj_row, col)] = self.B.band(val16, self.B.const(0xFFFF))
        self.written.add((adj_row, col))

    def set_generation(self, key, gen):
        """The FPU delivered generation `gen` of this input cell: capture any
        SFPU-written value as an output of the old epoch, drop the cell."""
        if (
            self.gen.get(key, 0) == gen
            and key in self.cells
            and key not in self.written
        ):
            return  # already at this generation
        if key in self.written:
            self.outputs[(key, self.gen.get(key, 0))] = self.cells[key]
            self.written.discard(key)
        self.cells.pop(key, None)
        self.gen[key] = gen

    def final_outputs(self):
        out = dict(self.outputs)
        for key in self.written:
            out[(key, self.gen.get(key, 0))] = self.cells[key]
        return out


# ---------------------------------------------------------------------------
# The executor.
# ---------------------------------------------------------------------------

NLANES = 2  # lanes modeled exactly (0 and 1)

# constant LRegs at reset (tensix.cpp:100-108)
LREG_RESET = {8: 0x3F56594B, 9: 0, 10: 0x3F800000, 11: 0, 12: 0, 13: 0, 14: 0}
# SFPCONFIG VD 11..14 IS_VALUE constants (tensix.cpp:9709-9714)
SFPCONFIG_CONST = {11: 0xBF800000, 12: 0x3B000000, 13: 0xBF2CC4C7, 14: 0xBEB08FF9}


class LaneState:
    """Registers of ONE lane: lregs[0..16] (16 = macro LReg), cc bit."""

    def __init__(self, B, lane):
        self.lregs = [B.const(0)] * 17
        for r, v in LREG_RESET.items():
            self.lregs[r] = B.const(v)
        self.lregs[15] = B.const(lane << 1)  # tensix.cpp:106-108
        self.cc = True  # bool term: lane enabled


class Executor:
    def __init__(self, B, isa, dst, tag="", journal=None):
        self.B = B
        self.isa = isa
        self.dst = dst
        self.tag = tag
        self.journal = journal  # symbolic: {n_exec: {cell: canonical gen}}
        self.adoptions = []  # concrete: (n_exec, cell, raw16) input-adoption events
        self.current_v = None  # concrete: V words paired with the current I
        self.lanes = [LaneState(B, l) for l in range(NLANES)]
        self.cc_en = True  # global, concrete by construction
        self.cc_stack = []  # list of (cc_en, [cc bool per lane])
        self.templates = [0, 0, 0, 0]
        self.sequences = [0, 0, 0, 0]
        self.misc = 0
        self.lane_config = [0] * NLANES  # via SFPCONFIG VD15
        self.pending_cc = None  # (cc_en, [cc per lane]) deferred macro CC
        self.n_exec = 0
        self.refusals = []

    # ---- lane mask -------------------------------------------------------
    def lane_mask(self, lane):
        """bool term: lane enabled for a predicated op."""
        return self.lanes[lane].cc if self.cc_en else True

    # ---- generic write helpers -------------------------------------------
    def write_lreg(self, lane, reg, val, gate=None):
        if reg >= 17:
            raise ScopeRefusal("lreg-index-%d" % reg)
        old = self.lanes[lane].lregs[reg]
        g = self.lane_mask(lane) if gate is None else gate
        self.lanes[lane].lregs[reg] = (
            self.B.ite(g, val, old) if not (g is True) else val
        )

    def set_cc(self, lane, val, gate=None):
        old = self.lanes[lane].cc
        g = self.lane_mask(lane) if gate is None else gate
        if g is True:
            self.lanes[lane].cc = val
        else:
            self.lanes[lane].cc = self.B.bool_or(
                self.B.bool_and(g, val), self.B.bool_and(self.B.bool_not(g), old)
            )

    # ---- dst load/store conversions (tensix.cpp SFPLOAD/SFPSTORE) --------
    def dst_load_keys(self, row, col, mod0, flags):
        """The exact dst cells whose bits the loaded value depends on."""
        dst32_layout = bool(flags & 1) or bool(flags & 2)
        remap, swizzle = bool(flags & 4), bool(flags & 8)
        ar32 = dst32b_adjust_row(row, remap, swizzle)
        if mod0 in (3, 4, 10, 12):
            return [(ar32, col), (ar32 + 8, col)]
        if mod0 == 2 and dst32_layout:
            return [(ar32, col)]  # only the hi half feeds the value
        if flags & 1:  # dst_32bit_addr_en: 16b reads see the hi half
            return [(ar32, col)]
        return [(dst16b_adjust_row(row), col)]

    def dst_load_convert(self, raw_terms, mod0, flags, cur_lreg):
        """Apply the SFPLOAD mod0 conversion (tensix.cpp:8450-8496) to the
        raw cell terms returned for dst_load_keys."""
        B = self.B
        c = B.const
        if mod0 in (3, 4, 10, 12):
            hi, lo = raw_terms
            v32 = B.bor(B.shl(B.band(hi, c(0xFFFF)), 16), B.band(lo, c(0xFFFF)))
            return decode_fp32(B, v32)
        v = B.band(raw_terms[0], c(0xFFFF))
        if mod0 == 1:
            s = B.lshr(v, 15)
            e = B.band(v, c(31))
            m = B.band(B.lshr(v, 5), c(1023))
            e = B.ite(B.ne(e, c(0)), B.add(e, c(112)), e)
            return B.bor(B.shl(s, 31), B.bor(B.shl(e, 23), B.shl(m, 13)))
        if mod0 == 2:
            return B.shl(decode_bf16(B, v), 16)
        if mod0 == 6:
            return v
        if mod0 == 5:
            return sign_mag8_to_32(B, v)
        if mod0 == 7:
            return B.shl(v, 16)
        if mod0 == 8:
            return sign_mag16_to_32(B, v)
        if mod0 == 9:
            return v
        if mod0 == 13:
            return sign_mag_to_twos(B, sign_mag11_to_32(B, v))
        if mod0 == 14:
            return B.bor(B.band(cur_lreg, c(0xFFFF0000)), v)
        if mod0 == 15:
            return B.bor(B.shl(v, 16), B.band(cur_lreg, c(0xFFFF)))
        raise ScopeRefusal("sfpload-mod0-%d" % mod0)

    def dst_store(self, row, col, mod0, flags, value, gate, lane_cfg, lreg_ind):
        """tensix.cpp sfpstore_values (8607-8685)."""
        B = self.B
        c = B.const
        full32 = bool(flags & 1) or bool(flags & 2)
        remap, swizzle = bool(flags & 4), bool(flags & 8)
        if lane_cfg & (1 << 4):  # block_wr
            return
        ar32 = dst32b_adjust_row(row, remap, swizzle)
        ar16 = dst16b_adjust_row(row)

        def w32(data32):
            hi, lo = B.lshr(data32, 16), B.band(data32, c(0xFFFF))
            old_hi = self.dst.read16(ar32, col)
            old_lo = self.dst.read16(ar32 + 8, col)
            if old_hi is None:
                old_hi = c(0)
            if old_lo is None:
                old_lo = c(0)
            self.dst.write16(
                ar32, col, B.ite(gate, hi, old_hi) if gate is not True else hi
            )
            self.dst.write16(
                ar32 + 8, col, B.ite(gate, lo, old_lo) if gate is not True else lo
            )

        def w16(data16):
            if flags & 1:  # dst_32bit_addr_en
                w32(B.shl(B.band(data16, c(0xFFFF)), 16))
                return
            old = self.dst.read16(ar16, col)
            if old is None:
                old = c(0)
            self.dst.write16(
                ar16, col, B.ite(gate, data16, old) if gate is not True else data16
            )

        if mod0 == 1:
            w16(encode_fp16(B, store_to_fp16(B, value)))
        elif mod0 == 2:
            v = denormals_as_zeros(B, value)
            data = encode_bf16(B, B.lshr(v, 16))
            if full32:
                data32 = B.shl(data, 16)
                preserve = (lreg_ind == 9) or bool(lane_cfg & 4)
                if preserve:
                    old_lo = self.dst.read16(ar32 + 8, col)
                    if old_lo is None:
                        old_lo = c(0)
                    data32 = B.bor(data32, B.band(old_lo, c(0xFFFF)))
                w32(data32)
            else:
                w16(data)
        elif mod0 == 3:
            w32(encode_fp32(B, denormals_as_zeros(B, value)))
        elif mod0 == 4:
            w32(encode_fp32(B, value))
        elif mod0 in (6, 14):
            w16(B.band(value, c(0xFFFF)))
        elif mod0 == 7:
            w32(value)
        elif mod0 == 9:
            w32(B.bor(B.shl(value, 16), B.lshr(value, 16)))
        elif mod0 == 12:
            w32(encode_fp32(B, value))
        elif mod0 == 15:
            w16(B.lshr(value, 16))
        else:
            raise ScopeRefusal("sfpstore-mod0-%d" % mod0)

    # ---- instruction execution -------------------------------------------
    def exec_inst(self, inst, aux0, aux1):
        B = self.B
        opcode = (inst >> 24) & 0xFF

        # template-capture paths (tensix.cpp:11097-11134)
        if opcode == 0x70:
            vd = (inst >> 20) & 0xF
            if 12 <= vd <= 15 and not (self.lane_config[0] & 2):
                self.templates[vd - 12] = inst
                self.n_exec += 1
                return
        elif opcode in (
            0x79,
            0x7A,
            0x7B,
            0x7F,
            0x80,
            0x84,
            0x89,
            0x8A,
            0x8E,
            0x90,
            0x92,
            0x94,
            0x97,
            0x98,
            0x99,
        ):
            vd = (inst >> 4) & 0xF
            if 12 <= vd <= 15:
                self.templates[vd - 12] = inst
                self.n_exec += 1
                return

        _, name, f = self.isa.decode(inst)
        self.n_exec += 1

        if opcode == 0x70:  # SFPLOAD
            row, mod0 = aux0 & 0xFFFF, (aux0 >> 16) & 0xF
            self._do_load(inst, row, mod0, aux1)
        elif opcode == 0x71:
            self._loadi(f)
        elif opcode == 0x72:  # SFPSTORE
            row, mod0 = aux0 & 0xFFFF, (aux0 >> 16) & 0xF
            self._do_store(inst, row, mod0, aux1)
        elif opcode == 0x93:  # SFPLOADMACRO launch: only the SFPLOAD part here
            row, mod0 = aux0 & 0xFFFF, (aux0 >> 16) & 0xF
            vd_lo = (inst >> 20) & 3
            vd = ((f["dest_reg_addr"] & 1) << 2) | vd_lo
            self._do_load_into(vd, row, mod0, aux1)
        else:
            self.exec_compute(opcode, f, None, None)

    def _do_load(self, inst, row, mod0, aux1):
        lreg_ind = (inst >> 20) & 0xF
        self._do_load_into(lreg_ind, row, mod0, aux1)

    def _do_load_into(self, lreg_ind, row, mod0, aux1):
        """tensix.cpp SFPLOAD body 8390-8515, plus the input-adoption /
        FPU-rewrite journal (see Dst docstring)."""
        B = self.B
        mask_all = mod0 == 10  # mod0 10: all lanes
        # symbolic replay of the concrete run's input-adoption journal:
        # set each cell to its canonical generation before this load reads it
        if B.symbolic and self.journal is not None:
            for key, gen in self.journal.get(self.n_exec, {}).items():
                self.dst.set_generation(key, gen)
        for lane in range(NLANES):
            lane_cfg = (aux1 >> (8 * (lane + 1))) & 0xFF
            if lane_cfg & (1 << 5):  # block_rd
                continue
            gate = True if mask_all else self.lane_mask(lane)
            r = row
            rrow = (r & ~3) + lane // 8
            col = 2 * (lane & 7) + (1 if ((r & 2) >> 1) or (lane_cfg & (1 << 6)) else 0)
            keys = self.dst_load_keys(rrow, col, mod0, aux1)
            if len(keys) == 2:
                self.dst.tags[keys[0]] = "32hi"
                self.dst.tags[keys[1]] = "32lo"
            else:
                self.dst.tags.setdefault(keys[0], "16")
            cur = self.lanes[lane].lregs[lreg_ind] if lreg_ind < 8 else B.const(0)
            if B.symbolic:
                raws = [self.dst.read16(*k) for k in keys]
                v = self.dst_load_convert(raws, mod0, aux1, cur)
                if lreg_ind < 8:
                    self.write_lreg(lane, lreg_ind, v, gate)
                continue
            # ---- concrete mode ----
            known = all(k in self.dst.cells for k in keys)
            traced = None
            if gate is True and lreg_ind < 8 and self.current_v is not None:
                traced = self.current_v[lane * 8 + lreg_ind]
            if known:
                raws = [self.dst.cells[k] for k in keys]
                v = self.dst_load_convert(raws, mod0, aux1, cur)
                if traced is not None and v != traced:
                    # the FPU rewrote these cells since we last saw them:
                    # re-adopt from the trace (event recorded by _adopt_cells)
                    _adopt_cells(self, rrow, col, mod0, aux1, traced, cur)
                    v = traced
            else:
                if traced is None:
                    continue  # masked-off or untraceable: leave cells unknown
                _adopt_cells(self, rrow, col, mod0, aux1, traced, cur)
                v = traced
            if lreg_ind < 8 and (gate is not False):
                self.write_lreg(lane, lreg_ind, v, gate)

    def _do_store(self, inst, row, mod0, aux1):
        B = self.B
        lreg_ind = (inst >> 20) & 0xF
        if lreg_ind >= 12:
            raise ScopeRefusal("sfpstore-lreg-%d" % lreg_ind)
        for lane in range(NLANES):
            lane_cfg = (aux1 >> (8 * (lane + 1))) & 0xFF
            gate = self.lane_mask(lane)
            r = row
            rrow = (r & ~3) + lane // 8
            col = 2 * (lane & 7) + (1 if ((r & 2) >> 1) or (lane_cfg & (1 << 7)) else 0)
            val = self.lanes[lane].lregs[lreg_ind] if lreg_ind < 17 else B.const(0)
            self.dst_store(rrow, col, mod0, aux1, val, gate, lane_cfg, lreg_ind)

    def _loadi(self, f):
        """tensix.cpp SFPLOADI 8521-8574."""
        B = self.B
        c = B.const
        mod0, imm16, vd = f["instr_mod0"], f["imm16"], f["lreg_ind"]
        if mod0 == 0 or mod0 == 8:
            val = c(imm16 << 16)
        elif mod0 == 1:
            s = imm16 & 0x8000
            em = imm16 & 0x7FFF
            val = c((s << 16) | ((em + (112 << 10)) << 13))
        elif mod0 in (2, 3, 10):
            val = c(imm16)
        elif mod0 in (4, 5, 6, 7):
            v = imm16 if imm16 < 0x8000 else imm16 - 0x10000
            val = c(v & 0xFFFFFFFF)
        else:
            raise ScopeRefusal("sfploadi-mod0-%d" % mod0)
        for lane in range(NLANES):
            cur = self.lanes[lane].lregs[vd]
            if mod0 == 8:
                nv = B.bor(B.band(cur, c(0xFFFF)), val)
            elif mod0 == 10:
                nv = B.bor(B.band(cur, c(0xFFFF0000)), val)
            else:
                nv = val
            self.write_lreg(lane, vd, nv)

    # ---- compute ops (shared by ordinary and macro-template paths) --------
    def exec_compute(
        self, opcode, f, read_override, write_override, mask_override=None
    ):
        """Execute a non-load/store SFPU op.  read_override/write_override:
        (lane, reg)->term accessors used by the macro direct evaluator; None
        means the live register file.  mask_override: per-lane bool gates."""
        B = self.B
        c = B.const

        def rd(lane, reg):
            if read_override:
                return read_override(lane, reg)
            return self.lanes[lane].lregs[reg]

        def wr(lane, reg, val, gate):
            if write_override:
                write_override(lane, reg, val, gate)
            else:
                if reg < 8 or reg == 16:
                    self.write_lreg(lane, reg, val, gate)

        def gate_of(lane):
            if mask_override is not None:
                return mask_override[lane]
            return self.lane_mask(lane)

        if opcode == 0x73:  # SFPLUT (8754-8779)
            mod0 = f["instr_mod0"]
            vd = f["lreg_ind"]
            for lane in range(NLANES):
                g = gate_of(lane)
                l3 = rd(lane, 3)
                b = B.band(l3, c(0x7FFFFFFF))
                coeffs = B.ite(
                    B.ult(b, c(0x3F800000)),
                    rd(lane, 0),
                    B.ite(B.ult(b, c(0x40000000)), rd(lane, 1), rd(lane, 2)),
                )
                a = lut8_to_fp32(B, B.band(B.lshr(coeffs, 8), c(0xFF)))
                cc0 = lut8_to_fp32(B, B.band(coeffs, c(0xFF)))
                d = fma_model_bh(B, a, b, cc0)
                if mod0 & 4:
                    d = B.bor(B.band(d, c(0x7FFFFFFF)), B.band(l3, c(0x80000000)))
                if mod0 & 8:
                    raise ScopeRefusal("sfplut-indirect-vd")
                wr(lane, vd, d, g)
        elif opcode in (0x74, 0x75):  # SFPMULI / SFPADDI (8785-8817)
            vd = f["lreg_dest"]
            imm = c((f["imm16_math"] << 16) & 0xFFFFFFFF)
            for lane in range(NLANES):
                g = gate_of(lane)
                cur = rd(lane, vd)
                if opcode == 0x74:
                    res = fma_model_bh(B, imm, cur, c(0))
                else:
                    res = fma_model_bh(B, imm, c(0x3F800000), cur)
                wr(lane, vd, res, g)
        elif opcode == 0x76:  # SFPDIVP2 (8819-8843)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            imm8 = imm12 & 0xFF
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                e = B.band(B.lshr(src, 23), c(255))
                if mod1 & 1:
                    e = B.ite(B.ne(e, c(255)), B.band(B.add(e, c(imm8)), c(255)), e)
                else:
                    e = c(imm8)
                wr(lane, vd, B.bor(B.shl(e, 23), B.band(src, c(0x807FFFFF))), g)
        elif opcode == 0x77:  # SFPEXEXP (8845-8875)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_c"]
            bias = 0 if (mod1 & 1) else 127
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                exp = B.band(B.lshr(src, 23), c(255))
                dstv = B.sub(exp, c(bias))
                wr(lane, vd, dstv, g)
                neg = B.slt(dstv, c(0))
                if mod1 == 10:
                    self.set_cc(lane, B.bool_not(neg), g)
                elif mod1 == 2:
                    self.set_cc(lane, neg, g)
        elif opcode == 0x78:  # SFPEXMAN (8877-8892)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_c"]
            hidden = 0 if mod1 else 0x800000
            for lane in range(NLANES):
                g = gate_of(lane)
                wr(lane, vd, B.bor(B.band(rd(lane, vc), c(0x7FFFFF)), c(hidden)), g)
        elif opcode == 0x79:  # SFPIADD (8894-8929; direct: 11477-11511)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            vb = f.get("_vb_force", vd)
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                if mod1 & 1:
                    simm = imm12 if imm12 < 0x800 else imm12 - 0x1000
                    src = B.add(src, c(simm & 0xFFFFFFFF))
                elif mod1 & 2:
                    src = B.sub(src, rd(lane, vb))
                else:
                    src = B.add(src, rd(lane, vb))
                neg = B.ne(B.band(src, c(0x80000000)), c(0))
                if mod1 & 8:
                    self.set_cc(lane, B.bool_not(neg), g)
                elif not (mod1 & 4):
                    self.set_cc(lane, neg, g)
                wr(lane, vd, src, g)
        elif opcode == 0x7A:  # SFPSHFT (8931-8967)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            simm = imm12 if imm12 < 0x800 else imm12 - 0x1000
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc) if (mod1 & 4) else rd(lane, vd)
                if mod1 & 1:
                    amt = simm
                    if amt >= 0:
                        res = B.shl(src, amt & 31)
                    elif mod1 & 2:
                        res = B.ashr(src, (-amt) & 31)
                    else:
                        res = B.lshr(src, (-amt) & 31)
                else:
                    amt = rd(lane, vc)
                    npos = B.sle(c(0), amt)
                    left = B.shl_v(src, B.band(amt, c(31)))
                    ramt = B.band(B.sub(c(0), amt), c(31))
                    right = B.ashr_v(src, ramt) if (mod1 & 2) else B.lshr_v(src, ramt)
                    res = B.ite(npos, left, right)
                wr(lane, vd, res, g)
        elif opcode == 0x7B:  # SFPSETCC (8969-8999)
            mod1, imm12, vc = f["instr_mod1"], f["imm12_math"], f["lreg_c"]
            if f["lreg_dest"] >= 12:
                return
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                is_zero = B.eq(src, c(0))
                is_neg = B.ne(B.band(src, c(0x80000000)), c(0))
                cc_sel = B.bool_not(is_zero) if (mod1 & 2) else is_neg
                if mod1 & 8:
                    cc_res = False
                elif mod1 & 1:
                    cc_res = bool(imm12 & 1)
                elif mod1 & 4:
                    cc_res = B.bool_not(cc_sel)
                else:
                    cc_res = cc_sel
                if not self.cc_en:
                    cc_res = False
                self.set_cc(lane, cc_res, g)
        elif opcode == 0x7C:  # SFPMOV (9002-9027)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_c"]
            if mod1 == 8:
                raise ScopeRefusal("sfpmov-prng")
            for lane in range(NLANES):
                g = True if mod1 == 2 else gate_of(lane)
                src = rd(lane, vc)
                if mod1 & 1:
                    src = B.bxor(src, c(0x80000000))
                wr(lane, vd, src, g)
        elif opcode == 0x7D:  # SFPABS (9030-9052)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_c"]
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                if mod1 & 1:
                    res = B.ite(
                        B.ule(src, c(0xFF800000)), B.band(src, c(0x7FFFFFFF)), src
                    )
                else:
                    res = B.ite(B.ule(c(0x80000000), src), B.sub(c(0), src), src)
                wr(lane, vd, res, g)
        elif opcode in (0x7E, 0x7F, 0x8D, 0x80):  # AND/OR/XOR/NOT (9068-9104)
            mod1 = f.get("instr_mod1", 0)
            imm12 = f.get("imm12_math", 0)
            vd, vc = f["lreg_dest"], f["lreg_c"]
            vb = vd
            if opcode in (0x7E, 0x7F) and mod1 == 1:
                vb = imm12 & 0xF
            vb = f.get("_vb_force", vb)
            for lane in range(NLANES):
                g = gate_of(lane)
                b_, c_ = rd(lane, vb), rd(lane, vc)
                if opcode == 0x7E:
                    res = B.band(b_, c_)
                elif opcode == 0x7F:
                    res = B.bor(b_, c_)
                elif opcode == 0x8D:
                    res = B.bxor(b_, c_)
                else:
                    res = B.bnot(c_)
                wr(lane, vd, res, g)
        elif opcode == 0x81:  # SFPLZ (9106-9131)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_c"]
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                if mod1 & 4:
                    src = B.band(src, c(0x7FFFFFFF))
                wr(lane, vd, B.clz32_t(src) if B.symbolic else c(B.clz32(src)), g)
                if mod1 & 2:
                    self.set_cc(lane, B.ne(src, c(0)), g)
        elif opcode == 0x82:  # SFPSETEXP (9133-9159)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            for lane in range(NLANES):
                g = gate_of(lane)
                src = B.band(rd(lane, vc), c(0x807FFFFF))
                if mod1 == 1:
                    exp = c(imm12 & 0xFF)
                elif mod1 == 2:
                    exp = B.band(B.lshr(rd(lane, vd), 23), c(0xFF))
                else:
                    exp = B.band(rd(lane, vd), c(0xFF))
                wr(lane, vd, B.bor(src, B.shl(exp, 23)), g)
        elif opcode == 0x83:  # SFPSETMAN (9161-9181)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            for lane in range(NLANES):
                g = gate_of(lane)
                src = B.band(rd(lane, vc), c(0xFF800000))
                if mod1 & 1:
                    src = B.bor(src, c((imm12 << 11) & 0x7FFFFF))
                else:
                    src = B.bor(src, B.band(rd(lane, vd), c(0x7FFFFF)))
                wr(lane, vd, src, g)
        elif opcode in (0x84, 0x85, 0x86):  # SFPMAD/ADD/MUL (9183-9315)
            mod1 = f["instr_mod1"]
            vd, va, vb, vc = (
                f["lreg_dest"],
                f["lreg_src_a"],
                f["lreg_src_b"],
                f["lreg_src_c"],
            )
            if opcode == 0x85 and va != 10:
                va, vb = vb, va  # make va = LReg[10]
            if mod1 & 12:
                raise ScopeRefusal("mad-indirect-mod1-%d" % mod1)
            for lane in range(NLANES):
                g = gate_of(lane)
                a = c(0x3F800000) if opcode == 0x85 else rd(lane, va)
                b_ = rd(lane, vb)
                c_ = rd(lane, vc)
                if mod1 & 1:
                    b_ = B.bxor(b_, c(0x80000000))
                if mod1 & 2:
                    c_ = B.bxor(c_, c(0x80000000))
                wr(lane, vd, fma_model_bh(B, a, b_, c_), g)
        elif opcode == 0x87:  # SFPPUSHC (9337-9372)
            mod1 = f["instr_mod1"]
            if f["lreg_dest"] >= 12:
                return
            if not mod1:
                self.cc_stack.append((self.cc_en, [ln.cc for ln in self.lanes]))
            else:
                if not self.cc_stack:
                    raise ScopeRefusal("pushc-modify-empty-stack")
                ten, tcc = self.cc_stack[-1]
                if mod1 <= 12:
                    ncc = [
                        cc_boolop(B, mod1, tcc[l], self.lanes[l].cc)
                        for l in range(NLANES)
                    ]
                    self.cc_stack[-1] = (self.cc_en, ncc)
                elif mod1 == 13:
                    for l in range(NLANES):
                        self.lanes[l].cc = B.bool_not(self.lanes[l].cc)
                    self.cc_stack[-1] = (self.cc_en, [ln.cc for ln in self.lanes])
                elif mod1 == 14:
                    self.cc_stack[-1] = (True, [True] * NLANES)
                else:
                    self.cc_stack[-1] = (True, [False] * NLANES)
        elif opcode == 0x88:  # SFPPOPC (9374-9410)
            mod1 = f["instr_mod1"]
            if f["lreg_dest"] >= 12:
                return
            top_en = self.cc_stack[-1][0] if self.cc_stack else False
            top_cc = self.cc_stack[-1][1] if self.cc_stack else [False] * NLANES
            if not mod1:
                if not self.cc_stack:
                    raise ScopeRefusal("popc-underflow")
                self.cc_stack.pop()
                self.cc_en = top_en
                for l in range(NLANES):
                    self.lanes[l].cc = top_cc[l]
            elif mod1 <= 12:
                self.cc_en = top_en
                for l in range(NLANES):
                    self.lanes[l].cc = cc_boolop(B, mod1, self.lanes[l].cc, top_cc[l])
            elif mod1 == 13:
                for l in range(NLANES):
                    self.lanes[l].cc = B.bool_not(self.lanes[l].cc)
            elif mod1 == 14:
                self.cc_en = True
                for l in range(NLANES):
                    self.lanes[l].cc = True
            else:
                self.cc_en = True
                for l in range(NLANES):
                    self.lanes[l].cc = False
        elif opcode == 0x89:  # SFPSETSGN (9412-9435; direct: sign reg = vd or 16)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            sign_reg = f.get("_vb_force", vd)
            for lane in range(NLANES):
                g = gate_of(lane)
                src = B.band(rd(lane, vc), c(0x7FFFFFFF))
                if mod1 & 1:
                    src = B.bor(src, c((imm12 & 1) << 31))
                else:
                    src = B.bor(src, B.band(rd(lane, sign_reg), c(0x80000000)))
                wr(lane, vd, src, g)
        elif opcode == 0x8A:  # SFPENCC (9437-9456)
            mod1, imm12 = f["instr_mod1"], f["imm12_math"]
            if f["lreg_dest"] >= 12:
                return
            if mod1 & 8:
                v = bool(imm12 & 2)
            else:
                v = True
            for l in range(NLANES):
                self.lanes[l].cc = v
            if mod1 & 2:
                self.cc_en = bool(imm12 & 1)
            elif mod1 & 1:
                self.cc_en = not self.cc_en
        elif opcode == 0x8B:  # SFPCOMPC (9458-9477)
            if f["lreg_dest"] >= 12:
                return
            if self.cc_stack:
                ten, tcc = self.cc_stack[-1]
                for l in range(NLANES):
                    if self.cc_en and ten:
                        self.lanes[l].cc = B.bool_and(
                            tcc[l], B.bool_not(self.lanes[l].cc)
                        )
                    else:
                        self.lanes[l].cc = False
            else:
                for l in range(NLANES):
                    self.lanes[l].cc = (
                        B.bool_not(self.lanes[l].cc) if self.cc_en else False
                    )
        elif opcode == 0x8C:
            raise ScopeRefusal("sfptransp-crosslane")
        elif opcode == 0x8E:  # SFP_STOCH_RND (9508-9595)
            self._stoch_rnd(f, rd, wr, gate_of)
        elif opcode == 0x8F:
            pass
        elif opcode == 0x90:  # SFPCAST (9601-9641)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_src_c"]
            if mod1 == 1:
                raise ScopeRefusal("sfpcast-stochastic")
            for lane in range(NLANES):
                g = gate_of(lane)
                src = rd(lane, vc)
                sign = B.band(src, c(0x80000000))
                if mod1 == 3:
                    res = B.bor(sign, B.ite(B.ne(sign, c(0)), B.sub(c(0), src), src))
                else:
                    mag = B.band(src, c(0x7FFFFFFF))
                    lz = B.clz32_t(mag)
                    m2 = B.shl_v(mag, lz)
                    dstv = B.bor(
                        sign, B.add(B.shl(B.sub(c(157), lz), 23), B.lshr(m2, 8))
                    )
                    rup = B.bool_and(
                        B.ne(B.band(m2, c(0x80)), c(0)),
                        B.ne(B.band(m2, c(0x17F)), c(0)),
                    )
                    dstv = B.ite(rup, B.add(dstv, c(1)), dstv)
                    res = B.ite(B.eq(mag, c(0)), sign, dstv)
                wr(lane, vd, res, g)
        elif opcode == 0x91:  # SFPCONFIG (9643-9751)
            self._sfpconfig(f)
        elif opcode == 0x92:  # SFPSWAP (9753-9806)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_src_c"]
            vd_gets_min = swap_min_mask(mod1)
            for lane in range(NLANES):
                g = gate_of(lane)
                c_ = rd(lane, vc)
                d_ = rd(lane, vd)
                if mod1 == 0:
                    should = True
                else:
                    cu = sm_total_order(B, c_)
                    du = sm_total_order(B, d_)
                    lt = B.slt(cu, du)
                    should = lt if ((vd_gets_min >> lane) & 1) else B.bool_not(lt)
                    if self.lane_config[lane] & ((1 << 8) | (1 << 11)):
                        should = B.bool_not(should)
                gg = B.bool_and(g, should)
                if self.lane_config[lane] & 4:
                    if vc < 4:
                        wr(lane, vc, d_, gg)
                    if vd < 4:
                        wr(lane, vd, c_, gg)
                    vca, vda = 4 + (vc & 3), 4 + (vd & 3)
                    a_, b_ = rd(lane, vca), rd(lane, vda)
                    wr(lane, vca, b_, gg)
                    wr(lane, vda, a_, gg)
                else:
                    if vc < 8:
                        wr(lane, vc, d_, gg)
                    if vd < 8:
                        wr(lane, vd, c_, gg)
        elif opcode == 0x94:  # SFPSHFT2 (9942-10112)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_src_c"],
                f["imm12_math"],
            )
            if mod1 in (1, 2, 3, 4):
                raise ScopeRefusal("sfpshft2-crosslane-mod%d" % mod1)
            if mod1 == 0:
                for lane in range(NLANES):
                    g = gate_of(lane)
                    v1, v2, v3 = rd(lane, 1), rd(lane, 2), rd(lane, 3)
                    wr(lane, 0, v1, g)
                    wr(lane, 1, v2, g)
                    wr(lane, 2, v3, g)
                    wr(lane, 3, c(0), g)
            elif mod1 == 5:
                vb = f.get("_vb_force", imm12 & 15)
                for lane in range(NLANES):
                    g = gate_of(lane)
                    amt = rd(lane, vc)
                    src = rd(lane, vb)
                    npos = B.sle(c(0), amt)
                    res = B.ite(
                        npos,
                        B.shl_v(src, B.band(amt, c(31))),
                        B.lshr_v(src, B.band(B.sub(c(0), amt), c(31))),
                    )
                    wr(lane, vd, res, g)
            else:  # mod1 == 6
                shift = imm12 if imm12 < 0x800 else imm12 - 0x1000
                vb = f.get("_vb_force", imm12 & 15)
                for lane in range(NLANES):
                    g = gate_of(lane)
                    src = rd(lane, vb)
                    res = (
                        B.shl(src, shift & 31)
                        if shift >= 0
                        else B.lshr(src, (-shift) & 31)
                    )
                    wr(lane, vd, res, g)
        elif opcode == 0x95:  # SFPLUTFP32 (10114-10208)
            self._lutfp32(f, rd, wr, gate_of)
        elif opcode in (0x96, 0x97):  # SFPLE / SFPGT (10210-10260)
            mod1, vd, vc = f["instr_mod1"], f["lreg_dest"], f["lreg_c"]
            for lane in range(NLANES):
                g = gate_of(lane)
                c_ = rd(lane, vc)
                d_ = rd(lane, vd)
                du, cu = sm_total_order(B, d_), sm_total_order(B, c_)
                res = B.sle(du, cu) if opcode == 0x96 else B.bool_not(B.sle(du, cu))
                if mod1 == 1:
                    self.set_cc(lane, res, g)
                else:
                    wr(lane, vd, B.ite(res, c(0xFFFFFFFF), c(0)), g)
        elif opcode == 0x98:  # SFPMUL24 (10262-10279)
            mod1 = f["instr_mod1"]
            vd, va, vb = f["lreg_dest"], f["lreg_src_a"], f["lreg_src_b"]
            for lane in range(NLANES):
                g = gate_of(lane)
                a, b_ = rd(lane, va), rd(lane, vb)
                if mod1 & 1:
                    prod = B.mul64(B.band(a, c(0x7FFFFF)), B.band(b_, c(0x7FFFFF)))
                    res = B.trunc32(B.lshr64(prod, 23))
                else:
                    res = B.band(B.trunc32(B.mul64(a, b_)), c(0x7FFFFF))
                wr(lane, vd, res, g)
        elif opcode == 0x99:  # SFPARECIP (10364-10414)
            mod1, vd, vc, imm12 = (
                f["instr_mod1"],
                f["lreg_dest"],
                f["lreg_c"],
                f["imm12_math"],
            )
            if mod1 == 2:
                raise ScopeRefusal("sfparecip-exp")  # approx_exp: add if needed
            for lane in range(NLANES):
                g = gate_of(lane)
                x = rd(lane, vc)
                if mod1 == 0:
                    res = B.bor(
                        B.band(x, c(0x80000000)),
                        approx_recip(B, B.band(x, c(0x7FFFFFFF))),
                    )
                else:  # COND_RECIP
                    vb = imm12 & 0xF
                    neg = B.slt(rd(lane, vb), c(0))
                    res = B.ite(neg, approx_recip(B, B.band(x, c(0x7FFFFFFF))), x)
                wr(lane, vd, res, g)
        else:
            raise ScopeRefusal("opcode-0x%02x" % opcode)

    def _stoch_rnd(self, f, rd, wr, gate_of):
        """SFP_STOCH_RND rnd_mode==0 (deterministic midpoint sample=0x80).
        tensix.cpp:9508-9595 with srnd_round_up_sample(0x80, d, w):
        round up iff top-8-of-discarded >= 0x80 (masked per stoch_mask)."""
        B = self.B
        c = B.const
        if f.get("rnd_mode", 0):
            raise ScopeRefusal("stochrnd-stochastic")
        mod1 = f["instr_mod1"]
        mode = mod1 & 7
        vd, vc, vb = f["lreg_dest"], f["lreg_src_c"], f["lreg_src_b"]
        imm8 = f.get("imm8_math", 0)

        def round_up(discarded, bits):
            # sample = 0x80 midpoint (srnd_round_up_sample, tensix.cpp:2794-2814)
            if bits >= 8:
                trunc = B.band(B.lshr(discarded, bits - 8), c(0xFF))
                mask = 0xFF
            else:
                trunc = B.band(
                    B.shl(B.band(discarded, c((1 << bits) - 1)), 8 - bits), c(0xFF)
                )
                mask = (0xFF << (8 - bits)) & 0xFF
            return B.ule(c(0x80 & mask), B.band(trunc, c(mask)))

        for lane in range(NLANES):
            g = gate_of(lane)
            src = rd(lane, vc)
            if mode in (0, 1):
                exp = B.band(B.lshr(src, 23), c(255))
                dw = 13 if mode == 0 else 16
                dmask = (1 << dw) - 1
                disc = B.band(src, c(dmask))
                base = B.sub(src, disc)
                inc = B.ite(round_up(disc, dw), c(1 << dw), c(0))
                mid = B.add(base, inc)
                res = B.ite(
                    B.eq(exp, c(0)),
                    c(0),
                    B.ite(B.eq(exp, c(255)), B.band(src, c(0xFF800000)), mid),
                )
                wr(lane, vd, res, g)
            elif mode in (2, 3, 6, 7):
                keep_sign = bool(mode & 1)
                maxmag = {6: 65535, 7: 32767, 2: 255, 3: 127}[mode]
                sign = B.band(src, c(0x80000000)) if keep_sign else c(0)
                exp = B.sub(B.band(B.lshr(src, 23), c(255)), c(127))
                mag0 = B.bor(c(0x800000), B.band(src, c(0x7FFFFF)))
                # 64-bit shifted magnitude: exp in [-1, 15] on the live path
                mag64 = B.shl64(B.mul64(mag0, c(1)), 0)
                pos = B.sle(c(0), exp)
                sh = B.band(exp, c(63))
                nsh = B.band(B.sub(c(0), exp), c(63))
                if B.symbolic:
                    mag_s = z3.If(
                        pos,
                        mag64 << z3.ZeroExt(32, Symbolic._bv(sh)),
                        z3.LShR(mag64, z3.ZeroExt(32, Symbolic._bv(nsh))),
                    )
                else:
                    mag_s = (
                        Concrete.shl64(mag64, sh)
                        if pos
                        else Concrete.lshr64(mag64, nsh)
                    )
                int_part = B.trunc32(B.lshr64(mag_s, 23))
                frac = B.band(B.trunc32(mag_s), c(0x7FFFFF))
                mag = B.add(int_part, B.ite(round_up(frac, 23), c(1), c(0)))
                mag = B.ite(B.ult(c(maxmag), mag), c(maxmag), mag)
                signf = B.ite(B.eq(mag, c(0)), c(0), sign)
                mid = B.add(signf, mag)
                res = B.ite(
                    B.slt(exp, c(0xFFFFFFFF)),  # exp < -1
                    c(0),
                    B.ite(B.sle(c(16), exp), B.bor(sign, c(maxmag)), mid),
                )
                wr(lane, vd, res, g)
            else:  # modes 4,5: INT32 -> [U]INT8 descale
                sign = B.band(src, c(0x80000000))
                mag64 = B.shl64(B.mul64(B.band(src, c(0x7FFFFFFF)), c(1)), 23)
                if mod1 & 8:
                    db = c(imm8 & 0xFF)
                else:
                    db = B.band(rd(lane, vb), c(31))
                if B.symbolic:
                    mag_s = z3.LShR(mag64, z3.ZeroExt(32, Symbolic._bv(db)))
                else:
                    mag_s = Concrete.lshr64(mag64, db)
                int_part = B.trunc32(B.lshr64(mag_s, 23))
                frac = B.band(B.trunc32(mag_s), c(0x7FFFFF))
                mag = B.add(int_part, B.ite(round_up(frac, 23), c(1), c(0)))
                if mode == 4:
                    mag = B.ite(B.ult(c(255), mag), c(255), mag)
                    res = mag
                else:
                    mag = B.ite(B.ult(c(127), mag), c(127), mag)
                    signf = B.ite(B.eq(mag, c(0)), c(0), sign)
                    res = B.add(signf, mag)
                wr(lane, vd, res, g)

    def _lutfp32(self, f, rd, wr, gate_of):
        B = self.B
        c = B.const
        mod1, vd = f["instr_mod1"], f["lreg_dest"]
        fp16_gate = bool(mod1 & 2)
        fp16_3entry = (mod1 & 10) == 10
        fp16_6_t2 = fp16_gate and not fp16_3entry and ((mod1 & 3) == 3)
        sgn_retain = bool(mod1 & 4)
        indirect_vd = bool(mod1 & 8)
        if indirect_vd or fp16_3entry:
            raise ScopeRefusal("lutfp32-indirect-vd")
        for lane in range(NLANES):
            g = gate_of(lane)
            l3 = rd(lane, 3)
            b = B.band(l3, c(0x7FFFFFFF))
            lt1 = B.ult(b, c(0x3F800000))
            lt2 = B.ult(b, c(0x40000000))

            def pick(base):
                return B.ite(
                    lt1,
                    rd(lane, base + 0),
                    B.ite(lt2, rd(lane, base + 1), rd(lane, base + 2)),
                )

            if fp16_gate:
                cut = 0x40800000 if fp16_6_t2 else 0x40400000
                j_hi = B.bool_or(
                    B.bool_and(B.ule(c(0x3F000000), b), lt1),
                    B.bool_and(B.ule(c(0x3FC00000), b), lt2),
                    B.bool_and(B.ule(c(cut), b), True),
                )
                # j = 16 in ranges [0.5,1) [1.5,2) [cut,inf); careful: third
                # condition must exclude b<2.0 handled above
                j_hi = B.bool_or(
                    B.bool_and(B.ule(c(0x3F000000), b), B.ult(b, c(0x3F800000))),
                    B.bool_and(B.ule(c(0x3FC00000), b), B.ult(b, c(0x40000000))),
                    B.ule(c(cut), b),
                )
                li = pick(0)
                lc = pick(4)
                a_half = B.ite(j_hi, B.lshr(li, 16), li)
                c_half = B.ite(j_hi, B.lshr(lc, 16), lc)
                a_bits = lut16_to_fp32(B, B.band(a_half, c(0xFFFF)))
                c_bits = lut16_to_fp32(B, B.band(c_half, c(0xFFFF)))
            else:
                a_bits = pick(0)
                c_bits = pick(4)
            d = fma_model_bh(B, a_bits, b, c_bits)
            if sgn_retain:
                d = B.bor(B.band(d, c(0x7FFFFFFF)), B.band(l3, c(0x80000000)))
            wr(lane, vd, d, g)

    def _sfpconfig(self, f):
        """tensix.cpp:9643-9751 (subset: no lane-masked writes)."""
        mod1, dest, imm16 = f["instr_mod1"], f["config_dest"], f["imm16_math"]
        if mod1 & 8:
            raise ScopeRefusal("sfpconfig-lane-masked")
        imm_is_value = bool(mod1 & 1)
        B = self.B
        l00 = self.lanes[0].lregs[0]
        if dest <= 3:
            if not B.is_const(l00):
                raise ScopeRefusal("sfpconfig-symbolic-template")
            self.templates[dest] = B.const_val(l00)
        elif dest <= 7:
            v = imm16 if imm_is_value else B.const_val(l00)
            self.sequences[dest - 4] = v
        elif dest == 8:
            v = imm16 if imm_is_value else B.const_val(l00)
            self.misc = v & 0xFFF
        elif dest in (9, 10):
            raise ScopeRefusal("sfpconfig-dest-%d" % dest)
        elif dest <= 14:
            if imm_is_value:
                val = B.const(SFPCONFIG_CONST[dest])
                for lane in range(NLANES):
                    self.lanes[lane].lregs[dest] = val
            else:
                for lane in range(NLANES):
                    self.lanes[lane].lregs[dest] = self.lanes[lane].lregs[0]
        else:  # 15: LaneConfig
            v = imm16 if imm_is_value else B.const_val(l00)
            for lane in range(NLANES):
                self.lane_config[lane] = v

    # ---- macro events ------------------------------------------------------
    def exec_macro_group(self, events):
        """events: list of parsed M records sharing one group id.
        Snapshot semantics per tensix.cpp:11754-11966 + evaluate_group."""
        B = self.B
        snap_lregs = [
            [self.lanes[l].lregs[r] for r in range(17)] for l in range(NLANES)
        ]
        snap_cc = [self.lanes[l].cc for l in range(NLANES)]
        snap_cc_en = self.cc_en
        writes = []  # (kind, ...) collected then applied
        cc_writes = None
        store_apply = None

        def read_snap(lane, reg):
            return snap_lregs[lane][reg]

        for ev in events:
            ck = ev["ck"]
            if ck == 2:  # NOP
                continue
            if ck == 3:  # store
                tl = ev["tl"]
                mask = [snap_cc[l] if snap_cc_en else True for l in range(NLANES)]
                vals = [snap_lregs[l][tl] for l in range(NLANES)]
                store_apply = (ev, mask, vals)
                continue
            # template event
            local_writes = {}

            def wr_local(lane, reg, val, gate):
                old = local_writes.get((lane, reg), snap_lregs[lane][reg])
                if gate is True:
                    local_writes[(lane, reg)] = val
                else:
                    local_writes[(lane, reg)] = B.ite(gate, val, old)

            snap_mask = [snap_cc[l] if snap_cc_en else True for l in range(NLANES)]
            inst = ev["di"] if not (ev["v16"] or ev["ovb_shft2"]) else ev["tmpl"]
            opcode = (inst >> 24) & 0xFF
            _, name, fields = self.isa.decode(inst)
            if ev["v16"] or ev["ovb_shft2"]:
                # direct evaluator: apply override rules (tensix.cpp:11339+)
                fields = dict(fields)
                tl = ev["tl"]
                if opcode in (0x84, 0x85, 0x86, 0x8E, 0x98):
                    if ev["ovb"]:
                        fields["lreg_src_b"] = ev["vd"]
                    else:
                        fields["lreg_src_c"] = ev["vd"]
                elif opcode == 0x94:
                    # direct SFPSHFT2 (tensix.cpp:11550-11582): override_vb
                    # forces VB<-VD (the VB nibble aliases Imm12, so it can't
                    # be a rewritten word); otherwise VC<-VD.
                    if ev["ovb"]:
                        fields["_vb_force"] = ev["vd"]
                    else:
                        fields["lreg_src_c"] = ev["vd"]
                elif opcode == 0x79:
                    # direct SFPIADD (11477-11511): lreg_b = ovb ? vd : tmpl_vd
                    if ev["ovb"]:
                        fields["_vb_force"] = ev["vd"]
                    else:
                        fields["lreg_c"] = ev["vd"]
                elif opcode in (0x7E, 0x7F):
                    # direct bitops (11512-11540)
                    if ev["ovb"]:
                        fields["_vb_force"] = ev["vd"]
                    else:
                        fields["lreg_c"] = ev["vd"]
                elif opcode == 0x89:
                    # direct SFPSETSGN (11670+): sign source = ovb ? vd : LReg16
                    fields["_vb_force"] = ev["vd"] if ev["ovb"] else 16
                elif opcode in (
                    0x7A,
                    0x7B,
                    0x7C,
                    0x7D,
                    0x80,
                    0x81,
                    0x82,
                    0x83,
                    0x90,
                    0x97,
                    0x99,
                ):
                    if not ev["ovb"]:
                        key = "lreg_src_c" if "lreg_src_c" in fields else "lreg_c"
                        fields[key] = ev["vd"]

                # write target override
                def wr_target(lane, reg, val, gate, _tl=tl):
                    wr_local(lane, _tl, val, gate)

                self.exec_compute(opcode, fields, read_snap, wr_target, snap_mask)
            else:
                self.exec_compute(opcode, fields, read_snap, wr_local, snap_mask)
            # capture CC effects of the event (SETCC-class templates)
            new_cc = [self.lanes[l].cc for l in range(NLANES)]
            if (
                any(new_cc[l] is not snap_cc[l] for l in range(NLANES))
                or self.cc_en != snap_cc_en
            ):
                cc_writes = (self.cc_en, new_cc)
                # restore live CC (deferred visibility)
                for l in range(NLANES):
                    self.lanes[l].cc = snap_cc[l]
                self.cc_en = snap_cc_en
            writes.append(local_writes)

        # merge and apply register writes
        for local in writes:
            for (lane, reg), val in local.items():
                self.lanes[lane].lregs[reg] = val
        # CC deferral (visible only after the C record)
        if events and events[0]["ccd"]:
            if cc_writes is None:
                cc_writes = (self.cc_en, [self.lanes[l].cc for l in range(NLANES)])
            self.pending_cc = cc_writes
        elif cc_writes is not None:
            self.cc_en = cc_writes[0]
            for l in range(NLANES):
                self.lanes[l].cc = cc_writes[1][l]
        # store applies against the snapshot values
        if store_apply is not None:
            ev, mask, vals = store_apply
            flags = (1 if ev["f32"] else 0) | (2 if ev["f32"] else 0)
            for lane in range(NLANES):
                lane_cfg = ev["lc0"] if lane == 0 else ev["lc1"]
                r = ev["dr"]
                rrow = (r & ~3) + lane // 8
                col = 2 * (lane & 7) + (
                    1 if ((r & 2) >> 1) or (lane_cfg & (1 << 7)) else 0
                )
                self.dst_store(
                    rrow, col, ev["sm"], flags, vals[lane], mask[lane], lane_cfg, 0
                )

    def apply_pending_cc(self):
        if self.pending_cc is not None:
            self.cc_en = self.pending_cc[0]
            for l in range(NLANES):
                self.lanes[l].cc = self.pending_cc[1][l]
            self.pending_cc = None


# ---------------------------------------------------------------------------
# small shared helpers (transcribed from tensix.cpp:3565-3830, 8577-8605)
# ---------------------------------------------------------------------------


def decode_bf16(B, x):
    c = B.const
    e = B.band(x, c(255))
    m = B.band(B.lshr(x, 8), c(127))
    return B.bor(B.band(x, c(0x8000)), B.bor(B.shl(e, 7), m))


def encode_bf16(B, x):
    c = B.const
    e = B.band(B.lshr(x, 7), c(255))
    m = B.band(x, c(127))
    return B.bor(B.band(x, c(0x8000)), B.bor(B.shl(m, 8), e))


def decode_fp32(B, x):
    hi = decode_bf16(B, B.lshr(x, 16))
    return B.bor(B.shl(hi, 16), B.band(x, B.const(0xFFFF)))


def encode_fp32(B, x):
    hi = encode_bf16(B, B.lshr(x, 16))
    return B.bor(B.shl(hi, 16), B.band(x, B.const(0xFFFF)))


def encode_fp16(B, x):
    c = B.const
    e = B.band(B.lshr(x, 10), c(31))
    m = B.band(x, c(1023))
    return B.bor(B.band(x, c(0x8000)), B.bor(B.shl(m, 5), e))


def store_to_fp16(B, x):
    """sfpu_store_to_fp16 (tensix.cpp:8577-8589)."""
    c = B.const
    s = B.lshr(x, 31)
    e32 = B.band(B.lshr(x, 23), c(255))
    m = B.band(x, c(0x7FFFFF))
    e = B.sub(e32, c(112))
    lo = B.shl(s, 15)
    hi = B.bor(B.shl(s, 15), c(0x7FFF))
    mid = B.bor(B.shl(s, 15), B.bor(B.shl(B.band(e, c(0x1F)), 10), B.lshr(m, 13)))
    return B.ite(B.sle(e, c(0)), lo, B.ite(B.slt(c(31), e), hi, mid))


def denormals_as_zeros(B, u):
    return B.ite(
        B.eq(B.band(u, B.const(0x7F800000)), B.const(0)),
        B.band(u, B.const(0x80000000)),
        u,
    )


def sign_mag_to_twos(B, x):
    mag = B.band(x, B.const(0x7FFFFFFF))
    return B.ite(
        B.ne(B.band(x, B.const(0x80000000)), B.const(0)), B.sub(B.const(0), mag), mag
    )


def sign_mag8_to_32(B, x):
    s = B.lshr(x, 15)
    mag = B.band(B.lshr(x, 5), B.const(0x7F))
    return B.bor(B.shl(s, 31), mag)


def sign_mag11_to_32(B, x):
    s = B.lshr(x, 15)
    mag = B.band(B.lshr(x, 5), B.const(0x3FF))
    return B.bor(B.shl(s, 31), mag)


def sign_mag16_to_32(B, x):
    s = B.lshr(x, 15)
    mag = B.band(x, B.const(0x7FFF))
    return B.bor(B.shl(s, 31), mag)


def sm_total_order(B, x):
    return B.ite(
        B.ne(B.band(x, B.const(0x80000000)), B.const(0)),
        B.bxor(x, B.const(0x7FFFFFFF)),
        x,
    )


def lut8_to_fp32(B, x):
    c = B.const
    s = B.lshr(x, 7)
    e = B.band(B.lshr(x, 4), c(7))
    m = B.band(x, c(15))
    v = B.bor(B.shl(s, 31), B.bor(B.shl(B.sub(c(127), e), 23), B.shl(m, 19)))
    return B.ite(B.eq(x, c(255)), c(0), v)


def lut16_to_fp32(B, x):
    c = B.const
    s = B.lshr(x, 15)
    e = B.band(B.lshr(x, 10), c(31))
    m = B.band(x, c(0x3FF))
    ee = B.ite(B.eq(e, c(31)), c(0), B.add(c(112), e))
    return B.bor(B.shl(s, 31), B.bor(B.shl(ee, 23), B.shl(m, 13)))


def cc_boolop(B, mod, a, b):
    """sfpu_cc_boolop (tensix.cpp:9318-9334) on bool terms."""
    n = B.bool_not
    return {
        1: b,
        2: n(b),
        3: B.bool_and(a, b),
        4: B.bool_or(a, b),
        5: B.bool_and(a, n(b)),
        6: B.bool_or(a, n(b)),
        7: B.bool_and(n(a), b),
        8: B.bool_or(n(a), b),
        9: B.bool_and(n(a), n(b)),
        10: B.bool_or(n(a), n(b)),
        11: B.bool_or(B.bool_and(a, n(b)), B.bool_and(n(a), b)),
        12: B.bool_not(B.bool_or(B.bool_and(a, n(b)), B.bool_and(n(a), b))),
    }[mod]


def swap_min_mask(mod1):
    return {
        0: 0xFFFFFFFF,
        1: 0xFFFFFFFF,
        2: 0x0000FFFF,
        3: 0x00FF00FF,
        4: 0xFF0000FF,
        5: 0x000000FF,
        6: 0x0000FF00,
        7: 0x00FF0000,
        8: 0xFF000000,
    }.get(mod1, 0x00000000)


# ---------------------------------------------------------------------------
# Trace parsing.
# ---------------------------------------------------------------------------

I_RE = re.compile(
    r"SFPUJO I t=(\d+) p=(\d+) i=([0-9a-f]{8}) a=([0-9a-f]{8}) b=([0-9a-f]{8})"
)
V_RE = re.compile(r"SFPUJO V t=(\d+) en=(\d+) cc=([0-9a-f]{8})((?: [0-9a-f]{8}){18})")
M_RE = re.compile(
    r"SFPUJO M t=(\d+) g=(\d+) su=(\d+) ck=(\d+) tmpl=([0-9a-f]{8}) di=([0-9a-f]{8}) "
    r"ovb=(\d+) v16=(\d+) vd=(\d+) tl=(\d+) sm=(\d+) dr=(\d+) f32=(\d+) "
    r"lc0=([0-9a-f]+) lc1=([0-9a-f]+) ccd=(\d+)"
)
C_RE = re.compile(r"SFPUJO C t=(\d+) cc=([0-9a-f]{8}) en=(\d+)")


def parse_trace(path):
    """Returns list of records: ('I', tile, inst, aux0, aux1), ('M', tile,
    group, dict), ('C', tile, cc, en), ('V', tile, en, cc, [18 words])."""
    recs = []
    with open(path) as fh:
        for line in fh:
            if "SFPUJO" not in line:
                continue
            m = I_RE.search(line)
            if m:
                recs.append(
                    (
                        "I",
                        int(m.group(1)),
                        int(m.group(3), 16),
                        int(m.group(4), 16),
                        int(m.group(5), 16),
                    )
                )
                continue
            m = V_RE.search(line)
            if m:
                words = [int(w, 16) for w in m.group(4).split()]
                recs.append(
                    ("V", int(m.group(1)), int(m.group(2)), int(m.group(3), 16), words)
                )
                continue
            m = M_RE.search(line)
            if m:
                d = dict(
                    g=int(m.group(2)),
                    su=int(m.group(3)),
                    ck=int(m.group(4)),
                    tmpl=int(m.group(5), 16),
                    di=int(m.group(6), 16),
                    ovb=bool(int(m.group(7))),
                    v16=bool(int(m.group(8))),
                    vd=int(m.group(9)),
                    tl=int(m.group(10)),
                    sm=int(m.group(11)),
                    dr=int(m.group(12)),
                    f32=bool(int(m.group(13))),
                    lc0=int(m.group(14), 16),
                    lc1=int(m.group(15), 16),
                    ccd=bool(int(m.group(16))),
                )
                d["ovb_shft2"] = ((d["tmpl"] >> 24) & 0xFF) == 0x94 and d["ovb"]
                recs.append(("M", int(m.group(1)), d))
                continue
            m = C_RE.search(line)
            if m:
                recs.append(
                    ("C", int(m.group(1)), int(m.group(2), 16), int(m.group(3)))
                )
    return recs


# ---------------------------------------------------------------------------
# Trace replay (concrete validation / symbolic execution share this driver).
# ---------------------------------------------------------------------------


def run_trace(
    recs,
    B,
    dst,
    tile=None,
    validate=False,
    tag="",
    journal=None,
    isa=None,
    stop_at=None,
):
    isa = isa or Isa()
    ex = Executor(B, isa, dst, tag, journal=journal)
    pending_group = []
    pending_gid = None
    checked = 0

    # pair each I record with its immediately-following V record (same tile)
    recs = [r for r in recs if tile is None or r[1] == tile]
    next_v = [None] * len(recs)
    last_v = None
    for i in range(len(recs) - 1, -1, -1):
        if recs[i][0] == "V":
            last_v = recs[i][4]
        next_v[i] = last_v

    def flush_group():
        nonlocal pending_group, pending_gid
        if pending_group:
            ex.exec_macro_group(pending_group)
            pending_group, pending_gid = [], None

    for i, rec in enumerate(recs):
        kind = rec[0]
        if kind == "M":
            d = rec[2]
            if pending_gid is not None and d["g"] != pending_gid:
                flush_group()
            pending_gid = d["g"]
            pending_group.append(d)
            continue
        flush_group()
        if kind == "I":
            if stop_at is not None and ex.n_exec >= stop_at:
                break
            _, _, inst, aux0, aux1 = rec
            ex.current_v = next_v[i] if not B.symbolic else None
            ex.exec_inst(inst, aux0, aux1)
            ex.current_v = None
        elif kind == "C":
            ex.apply_pending_cc()
        elif kind == "V":
            _, _, en, cc, words = rec
            if validate:
                _check_v(ex, en, cc, words, checked, tag)
                checked += 1
    flush_group()
    return ex, checked


def _adopt_cells(ex, rrow, col, mod0, aux1, value, cur_lreg):
    """Concrete mode: adopt the dst cells feeding a traced load — recover
    the raw 16-bit cells from the traced loaded value (inverse of the load
    conversion) so later reads/stores see consistent memory."""
    keys = ex.dst_load_keys(rrow, col, mod0, aux1)
    if mod0 in (3, 4, 10, 12):
        raw = encode_fp32(Concrete, value)
        raws = [(raw >> 16) & 0xFFFF, raw & 0xFFFF]
    elif mod0 == 2:
        raws = [encode_bf16(Concrete, (value >> 16) & 0xFFFF)]
    elif mod0 == 1:
        s = (value >> 31) & 1
        e = (value >> 23) & 0xFF
        m = (value >> 13) & 0x3FF
        e5 = e - 112 if e else 0
        raws = [(s << 15) | (m << 5) | (e5 & 31)]
    elif mod0 in (6, 9):
        raws = [value & 0xFFFF]
    elif mod0 == 7:
        raws = [(value >> 16) & 0xFFFF]
    elif mod0 == 5:
        raws = [(((value >> 31) & 1) << 15) | ((value & 0x7F) << 5)]
    elif mod0 == 8:
        raws = [(((value >> 31) & 1) << 15) | (value & 0x7FFF)]
    elif mod0 == 13:
        if value & 0x80000000:
            raws = [(1 << 15) | ((((-value) & 0x3FF)) << 5)]
        else:
            raws = [(value & 0x3FF) << 5]
    elif mod0 == 14:
        raws = [value & 0xFFFF]
    elif mod0 == 15:
        raws = [(value >> 16) & 0xFFFF]
    else:
        raise ScopeRefusal("adopt-mod0-%d" % mod0)
    for k, r in zip(keys, raws):
        prev = ex.dst.cells.get(k)
        if prev is None or prev != (r & 0xFFFF):
            ex.adoptions.append((ex.n_exec, k, r & 0xFFFF))
        ex.dst.cells[k] = r & 0xFFFF
        ex.dst.written.discard(k)


def _check_v(ex, en, cc, words, idx, tag):
    if bool(en) != bool(ex.cc_en):
        raise ValidationError(
            "%s V#%d cc_en sim=%d model=%d" % (tag, idx, en, ex.cc_en)
        )
    for lane in range(NLANES):
        sim_cc = bool((cc >> lane) & 1)
        model_cc = ex.lanes[lane].cc
        if not isinstance(model_cc, bool):
            raise ValidationError("%s V#%d lane%d cc not concrete" % (tag, idx, lane))
        if sim_cc != model_cc:
            raise ValidationError(
                "%s V#%d lane%d cc sim=%d model=%d" % (tag, idx, lane, sim_cc, model_cc)
            )
        for r in range(8):
            sim_v = words[lane * 8 + r]
            mod_v = ex.lanes[lane].lregs[r]
            if isinstance(mod_v, tuple):
                continue  # unresolved adopt on an lreg not checked this V
            if mod_v != sim_v:
                raise ValidationError(
                    "%s V#%d lane%d lreg%d sim=%08x model=%08x (after %d instrs)"
                    % (tag, idx, lane, r, sim_v, mod_v, ex.n_exec)
                )
        sim_16 = words[16 + lane]
        mod_16 = ex.lanes[lane].lregs[16]
        if not isinstance(mod_16, tuple) and mod_16 != sim_16:
            raise ValidationError(
                "%s V#%d lane%d lreg16 sim=%08x model=%08x"
                % (tag, idx, lane, sim_16, mod_16)
            )


# ---------------------------------------------------------------------------
# Equivalence query.
# ---------------------------------------------------------------------------


def align_generations(adopt_sem, adopt_hand):
    """Turn each leg's input-adoption event list [(n_exec, cell, value)...]
    into a symbolic journal {n_exec: {cell: canonical_gen}}.  Generations are
    aligned ACROSS legs by the observed VALUE SEQUENCE per cell (both legs run
    the same stimuli, so the FPU delivers the same value sequence; a leg whose
    predicated loads miss an epoch simply skips that generation)."""
    per_cell = {}
    for tag, events in (("sem", adopt_sem), ("hand", adopt_hand)):
        for n, cell, val in events:
            per_cell.setdefault(cell, {"sem": [], "hand": []})[tag].append(val)

    def is_subseq(short, long):
        it = iter(long)
        return all(any(v == w for w in it) for v in short)

    canon = {}
    for cell, seqs in per_cell.items():
        a, b = seqs["sem"], seqs["hand"]
        long_seq = a if len(a) >= len(b) else b
        short_seq = b if len(a) >= len(b) else a
        if not is_subseq(short_seq, long_seq):
            raise ScopeRefusal("gen-align-%s" % (cell,))
        canon[cell] = long_seq

    # The FPU redelivers ALL input tiles together at each block boundary, so
    # a generation advance observed at ANY cell advances the GLOBAL epoch: a
    # leg whose predicated loads never observe some cell's new value (e.g.
    # lanes masked off in the modeled lanes) still must bind that cell's
    # current-epoch symbol.  A cell whose value did not change across a
    # redelivery keeps its shorter canonical sequence; its generation
    # saturates at its last distinct value (bitwise-equal, so symbol
    # identification is exact there).
    def build(events):
        journal = {}
        ptr = {}
        epoch = 0
        for n, cell, val in events:
            seq = canon[cell]
            i = ptr.get(cell, 0)
            while i < len(seq) and seq[i] != val:
                i += 1
            if i >= len(seq):
                raise ScopeRefusal("gen-align-scan-%s" % (cell,))
            journal.setdefault(n, {})[cell] = i
            ptr[cell] = i + 1
            if i > epoch:
                epoch = i
                for c2, seq2 in canon.items():
                    journal[n].setdefault(c2, min(epoch, len(seq2) - 1))
        return journal

    return build(adopt_sem), build(adopt_hand)


def canon_hash(se, he):
    """Cheap structural hash of the (sem, hand) pair with symbols renamed by
    first occurrence — dedupes per-datum queries without z3.simplify (which
    is quadratic-ish on deep FP-approximation chains)."""
    import hashlib

    h = hashlib.sha256()
    symmap = {}
    for expr in (se, he):
        seen = {}
        stack = [(expr, False)]
        order = 0
        while stack:
            e, done = stack.pop()
            eid = e.get_id()
            if eid in seen:
                h.update(b"R%d" % seen[eid])
                continue
            seen[eid] = order
            order += 1
            d = e.decl()
            k = d.kind()
            if k == z3.Z3_OP_UNINTERPRETED:
                idx = symmap.setdefault(d.name(), len(symmap))
                h.update(b"S%d" % idx)
                continue
            if k == z3.Z3_OP_BNUM:
                h.update(b"N%d:%d" % (e.size(), e.as_long()))
                continue
            params = ",".join(str(p) for p in d.params())
            h.update(("O%d(%s)%d" % (k, params, e.num_args())).encode())
            for c in reversed(e.children()):
                stack.append((c, False))
        h.update(b"|")
    return h.digest()


def canonicalize(expr_pair, symbols_order):
    """Rename the dst symbols appearing in a (sem, hand) expression pair to
    canonical x0,x1,... in first-use order for structural dedup."""
    sem, hand = expr_pair
    seen = []
    for s in symbols_order:
        seen.append(s)
    subs = [(s, z3.BitVec("x%d" % i, 16)) for i, s in enumerate(seen)]
    return (
        z3.substitute(sem, *subs).sexpr() if subs else sem.sexpr(),
        z3.substitute(hand, *subs).sexpr() if subs else hand.sexpr(),
    )


def collect_symbols(expr):
    out = OrderedDict()
    stack = [expr]
    seen_ids = set()
    while stack:
        e = stack.pop()
        if e.get_id() in seen_ids:
            continue
        seen_ids.add(e.get_id())
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            out[e.decl().name()] = e
        else:
            stack.extend(e.children())
    return out


def domain_constraints(symbols, domain, tags=None):
    """Documented-contract input-domain constraints (binary corpus layout:
    tile pairs stride 128 pre-adjust rows; operand A rows have
    (row % 128) < 64, operand B rows >= 64).  Entries:
      {"which": "a"|"b"|"all", "mag_lt": N}          |value| & 0x7FFFFFFF < N
      {"which": ..., "int_min": lo, "int_max": hi}   signed int32 range
    applied to every complete 32-bit input datum (hi/lo cell pair)."""
    cons = []
    B = Symbolic
    # 16-bit bf16 datums (cells with no hi/lo partner): the sim-measured
    # DELIVERABLE set through unpack->datacopy is {+-normals, +0, +-Inf} —
    # denormals and -0 flush to +0 and every NaN payload canonicalizes to
    # +Inf (laneJO copy_dest identity probe, 2026-08-31).
    tags = tags or {}
    for (adjr, col, gen), sym in symbols.items():
        if tags.get((adjr, col)) != "16":
            continue
        for ent in domain:
            if not ent.get("bf16_deliverable"):
                continue
            d = decode_bf16(B, z3.ZeroExt(16, sym))
            e = d & z3.BitVecVal(0x7F80, 32)
            m = d & z3.BitVecVal(0x7F, 32)
            zero = z3.BitVecVal(0, 32)
            not_denorm = z3.Or(e != zero, d == zero)
            not_nan = z3.Or(e != z3.BitVecVal(0x7F80, 32), m == zero)
            cons.append(z3.And(not_denorm, not_nan))
    for (adjr, col, gen), hi_sym in symbols.items():
        if tags.get((adjr, col)) != "32hi":
            continue
        lo_key = (adjr + 8, col, gen)
        if lo_key not in symbols:
            continue
        lo_sym = symbols[lo_key]
        row_pre = ((adjr >> 1) & 0x1F8) | (adjr & 0x7)
        is_b = (row_pre % 128) >= 64
        v32 = decode_fp32(
            B, B.bor(B.shl(z3.ZeroExt(16, hi_sym), 16), z3.ZeroExt(16, lo_sym))
        )
        for ent in domain:
            which = ent.get("which", "all")
            if which == "a" and is_b:
                continue
            if which == "b" and not is_b:
                continue
            if "mag_lt" in ent:
                cons.append(
                    z3.ULT(
                        v32 & z3.BitVecVal(0x7FFFFFFF, 32),
                        z3.BitVecVal(ent["mag_lt"], 32),
                    )
                )
            if "int_min" in ent:
                cons.append(v32 >= z3.BitVecVal(ent["int_min"] & 0xFFFFFFFF, 32))
                cons.append(v32 <= z3.BitVecVal(ent["int_max"] & 0xFFFFFFFF, 32))
    return cons


# ---------------------------------------------------------------------------
# Exhaustive fallback: compile a z3 QF_BV term over ONE 16-bit symbol to a
# vectorized numpy function and evaluate all 2^16 inputs.  For bf16-input
# rows this decides the query exactly (and yields divergence COUNTS) far
# faster than bit-blasting deep FP-approximation chains.
# ---------------------------------------------------------------------------


def z3_to_numpy(expr, sym):
    import numpy as np

    K = z3.Z3_OP_UNINTERPRETED
    memo = {}

    def mask(width):
        return (
            np.uint64((1 << width) - 1) if width < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
        )

    def ev(e, x):
        key = e.get_id()
        if key in memo:
            return memo[key]
        d = e.decl()
        k = d.kind()
        if k == z3.Z3_OP_BNUM:
            r = np.uint64(e.as_long())
        elif k == K:
            if d.name() != sym:
                raise ValueError("unexpected symbol " + d.name())
            r = x
        else:
            ch = [ev(c, x) for c in e.children()]
            w = e.size() if z3.is_bv(e) else None
            if k == z3.Z3_OP_BADD:
                r = ch[0]
                for c in ch[1:]:
                    r = (r + c) & mask(w)
            elif k == z3.Z3_OP_BSUB:
                r = (ch[0] - ch[1]) & mask(w)
            elif k == z3.Z3_OP_BMUL:
                r = ch[0]
                for c in ch[1:]:
                    r = (r * c) & mask(w)
            elif k == z3.Z3_OP_BAND:
                r = ch[0]
                for c in ch[1:]:
                    r = r & c
            elif k == z3.Z3_OP_BOR:
                r = ch[0]
                for c in ch[1:]:
                    r = r | c
            elif k == z3.Z3_OP_BXOR:
                r = ch[0]
                for c in ch[1:]:
                    r = r ^ c
            elif k == z3.Z3_OP_BNOT:
                r = (~ch[0]) & mask(w)
            elif k == z3.Z3_OP_BSHL:
                sh = np.minimum(ch[1], np.uint64(63))
                r = (ch[0] << sh) & mask(w)
                r = np.where(ch[1] >= np.uint64(w), np.uint64(0), r)
            elif k == z3.Z3_OP_BLSHR:
                sh = np.minimum(ch[1], np.uint64(63))
                r = ch[0] >> sh
                r = np.where(ch[1] >= np.uint64(w), np.uint64(0), r)
            elif k == z3.Z3_OP_BASHR:
                wm = e.children()[0].size()
                sign = (ch[0] >> np.uint64(wm - 1)) & np.uint64(1)
                sh = np.minimum(ch[1], np.uint64(wm - 1))
                ext = ((np.uint64(1) << sh) - np.uint64(1)) << (np.uint64(wm) - sh)
                r = ((ch[0] >> sh) | np.where(sign != 0, ext, np.uint64(0))) & mask(wm)
            elif k == z3.Z3_OP_CONCAT:
                r = ch[0]
                for c, cc in zip(ch[1:], e.children()[1:]):
                    r = ((r << np.uint64(cc.size())) | c) & mask(w)
            elif k == z3.Z3_OP_EXTRACT:
                hi, lo = d.params()
                r = (ch[0] >> np.uint64(lo)) & mask(hi - lo + 1)
            elif k == z3.Z3_OP_ZERO_EXT:
                r = ch[0]
            elif k == z3.Z3_OP_SIGN_EXT:
                wm = e.children()[0].size()
                sign = (ch[0] >> np.uint64(wm - 1)) & np.uint64(1)
                ext = (mask(w) >> np.uint64(wm)) << np.uint64(wm)
                r = ch[0] | np.where(sign != 0, ext, np.uint64(0))
            elif k == z3.Z3_OP_ITE:
                r = np.where(ch[0], ch[1], ch[2])
            elif k == z3.Z3_OP_EQ:
                r = ch[0] == ch[1]
            elif k == z3.Z3_OP_DISTINCT:
                r = ch[0] != ch[1]
            elif k == z3.Z3_OP_ULT:
                r = ch[0] < ch[1]
            elif k == z3.Z3_OP_ULEQ:
                r = ch[0] <= ch[1]
            elif k == z3.Z3_OP_UGT:
                r = ch[0] > ch[1]
            elif k == z3.Z3_OP_UGEQ:
                r = ch[0] >= ch[1]
            elif k in (z3.Z3_OP_SLT, z3.Z3_OP_SLEQ, z3.Z3_OP_SGT, z3.Z3_OP_SGEQ):
                wm = e.children()[0].size()
                half = np.uint64(1 << (wm - 1))
                a = ch[0] ^ half
                b = ch[1] ^ half
                r = {
                    z3.Z3_OP_SLT: a < b,
                    z3.Z3_OP_SLEQ: a <= b,
                    z3.Z3_OP_SGT: a > b,
                    z3.Z3_OP_SGEQ: a >= b,
                }[k]
            elif k == z3.Z3_OP_NOT:
                r = ~ch[0]
            elif k == z3.Z3_OP_AND:
                r = ch[0]
                for c in ch[1:]:
                    r = r & c
            elif k == z3.Z3_OP_OR:
                r = ch[0]
                for c in ch[1:]:
                    r = r | c
            elif k == z3.Z3_OP_TRUE:
                r = np.bool_(True)
            elif k == z3.Z3_OP_FALSE:
                r = np.bool_(False)
            else:
                raise ValueError("z3->numpy: unhandled op kind %d (%s)" % (k, d.name()))
        memo[key] = r
        return r

    def run(x):
        memo.clear()
        return ev(expr, x)

    return run


def exhaustive16(se, he, sym_name, constraints=()):
    """Evaluate both exprs over all 2^16 values of the single 16-bit symbol
    (restricted to any domain constraints).  Returns (n_diff, witness|None)."""
    import numpy as np

    x = np.arange(65536, dtype=np.uint64)
    vs = z3_to_numpy(se, sym_name)(x) & np.uint64(0xFFFF)
    vh = z3_to_numpy(he, sym_name)(x) & np.uint64(0xFFFF)
    diff = vs != vh
    for con in constraints:
        diff = diff & z3_to_numpy(con, sym_name)(x)
    n = int(diff.sum())
    if n == 0:
        return 0, None
    i = int(np.argmax(diff))
    return n, {
        "inputs": {sym_name: i},
        "sem_out16": "0x%04x" % int(vs[i]),
        "hand_out16": "0x%04x" % int(vh[i]),
    }


def prove_row(
    sem_final, hand_final, symbols, timeout_ms, out_dir, row, domain=None, tags=None
):
    """sem_final/hand_final: dict ((adj_row,col), gen) -> 16-bit BV expr.
    Returns (verdict, details)."""
    all_cells = sorted(set(sem_final) | set(hand_final))
    only_sem = [k for k in all_cells if k not in hand_final]
    only_hand = [k for k in all_cells if k not in sem_final]
    queries = OrderedDict()  # canonical key -> (cell, sem_expr, hand_expr, sym_map)
    n_trivial = 0
    for cell in all_cells:
        se = sem_final.get(cell)
        he = hand_final.get(cell)
        skey = (cell[0][0], cell[0][1], cell[1])
        if se is None:
            se = symbols.get(skey)  # untouched by sem: still the input
        if he is None:
            he = symbols.get(skey)
        if se is None or he is None:
            continue
        se = Symbolic._bv(se, 16) if isinstance(se, int) else se
        he = Symbolic._bv(he, 16) if isinstance(he, int) else he
        if se.eq(he):
            n_trivial += 1
            continue
        key = canon_hash(se, he)
        if key not in queries:
            queries[key] = (cell, se, he, None)
    details = {
        "cells": len(all_cells),
        "trivially_equal": n_trivial,
        "unique_queries": len(queries),
        "only_sem": len(only_sem),
        "only_hand": len(only_hand),
        "solver_times": [],
    }
    cons_by_sym = {}
    if domain:
        for con in domain_constraints(symbols, domain, tags):
            for nm in collect_symbols(con):
                cons_by_sym.setdefault(nm, []).append(con)
    verdict = (
        "PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN" if domain else "PROVEN-EQUIV-ALL-INPUTS"
    )
    witness = None
    for qi, (key, (cell, se, he, order)) in enumerate(queries.items()):
        syms_all = collect_symbols(se)
        syms_all.update(collect_symbols(he))
        if len(syms_all) == 1 and next(iter(syms_all.values())).size() == 16:
            nm = next(iter(syms_all))
            t0 = time.time()
            try:
                n_diff, wit16 = exhaustive16(
                    se, he, nm, [c for c in cons_by_sym.get(nm, ())]
                )
            except (ValueError, ImportError) as exc:
                n_diff, wit16 = None, None
                details.setdefault("exhaustive_errors", []).append(str(exc))
            if n_diff is not None:
                dt = time.time() - t0
                details["solver_times"].append(
                    (str(cell), "exhaustive16:%d-diffs" % n_diff, round(dt, 3))
                )
                if n_diff:
                    verdict = "DIVERGENT"
                    wit16["cell"] = list(cell)
                    wit16["diff_count_2^16"] = n_diff
                    witness = wit16
                    break
                continue
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(se != he)
        if cons_by_sym:
            seen = set()
            syms = collect_symbols(se)
            syms.update(collect_symbols(he))
            for nm in syms:
                for con in cons_by_sym.get(nm, ()):
                    if id(con) not in seen:
                        seen.add(id(con))
                        s.add(con)
        smt2_path = os.path.join(out_dir, "%s-q%d.smt2" % (row, qi))
        with open(smt2_path, "w") as fh:
            fh.write("; row=%s cell=%s\n" % (row, cell))
            fh.write(s.to_smt2())
        t0 = time.time()
        res = s.check()
        dt = time.time() - t0
        details["solver_times"].append((str(cell), str(res), round(dt, 3)))
        if res == z3.sat:
            m = s.model()
            assignment = {d.name(): m[d].as_long() for d in m.decls()}
            sev = m.eval(se, model_completion=True).as_long()
            hev = m.eval(he, model_completion=True).as_long()
            verdict = "DIVERGENT"
            witness = {
                "cell": list(cell),
                "inputs": assignment,
                "sem_out16": "0x%04x" % sev,
                "hand_out16": "0x%04x" % hev,
            }
            break
        if res == z3.unknown:
            verdict = "UNDECIDED"
            break
    details["witness"] = witness
    if only_sem or only_hand:
        details["note"] = "asymmetric write sets (sem-only %d, hand-only %d cells)" % (
            len(only_sem),
            len(only_hand),
        )
    return verdict, details


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row", required=True)
    ap.add_argument("--trace-sem", required=True)
    ap.add_argument("--trace-hand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument(
        "--timeout", type=int, default=3600, help="z3 timeout per query (s)"
    )
    ap.add_argument(
        "--first-epoch",
        action="store_true",
        help="compare only the first input epoch (truncate each trace before "
        "its first generation-1 adoption; sidesteps cross-epoch alignment)",
    )
    ap.add_argument(
        "--domain-json",
        default=None,
        help="documented-contract domain entries (JSON list); "
        "verdict becomes *-ON-DOCUMENTED-DOMAIN",
    )
    ap.add_argument("--isa-json", default=DEFAULT_ISA_JSON)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    result = {"row": args.row}
    t_start = time.time()

    recs_sem = parse_trace(args.trace_sem)
    recs_hand = parse_trace(args.trace_hand)
    result["n_records"] = {"sem": len(recs_sem), "hand": len(recs_hand)}

    # 1) concrete validation gate (also records the input-adoption events)
    try:
        dst_c1 = Dst(Concrete)
        ex1, chk1 = run_trace(
            recs_sem, Concrete, dst_c1, tile=args.tile, validate=True, tag="sem"
        )
        dst_c2 = Dst(Concrete)
        ex2, chk2 = run_trace(
            recs_hand, Concrete, dst_c2, tile=args.tile, validate=True, tag="hand"
        )
        journal_sem, journal_hand = align_generations(ex1.adoptions, ex2.adoptions)
        result["validation"] = {
            "sem_checkpoints": chk1,
            "hand_checkpoints": chk2,
            "sem_adoptions": len(ex1.adoptions),
            "hand_adoptions": len(ex2.adoptions),
            "status": "VALIDATED",
        }
    except (ValidationError, ScopeRefusal) as e:
        result["validation"] = {"status": "SEMANTICS-UNVALIDATED", "error": str(e)}
        result["verdict"] = "SEMANTICS-UNVALIDATED"
        _emit(args, result, t_start)
        return 1

    # 2) symbolic runs over a shared symbol table
    if z3 is None:
        result["verdict"] = "NO-Z3"
        _emit(args, result, t_start)
        return 1
    symbols = {}
    stop_sem = stop_hand = None
    if args.first_epoch:
        gen1_sem = [
            n for n, m in journal_sem.items() if any(g >= 1 for g in m.values())
        ]
        gen1_hand = [
            n for n, m in journal_hand.items() if any(g >= 1 for g in m.values())
        ]
        stop_sem = min(gen1_sem) if gen1_sem else None
        stop_hand = min(gen1_hand) if gen1_hand else None
        journal_sem = {
            n: m for n, m in journal_sem.items() if stop_sem is None or n < stop_sem
        }
        journal_hand = {
            n: m for n, m in journal_hand.items() if stop_hand is None or n < stop_hand
        }
        result["first_epoch"] = {"stop_sem": stop_sem, "stop_hand": stop_hand}
    try:
        dst_sem = Dst(Symbolic, symbols)
        run_trace(
            recs_sem,
            Symbolic,
            dst_sem,
            tile=args.tile,
            validate=False,
            tag="sem",
            journal=journal_sem,
            stop_at=stop_sem,
        )
        dst_hand = Dst(Symbolic, symbols)
        run_trace(
            recs_hand,
            Symbolic,
            dst_hand,
            tile=args.tile,
            validate=False,
            tag="hand",
            journal=journal_hand,
            stop_at=stop_hand,
        )
    except ScopeRefusal as e:
        result["verdict"] = "SCOPE-REFUSED"
        result["refusal"] = str(e)
        _emit(args, result, t_start)
        return 1

    sem_final = dst_sem.final_outputs()
    hand_final = dst_hand.final_outputs()
    domain = json.loads(args.domain_json) if args.domain_json else None
    tags = dict(dst_hand.tags)
    tags.update(dst_sem.tags)
    verdict, details = prove_row(
        sem_final,
        hand_final,
        symbols,
        args.timeout * 1000,
        args.out,
        args.row,
        domain=domain,
        tags=tags,
    )
    if domain:
        result["domain"] = domain
    result["verdict"] = verdict
    result["details"] = details
    _emit(args, result, t_start)
    return 0 if verdict == "PROVEN-EQUIV-ALL-INPUTS" else 2


def _emit(args, result, t_start):
    result["wall_s"] = round(time.time() - t_start, 2)
    path = os.path.join(args.out, "%s-verdict.json" % args.row)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
