# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Code layout pads for the measured loop of Wormhole perf kernels (LLK_DBG_BARRIER builds, see barrier.h): the NOP pads
P and Z (llk_loop_pad, llk_loop_end_pad) with the lowest modelled branch predictor and icache cost in the window.
"""
import fcntl
import functools
import hashlib
import json
import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

PAD_STEP = 4
# P covers the restart's address modulo the predictor and cache periods: 512 B on math (its restart is 1 KiB aligned
# inline, 512 B out of line, barrier.h), 1 KiB on the others (1 KiB aligned restart, 64 cache sets of 16 B)
P_SPAN = {"math": 512}
P_SPAN_DEFAULT = 1024
# callees vs the loop modulo the instruction cache period: 256 B on math, 1 KiB on the others
Z_SPAN = {"math": 512}
Z_SPAN_DEFAULT = 1024
MIN_WINDOW = 4000  # instructions modelled at least (or the whole window)
MAX_WINDOW = 60000  # and at most
TRACE_BUDGET = 4_000_000  # interpreted instructions before a kernel is given up on
MIS_COST, MISS_COST, HAZ_COST = 2.0, 7.0, 1.0
# every candidate class is evaluated when classes x window instructions is at most this
EXHAUSTIVE = 16384 * 6000
STARTS = 8  # P (then Z) values the coordinate search restarts from
EVAL_CHUNK = 1024  # candidates costed at once
BOUND_STEP = 16  # resolution of the largest pads that keep the code
MAX_VERIFY = 8  # chosen pads checked on their real link before the search gives up
# shorter windows are not padded: their time is mostly the start after the restart
MIN_WINDOW_INSTR = 200
K_PRED = 3  # a predictor update is seen by the predictions this many instructions later
SPIN_SPAN = 32  # only short backward branches can be spin waits
CLOCK_LO = 0xFFB121F0
EBREAK = 0x00100073
FILL = (0x00000013, 0x00000000)  # NOPs and the linker's zero fill
PARK_FIXED = 64  # NOP words between a park's ebreak and its alignment (barrier.h)
ZONE_RESERVE = "_ZN12llk_profiler12zone_reserveEv"
ZONE_RECORD = "_ZN12llk_profiler11zone_recordEtyy"
RUN_KERNEL = "_Z10run_kernelRK13RuntimeParams"
PARAMS, STACK, RETURN = 0x70000000, 0x7FFF0000, 0x7EEE0000
M32 = 0xFFFFFFFF
MAP_VERSION = 12
ALIGNS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _sext(x, bits):
    return x - (1 << bits) if x & (1 << (bits - 1)) else x


def _s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


class Elf:
    """Loaded sections, code words and symbols of a little endian ELF32."""

    def __init__(self, path):
        data = Path(path).read_bytes()
        (shoff,) = struct.unpack_from("<I", data, 0x20)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x2E)
        heads = [
            struct.unpack_from("<IIIIIIIIII", data, shoff + i * shentsize)
            for i in range(shnum)
        ]
        name = lambda off, tab: data[
            tab[4] + off : data.index(b"\0", tab[4] + off)
        ].decode()
        self.sections, self.symbols, self.region_end, self.code = {}, {}, None, []
        for h in heads:
            n = name(h[0], heads[shstrndx])
            if n == ".loader_init":  # placed right after the TRISC code region
                self.region_end = h[3]
            if h[2] & 2 and h[1] != 8:  # SHF_ALLOC, not SHT_NOBITS
                self.sections[n] = (h[3], data[h[4] : h[4] + h[5]])
                if h[2] & 4 and h[5] >= 4:  # SHF_EXECINSTR
                    self.code.append((h[3], data[h[4] : h[4] + h[5]]))
            if h[1] == 2:  # SHT_SYMTAB
                for off in range(h[4], h[4] + h[5], 16):
                    st_name, st_value, st_size = struct.unpack_from("<III", data, off)
                    self.symbols[name(st_name, heads[h[6]])] = (st_value, st_size)
        self.text_end = max(a + len(d) for a, d in self.code)

    def word(self, addr):
        for a, d in self.sections.values():
            if a <= addr < a + len(d) - 3:
                return struct.unpack_from("<I", d, addr - a)[0]
        return None

    def instructions(self):
        """(addresses, words) of every code word that is not fill, in address order"""
        addrs, words = [], []
        for a, d in sorted(self.code):
            w = np.frombuffer(d[: len(d) & ~3], dtype="<u4").astype(np.int64)
            keep = (w != FILL[0]) & (w != FILL[1])
            addrs.append(a + 4 * np.flatnonzero(keep))
            words.append(w[keep])
        return np.concatenate(addrs).astype(np.int64), np.concatenate(words)


def _norm(words):
    """instruction words without their immediates, which change when code moves (branches, jal, auipc, lui, la)"""
    op = words & 0x7F
    return np.where(
        np.isin(op, (0x63, 0x23)),
        words & 0x01FFF07F,
        np.where(
            np.isin(op, (0x6F, 0x17, 0x37)),
            words & 0xFFF,
            np.where(np.isin(op, (0x13, 0x03, 0x67)), words & 0xFFFFF, words),
        ),
    )


def _apply(model, s_in, P, Z):
    """shift after a gap, from the shift before it, for candidate pads (P, Z)"""
    cp, cz, A, f, mode = model
    ins = cp * P + cz * Z
    if not A:
        return s_in + ins
    # a fill that keeps its section's size modulo A: it absorbs the inserted pad bytes
    if mode == 2:
        return s_in + np.mod(f - ins, A) - f
    x = s_in + ins if mode else s_in
    return x + np.mod(f - x, A) - f + (0 if mode else ins)


class AddressMap:
    """Where each code address of the unpadded ELF lands under pads (P, Z), fitted from probe links: each gap where the
    shift changes is an insertion of P or Z and/or an alignment (cP, cZ = -1: a relaxed section's fill absorbs pads).
    """

    def __init__(self, base, period=None):
        # of the cost: a gap no exact model fits may be fitted modulo it
        self.period = period
        self.mod_gaps = set()
        self.addr, words = base.instructions()
        self.norm = _norm(words)
        self.words = words
        self.obs = {}
        self.gaps = []  # [(instruction index, [models])]
        self.rk = base.symbols.get(RUN_KERNEL)
        self.parks = _park_restarts(self.addr, words, self.rk)
        # ordinal of the TILE_LOOP park among self.parks, once found
        self.loop_park = None
        # index of its restart (the first instruction that moves with P)
        self.restart = None

    def find_loop_park(self, probes):
        """the TILE_LOOP park from probe links {(P, Z): Elf}: the park whose restart is 512 B aligned without pads and
        moves by exactly P in every probe with P > 0; None when no single park does (then only identical links count)
        """
        cands = [k for k, i in enumerate(self.parks) if self.addr[i] % 512 == 0]
        for pz, elf in probes.items():
            if elf is None or not pz[0]:
                continue
            a, w = elf.instructions()
            parks = _park_restarts(a, w, elf.symbols.get(RUN_KERNEL))
            if len(parks) != len(self.parks):
                return None
            cands = [
                k
                for k in cands
                if _whole_moves(a[parks[k]] - self.addr[self.parks[k]] - pz[0])
                is not None
            ]
        # the parks after the TILE_LOOP one move with it: the first is the TILE_LOOP park
        if cands:
            self.loop_park = cands[0]
            self.restart = self.parks[cands[0]]
        return self.loop_park

    def pair(self, pz, elf):
        """(shift of every instruction, whole) in a link with pads pz, or None when its code differs from the restart on;
        whole > 0: a relaxed branch before the park moved the restart by whole periods more (same layout modulo them)
        """
        a, w = elf.instructions()
        nw = _norm(w)
        if len(a) == len(self.addr) and np.array_equal(nw, self.norm):
            return a - self.addr, 0
        if self.restart is None:
            return None
        parks = _park_restarts(a, w, elf.symbols.get(RUN_KERNEL))
        if len(parks) != len(self.parks):
            return None
        rb, rp = self.restart, parks[self.loop_park]
        if len(a) - rp != len(self.addr) - rb or not np.array_equal(
            nw[rp:], self.norm[rb:]
        ):
            return None
        whole = _whole_moves(a[rp] - self.addr[rb] - pz[0])
        if whole is None:
            return None
        shift = np.zeros(len(self.addr), np.int64)
        shift[rb:] = a[rp:] - self.addr[rb:]
        return shift, whole

    def add(self, pz, elf):
        """record a probe link; False when its code differs from the restart on. A link whose restart moved by whole
        periods more keeps the code but is not used for the fit."""
        got = self.pair(pz, elf)
        if got is None:
            return False
        if not got[1]:
            self.obs[tuple(int(v) for v in pz)] = got[0]
        return True

    def fit(self):
        keys = sorted(self.obs)
        shifts = np.array([self.obs[k] for k in keys])
        pz = np.array(keys, np.int64).reshape(-1, 2)
        if (shifts[:, 0] != 0).any():
            raise ValueError("the first instruction moves")
        moved = np.flatnonzero((np.diff(shifts, axis=1) != 0).any(axis=0)) + 1
        self.gaps = []
        self.mod_gaps = set()
        for i in moved:
            s_in, s_out = shifts[:, i - 1], shifts[:, i]
            g0 = int(self.addr[i] - self.addr[i - 1] - 4)
            after_park = int(self.words[i - 1]) == EBREAK
            models = []
            # exact models first, else models right modulo the cost's period: the linker places run_kernel's callees by
            # its size before relaxation, which can move them by a whole period the cost cannot tell apart
            for modulo in (None, self.period):
                same = (
                    (lambda p, o: (p == o).all(axis=0))
                    if modulo is None
                    else (lambda p, o: ((p - o) % modulo == 0).all(axis=0))
                )
                for cp in (0, 1, -1):
                    for cz in (0, 1, -1):
                        ins = cp * pz[:, 0] + cz * pz[:, 1]
                        if same((s_in + ins)[:, None], s_out[:, None])[0]:
                            models.append((cp, cz, 0, 0, 0))
                        for A in ALIGNS:
                            f = np.arange(0, min(g0, A - 4) + 1, 4)
                            # insertion after the alignment, absorbed by it, or a size keeping fill
                            for mode in (0, 1, 2):
                                if mode == 2:
                                    pred = (
                                        s_in[:, None]
                                        + np.mod(f[None, :] - ins[:, None], A)
                                        - f[None, :]
                                    )
                                else:
                                    x = s_in[:, None] + (ins[:, None] if mode else 0)
                                    pred = (
                                        x
                                        + np.mod(f[None, :] - x, A)
                                        - f[None, :]
                                        + (0 if mode else ins[:, None])
                                    )
                                for ff in f[same(pred, s_out[:, None])]:
                                    models.append((cp, cz, A, int(ff), mode))
                if models or not modulo and not self.period:
                    break
                if modulo is None:
                    continue
            if models and modulo:
                self.mod_gaps.add(int(i))
            if not models:
                raise ValueError(f"no model for the shift change at {self.addr[i]:#x}")
            # simplest first: plain insertions, then alignments whose fill is the harness's own (a park aligns after
            # PARK_FIXED NOPs, a section alignment fills the whole gap), then the rest; models[0] is the one used
            structural = lambda m: m[3] == (g0 - PARK_FIXED if after_park else g0)
            models.sort(
                key=lambda m: (
                    m[2] != 0,
                    m[0] < 0 or m[1] < 0,
                    m[4] == 2,
                    not structural(m),
                    m[2],
                    m[4],
                )
            )
            self.gaps.append((int(i), models))

    def bounds(self):
        return np.array([self.addr[i] for i, _ in self.gaps], np.int64)

    def segment_shifts(self, P, Z):
        """[segments x candidates]: the shift of every segment between moving gaps"""
        s = np.zeros_like(P)
        out = [s]
        for _, models in self.gaps:
            s = _apply(models[0], s, P, Z)
            out.append(s)
        return np.array(out)

    def disagreement(self, P, Z):
        """a candidate (P, Z) on which the fitted models of a gap disagree, or None: the one that splits the models of
        the first such gap most evenly, so each link halves them"""
        s = np.zeros_like(P)
        for i, models in self.gaps:
            if len(models) > 1:
                same = self._agreement(models, s, P, Z, i in self.mod_gaps) / len(
                    models
                )
                split = np.flatnonzero(same < 1)
                if len(split):
                    k = split[np.argmin(np.abs(same[split] - 0.5))]
                    return int(P[k]), int(Z[k])
            s = _apply(models[0], s, P, Z)
        return None

    def _agreement(self, models, s, P, Z, modulo):
        """per candidate, how many models predict the same shift as models[0] (modulo the period for a gap fitted modulo
        it); models that differ only in their fill f are counted together: the shift is base, or base + A when f < r
        """
        red = (lambda v: v % self.period) if modulo else (lambda v: v)
        p0 = red(_apply(models[0], s, P, Z))
        agree = np.zeros(len(P), np.int64)
        groups = {}
        for cp, cz, A, f, mode in models:
            groups.setdefault((cp, cz, A, mode), []).append(f)
        for (cp, cz, A, mode), fs in groups.items():
            ins = cp * P + cz * Z
            if not A:
                agree += len(fs) * (red(s + ins) == p0)
                continue
            x = s + ins if mode == 1 else (ins if mode == 2 else s)
            r = np.mod(x, A)
            base = (s - r) if mode == 2 else (x - r + (0 if mode else ins))
            fs = np.sort(np.array(fs, np.int64))
            # fills below r take the next period
            n_lt = np.searchsorted(fs, r, side="left")
            agree += (len(fs) - n_lt) * (red(base) == p0) + n_lt * (red(base + A) == p0)
        return agree

    def in_range(self, P, Z):
        """candidates whose layout keeps every conditional branch in range: one pushed past +-4 KiB is relaxed by the
        assembler into two instructions, which changes the code"""
        seg = np.searchsorted(self.bounds(), self.addr, side="right")
        br = np.flatnonzero((self.words & 0x7F) == 0x63)
        # a branch before the restart may be relaxed: its park absorbs the change
        if self.restart is not None:
            br = br[br >= self.restart]
        off = np.array([_br_offset(int(w)) for w in self.words[br]], np.int64)
        ti = np.minimum(
            np.searchsorted(self.addr, self.addr[br] + off), len(self.addr) - 1
        )
        sb, st = seg[br], seg[ti]
        cross = sb != st
        ok = np.ones(len(P), bool)
        if cross.any():
            segs = self.segment_shifts(P, Z)
            for b, t, o in set(
                zip(sb[cross].tolist(), st[cross].tolist(), off[cross].tolist())
            ):
                new = o + segs[t] - segs[b]
                ok &= (new >= -4096) & (new <= 4094)
        return ok

    def to_json(self):
        return [[i, [list(m) for m in models]] for i, models in self.gaps]

    def from_json(self, gaps, loop_park=None, mod_gaps=()):
        self.gaps = [(int(i), [tuple(m) for m in models]) for i, models in gaps]
        self.mod_gaps = {int(i) for i in mod_gaps}
        if loop_park is not None and loop_park < len(self.parks):
            self.loop_park, self.restart = loop_park, self.parks[loop_park]


def _whole_moves(d):
    """d when it is a move by whole 512 B periods (0 included), else None"""
    return int(d) if d >= 0 and d % 512 == 0 else None


def _park_restarts(addr, words, rk):
    """indices of the first instruction after each park (its ebreak, barrier.h) in run_kernel, in address order"""
    if rk is None:
        return []
    e = np.flatnonzero((words == EBREAK) & (addr >= rk[0]) & (addr < rk[0] + rk[1]))
    return [int(i) + 1 for i in e if i + 1 < len(addr)]


def _jal_offset(w):
    return _sext(
        ((w >> 31) & 1) << 20
        | ((w >> 12) & 0xFF) << 12
        | ((w >> 20) & 1) << 11
        | ((w >> 21) & 0x3FF) << 1,
        21,
    )


def _br_offset(w):
    return _sext(
        ((w >> 31) & 1) << 12
        | ((w >> 7) & 1) << 11
        | ((w >> 25) & 0x3F) << 5
        | ((w >> 8) & 0xF) << 1,
        13,
    )


class Kernel:
    """One thread's unpadded ELF: run_kernel and the zone clock read sites."""

    def __init__(self, path):
        self.elf = Elf(path)
        self.rk0, size = self.elf.symbols[RUN_KERNEL]
        self.rk1 = self.rk0 + size
        self.words = {a: self.elf.word(a) for a in range(self.rk0, self.rk1, 4)}
        self.starts, self.ends = self._sites()

    def _sites(self):
        """clock reads after a zone_reserve call (zone starts) and before a zone_record call (zone ends)"""
        reserve = self.elf.symbols.get(ZONE_RESERVE, (None,))[0]
        record = self.elf.symbols.get(ZONE_RECORD, (None,))[0]

        def calls(a, offsets, target):
            for k in offsets:
                w = self.words.get(a + 4 * k)
                if (
                    w is not None
                    and w & 0x7F == 0x6F
                    and (a + 4 * k + _jal_offset(w)) & M32 == target
                ):
                    return True
            return False

        reads = []
        for a, w in self.words.items():
            nxt = self.words.get(a + 4)
            if (
                w is None
                or nxt is None
                or w & 0x707F != 0x2003
                or nxt & 0x707F != 0x2003
            ):
                continue
            if (
                _sext(w >> 20, 12) == 496
                and _sext(nxt >> 20, 12) == 504
                and (w >> 15) & 31 == (nxt >> 15) & 31
            ):
                reads.append(a)
        return (
            {a for a in reads if calls(a, range(-4, 0), reserve)},
            {a for a in reads if calls(a, range(1, 14), record)},
        )


def trace(
    kernel, runtime_bytes, restarts, min_window=MIN_WINDOW, max_window=MAX_WINDOW
):
    """The measured window from the TILE_LOOP restart (restarts: its possible addresses): (records, restart pc, restart
    index, lo, hi, complete) or None; records = control instructions (pc, taken, target, instruction index).
    """
    elf = kernel.elf
    l1 = bytearray(0x200000)
    ldm = bytearray(0x10000)
    other = {}
    for a, d in elf.sections.values():
        if a < 0x200000:
            l1[a : a + len(d)] = d
        elif 0xFFB00000 <= a < 0xFFB10000:
            ldm[a - 0xFFB00000 : a - 0xFFB00000 + len(d)] = d
    for i, b in enumerate(runtime_bytes):
        other[PARAMS + i] = b

    def ld(a, n):
        if a < 0x200000:
            return int.from_bytes(l1[a : a + n], "little")
        if 0xFFB00000 <= a < 0xFFB10000:
            o = a - 0xFFB00000
            return int.from_bytes(ldm[o : o + n], "little")
        if 0x70000000 <= a < 0x80000000:
            return sum(other.get(a + i, 0) << (8 * i) for i in range(n))
        return 0  # memory mapped register

    stores = 0
    x = [0] * 32
    x[2], x[3], x[10], x[1] = (
        STACK,
        elf.symbols.get("__global_pointer$", (0, 0))[0],
        PARAMS,
        RETURN,
    )
    words = dict(kernel.words)
    rec, seen = [], {}
    pc, icount = kernel.rk0, 0
    warm = lo = start = None
    pcs_seen, last_new = set(), 0
    complete = False
    for _ in range(TRACE_BUDGET):
        if pc == RETURN:
            break
        if warm is None and pc in restarts:
            warm, start = icount, pc
        if lo is not None:
            if pc not in pcs_seen:
                pcs_seen.add(pc)
                last_new = icount
            n = icount - lo
            if n >= max_window or (
                n >= min_window and icount - last_new >= max(min_window, n // 2)
            ):
                break
        w = words.get(pc)
        if w is None:
            w = words[pc] = elf.word(pc)
            if w is None:
                break
        nxt, op = pc + 4, w & 0x7F
        rd, f3, rs1, rs2 = (w >> 7) & 31, (w >> 12) & 7, (w >> 15) & 31, (w >> 20) & 31
        a, b = x[rs1], x[rs2]
        val = None
        if w & 3 != 3:
            pass  # Tensix instruction
        elif op == 0x37:
            val = w & 0xFFFFF000
        elif op == 0x17:
            val = pc + (w & 0xFFFFF000)
        elif op == 0x6F:
            val, nxt = pc + 4, (pc + _jal_offset(w)) & M32
            if warm is not None:
                rec.append((pc, 1, nxt, icount))
        elif op == 0x67:
            val, nxt = pc + 4, (a + _sext(w >> 20, 12)) & ~1 & M32
            if warm is not None:
                rec.append((pc, 1, nxt, icount))
        elif op == 0x63:
            t = (pc + _br_offset(w)) & M32
            if f3 == 0:
                c = a == b
            elif f3 == 1:
                c = a != b
            elif f3 == 4:
                c = _s32(a) < _s32(b)
            elif f3 == 5:
                c = _s32(a) >= _s32(b)
            elif f3 == 6:
                c = a < b
            elif f3 == 7:
                c = a >= b
            else:
                c = False
            # a spin wait: same registers and no RAM store since the last visit
            if c and t <= pc and pc - t <= SPIN_SPAN:
                key = (pc, tuple(x))
                c = seen.get(key) != stores
                seen[key] = stores
            if warm is not None:
                rec.append((pc, int(c), t, icount))
            nxt = t if c else pc + 4
        elif op == 0x03:
            ea = (a + _sext(w >> 20, 12)) & M32
            if ea == CLOCK_LO and warm is not None:
                if lo is None and (pc in kernel.starts or not kernel.starts):
                    lo = icount
                elif lo is not None and (pc in kernel.ends or not kernel.ends):
                    complete = True
                    break
            n = (1, 2, 4, 4, 1, 2, 4, 4)[f3]
            v = ld(ea, n)
            val = _sext(v, 8 * n) & M32 if f3 in (0, 1) else v
        elif op == 0x23:
            ea = (a + _sext(((w >> 25) << 5) | ((w >> 7) & 31), 12)) & M32
            n = (1, 2, 4, 4, 4, 4, 4, 4)[f3]
            if ea < 0x200000:
                l1[ea : ea + n] = (b & ((1 << (8 * n)) - 1)).to_bytes(n, "little")
                stores += 1
            elif 0xFFB00000 <= ea < 0xFFB10000:
                o = ea - 0xFFB00000
                ldm[o : o + n] = (b & ((1 << (8 * n)) - 1)).to_bytes(n, "little")
                stores += 1
            elif 0x70000000 <= ea < 0x80000000:
                for i in range(n):
                    other[ea + i] = (b >> (8 * i)) & 0xFF
                stores += 1
        elif op == 0x13:
            i = _sext(w >> 20, 12)
            if f3 == 0:
                val = a + i
            elif f3 == 1:
                val = a << ((w >> 20) & 31)
            elif f3 == 2:
                val = int(_s32(a) < i)
            elif f3 == 3:
                val = int(a < (i & M32))
            elif f3 == 4:
                val = a ^ (i & M32)
            elif f3 == 5:
                sh = (w >> 20) & 31
                val = (_s32(a) >> sh) if (w >> 30) & 1 else (a >> sh)
            elif f3 == 6:
                val = a | (i & M32)
            else:
                val = a & (i & M32)
        elif op == 0x33:
            f7 = w >> 25
            if f7 == 1:
                sa, sb = _s32(a), _s32(b)
                val = {
                    0: lambda: a * b,
                    1: lambda: (sa * sb) >> 32,
                    2: lambda: (sa * b) >> 32,
                    3: lambda: (a * b) >> 32,
                    4: lambda: (int(sa / sb) if sb else -1),
                    5: lambda: (a // b if b else M32),
                    6: lambda: (
                        (abs(sa) % abs(sb)) * (1 if sa >= 0 else -1) if sb else sa
                    ),
                    7: lambda: (a % b if b else a),
                }[f3]()
            else:
                val = {
                    0: lambda: (a - b) if f7 == 0x20 else (a + b),
                    1: lambda: a << (b & 31),
                    2: lambda: int(_s32(a) < _s32(b)),
                    3: lambda: int(a < b),
                    4: lambda: a ^ b,
                    5: lambda: (_s32(a) >> (b & 31)) if f7 == 0x20 else (a >> (b & 31)),
                    6: lambda: a | b,
                    7: lambda: a & b,
                }[f3]()
        elif op == 0x73 and f3:
            val = 0  # CSR reads
        if val is not None and rd:
            x[rd] = val & M32
        pc = nxt
        icount += 1
    if lo is None:
        return None
    return rec, start, warm, lo, icount, complete


def _bp_index(a):
    b = lambda i: (a >> i) & 1
    return (
        ((1 - (b(8) ^ b(6))) << 3)
        | ((b(7) ^ b(5)) << 2)
        | ((1 - (b(4) ^ b(3))) << 1)
        | b(2)
    )


def _regs_read(w):
    if w is None or w & 3 != 3:
        return ()
    op = w & 0x7F
    rs1, rs2 = (w >> 15) & 31, (w >> 20) & 31
    if op in (0x33, 0x23, 0x63):
        return (rs1, rs2)
    if op in (0x13, 0x03, 0x67, 0x73):
        return (rs1,)
    return ()


class Window:
    """The modelled window of one kernel thread and runtime configuration, to be costed for any pads."""

    def __init__(self, kernel, amap, records, start, warm, lo):
        self.amap, self.records = amap, records
        pcs, ctl, cur = [], [], start
        for pc, taken, target, _ in records:
            pcs.extend(range(cur, pc + 4, 4) if pc >= cur else [pc])
            ctl.append(len(pcs) - 1)
            cur = target if taken else pc + 4
        self.pcs = np.array(pcs, np.int64)
        self.ctl = ctl
        self.counted = np.arange(len(self.pcs)) + warm >= lo
        self.n = max(int(self.counted.sum()), 1)
        self.bounds = amap.bounds()
        idx = np.searchsorted(amap.addr, self.pcs, side="right") - 1
        # the non-fill instruction each executed one moves with
        self.idx = np.maximum(idx, 0)
        self.seg = np.searchsorted(self.bounds, amap.addr[self.idx], side="right")
        self.seg[idx < 0] = 0
        word = kernel.elf.word
        self.words = {int(a): word(int(a)) for a in np.unique(self.pcs)}
        # false hazard sites: (stream index, constant hazard or None, branch target segment, branch target, regs read)
        self.haz = []
        for i in np.flatnonzero(self.counted[:-2]):
            w = self.words.get(int(self.pcs[i]))
            if w is None or w & 3 != 3 or (w & 0x7F) not in (0x63, 0x23):
                continue
            reads = _regs_read(self.words.get(int(self.pcs[i + 2])))
            if not reads:
                continue
            if (w & 0x7F) == 0x23:
                f = (w >> 7) & 31
                if f != 0 and f in reads:
                    self.haz.append((int(i), True, None, None, reads))
            else:
                t = (int(self.pcs[i]) + _br_offset(w)) & M32
                ti = max(int(np.searchsorted(amap.addr, t, side="right")) - 1, 0)
                tseg = int(np.searchsorted(self.bounds, amap.addr[ti], side="right"))
                if tseg == self.seg[i]:
                    offs = t - int(self.pcs[i])
                    f = ((offs >> 1) & 0xF) << 1 | ((offs >> 11) & 1)
                    if f != 0 and f in reads:
                        self.haz.append((int(i), True, None, None, reads))
                else:
                    self.haz.append((int(i), None, (tseg, ti), t, reads))

    def cost(self, P, Z, nsets, actual=None):
        """(cost per window instruction, mispredicts, misses, hazards) per candidate (P[i], Z[i]); with actual (the
        shift of every non-fill instruction in a real link) for that one layout"""
        P = np.asarray(P, np.int64)
        Z = np.asarray(Z, np.int64)
        n = len(P)
        rows = np.arange(n)
        if actual is None:
            # [segments x candidates]
            segs = self.amap.segment_shifts(P, Z).astype(np.int32)
            shift = segs[self.seg]
            target_shift = lambda tseg, ti: segs[tseg].astype(np.int64)
        else:
            actual = np.asarray(actual, np.int64)
            shift = actual[self.idx][:, None].astype(np.int32)
            target_shift = lambda tseg, ti: np.full(n, int(actual[ti]), np.int64)
        counted = self.counted
        lines = (self.pcs[:, None].astype(np.int32) + shift) >> 4

        # instruction cache: 2 ways, first in first out, cold at the restart (BRISC invalidates it)
        change = np.ones(len(self.pcs), bool)
        change[1:] = (lines[1:] != lines[:-1]).any(axis=1)
        ways = np.full((n, nsets, 2), -1, np.int32)
        fill = np.zeros((n, nsets), np.int64)
        last = np.full(n, -1, np.int32)
        misses = np.zeros(n, np.int64)
        for k in np.flatnonzero(change):
            ln = lines[k]
            new = ln != last
            if not new.all():
                if not new.any():
                    continue
                r, l = rows[new], ln[new]
            else:
                r, l = rows, ln
            last[r] = l
            st = l % nsets
            hit = (ways[r, st, 0] == l) | (ways[r, st, 1] == l)
            miss = ~hit
            if not miss.any():
                continue
            r, l, st = r[miss], l[miss], st[miss]
            f = fill[r, st]
            ways[r, st, f & 1] = l
            fill[r, st] = f + 1
            if counted[k]:
                misses[r] += 1

        # branch predictor: next PC entries; the first execution of a control instruction is not predicted
        cnt = np.zeros((n, 16), np.int64)
        tgt = np.zeros((n, 16), np.int64)
        known = set()
        mispred = np.zeros(n, np.int64)
        pend = []
        mis_at = {}
        idx_cache = {}
        for r_i, k in enumerate(self.ctl):
            pc, taken, target, _ = self.records[r_i]
            nxt = target if taken else pc + 4
            while pend and pend[0][0] <= k:
                _, idx, c, t = pend.pop(0)
                cnt[rows, idx], tgt[rows, idx] = c, t
            idx = idx_cache.get(pc)
            if idx is None:
                idx = idx_cache[pc] = _bp_index(pc + shift[k].astype(np.int64))
            c, e = cnt[rows, idx], tgt[rows, idx]
            if pc in known:
                m = (e != nxt) | (((c >= 0) & (e != pc)) != bool(taken))
            else:
                m = np.full(n, bool(taken))
                known.add(pc)
            if counted[k]:
                mispred += m
            mis_at[k] = m
            right = e == nxt
            if taken:
                nc = np.where(~right, 1, np.minimum(c + 1, 1))
            else:
                nc = np.where(~right, -2, np.maximum(c - 1, -2))
            pend.append((k + K_PRED, idx, nc, nxt))

        # false hazard: instruction i+2 reads x[instr_i[11:7]] after a branch or store, unless a mispredict intervened
        haz = np.zeros(n, np.int64)
        for i, const, tseg, t, reads in self.haz:
            if const:
                h = np.ones(n, bool)
            else:
                offs = (t + target_shift(*tseg)) - (
                    int(self.pcs[i]) + shift[i].astype(np.int64)
                )
                f = ((offs >> 1) & 0xF) << 1 | ((offs >> 11) & 1)
                h = (f != 0) & np.isin(f, reads)
            if i in mis_at:
                h = h & ~mis_at[i]
            if i + 1 in mis_at:
                h = h & ~mis_at[i + 1]
            haz += h
        total = MIS_COST * mispred + MISS_COST * misses + HAZ_COST * haz
        return total / self.n, mispred, misses, haz


def flow_key(records, lo):
    """Identifies the modelled control flow, so runtime configurations that take the same path share a choice"""
    return hashlib.sha1(
        repr([(pc, t, tg, ic - lo) for pc, t, tg, ic in records]).encode()
    ).hexdigest()[:16]


@functools.lru_cache(maxsize=16)
def _kernel(elf_path):
    return Kernel(elf_path)


class _Lock:
    def __init__(self, path):
        self.path = Path(path).with_suffix(".lock")

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


class Maps:
    """The address map of one thread ELF, fitted once per variant and kept as json in the cache dir."""

    def __init__(self, base_elf, thread, relink, cache_dir):
        self.base_elf, self.thread, self.relink = str(base_elf), thread, relink
        self.file = Path(cache_dir) / f"map-{thread}.json"
        self.zspan = Z_SPAN.get(thread, Z_SPAN_DEFAULT)
        # of the cost (predictor hash, instruction cache sets)
        self.period = 512 if thread == "math" else 1024
        self.P = np.arange(0, P_SPAN.get(thread, P_SPAN_DEFAULT), PAD_STEP)
        self.Z = np.arange(0, self.zspan, PAD_STEP)
        pmax, zmax = int(self.P[-1]), int(self.Z[-1])
        self.probes = [(PAD_STEP, 0), (0, PAD_STEP), (pmax, 0), (0, zmax), (pmax, zmax)]
        self.state = None
        self.amap = None
        self.links = 0
        self._grid = None

    def _link(self, pz, d):
        out = Path(d) / f"{pz[0]}_{pz[1]}"
        out.mkdir(parents=True, exist_ok=True)
        self.links += 1
        try:
            self.relink(int(pz[0]), int(pz[1]), out)
            return Elf(out / f"{self.thread}.elf")
        except Exception:
            return None

    def _read(self):
        try:
            s = json.loads(self.file.read_text())
            return s if s.get("version") == MAP_VERSION else None
        except (OSError, ValueError):
            return None

    def get(self):
        """the fitted map (probe links once per variant thread, then from the json)"""
        if self.amap is not None:
            return self.amap
        base = Elf(self.base_elf)
        amap = AddressMap(base, self.period)
        s = self._read()
        if s is None:
            with _Lock(self.file):
                s = self._read()
                if s is None:
                    s = self._fit(base, amap)
                    write_json(self.file, s)
        self.state = s
        amap.from_json(s["gaps"], s.get("loop_park"), s.get("mod_gaps", ()))
        self.amap = amap
        return amap

    def _fit(self, base, amap):
        state = {"version": MAP_VERSION, "limit": None, "bounds": None, "verified": {}}
        with tempfile.TemporaryDirectory(prefix="llk_layout_") as d:
            linked = {}

            def probe(pz):
                e = self._link(pz, d)
                linked[pz] = e is not None
                return e is not None and amap.add(pz, e)

            first = {pz: self._link(pz, d) for pz in self.probes}
            amap.find_loop_park(first)
            for pz, e in first.items():
                linked[pz] = e is not None
            ok = {pz: e is not None and amap.add(pz, e) for pz, e in first.items()}
            # the largest pads overflow the code region: keep to what fits
            if not linked[self.probes[-1]]:
                state["limit"] = max(
                    (base.region_end or base.text_end) - base.text_end - 2048, 0
                )
            pmax, zmax = int(self.P[-1]), int(self.Z[-1])
            if not all(ok.values()):
                # pads that push a branch past +-4 KiB at assembly time (before the linker shrinks the code, so the
                # final ELF cannot tell) change the code: keep below the first that does, found to BOUND_STEP bytes
                def largest(point, hi):
                    lo = 0
                    while hi - lo > BOUND_STEP:
                        mid = (lo + hi) // 2 // PAD_STEP * PAD_STEP
                        if probe(point(mid)):
                            lo = mid
                        else:
                            hi = mid
                    return lo

                plim = pmax if ok[(pmax, 0)] else largest(lambda v: (v, 0), pmax)
                zlim = zmax if ok[(0, zmax)] else largest(lambda v: (0, v), zmax)
                slim = plim + zlim
                if (plim, zlim) != (pmax, zmax) or not ok[(pmax, zmax)]:
                    if not probe((plim, zlim)):
                        slim = largest(
                            lambda v: (min(v, plim), v - min(v, plim)), plim + zlim
                        )
                state["bounds"] = [plim, zlim, slim]
            amap.fit()
            self.state, self._grid = state, None
            tried = set()
            for _ in range(16):  # links that decide between the models a gap still has
                PP, ZZ = self.grid(amap)
                pz = amap.disagreement(PP, ZZ)
                if pz is None or pz in tried or not probe(pz):
                    break
                tried.add(pz)
                amap.fit()
                self._grid = None
        state["gaps"] = amap.to_json()
        state["loop_park"] = amap.loop_park
        state["mod_gaps"] = sorted(amap.mod_gaps)
        state["probes"] = [list(k) for k in sorted(amap.obs)]
        state["links"] = self.links
        return state

    def grid(self, amap=None):
        """the candidate pads: on the PAD_STEP grid, within the code region, no branch pushed out of range"""
        if getattr(self, "_grid", None) is None:
            amap = amap or self.get()
            PP, ZZ = np.meshgrid(self.P, self.Z, indexing="ij")
            PP, ZZ = PP.ravel(), ZZ.ravel()
            ok = amap.in_range(PP, ZZ)
            if self.state and self.state.get("limit") is not None:
                ok &= PP + ZZ <= self.state["limit"]
            if self.state and self.state.get("bounds") is not None:
                plim, zlim, slim = self.state["bounds"]
                ok &= (PP <= plim) & (ZZ <= zlim) & (PP + ZZ <= slim)
            self._grid = (PP[ok], ZZ[ok])
        return self._grid

    def verify(self, pz):
        """link the pads and compare every instruction's address with the map: (True, None) when it holds, else (False,
        the real shift of every non-fill instruction, or None); the outcome is kept in the json
        """
        key = f"{pz[0]},{pz[1]}"
        s = self._read() or self.state
        got = s.get("verified", {}).get(key)
        if got is True:
            return True, None
        amap = self.get()
        if got is not None and got is not False:
            return False, _expand(got, len(amap.addr))
        ok, actual = False, None
        with tempfile.TemporaryDirectory(prefix="llk_layout_") as d:
            e = self._link(pz, d)
            if e is not None:
                got = amap.pair(pz, e)
                if got is not None:
                    actual = got[0]
                    segs = amap.segment_shifts(np.array([pz[0]]), np.array([pz[1]]))[
                        :, 0
                    ]
                    seg = np.searchsorted(amap.bounds(), amap.addr, side="right")
                    # equal modulo the period the cost sees (exactly, unless the restart moved by whole periods more or
                    # a gap was fitted modulo the period)
                    ok = bool(((actual - segs[seg]) % self.period == 0).all())
        with _Lock(self.file):
            s = self._read() or self.state
            s.setdefault("verified", {})[key] = (
                True if ok else (_runs(actual) if actual is not None else False)
            )
            write_json(self.file, s)
        return ok, (None if ok else actual)


def _runs(shift):
    """[[first index, shift], ...] of a per instruction shift array"""
    starts = np.flatnonzero(np.diff(shift, prepend=shift[0] - 1))
    return [[int(i), int(shift[i])] for i in starts]


def _expand(runs, n):
    out = np.zeros(n, np.int64)
    for k, (i, v) in enumerate(runs):
        out[i : runs[k + 1][0] if k + 1 < len(runs) else n] = v
    return out


def code_key(assembly):
    """Identifies a thread's code: the hash of its assembly without the debug sections (they name the variant's build
    directory), so variants that compile to the same code share their layout work (choose's shared cache)
    """
    end = assembly.find("\t.section\t.debug_info")
    return hashlib.sha256(
        assembly[: end if end >= 0 else len(assembly)].encode()
    ).hexdigest()[:24]


def choose(elf_path, thread, runtime_bytes, cache_dir, relink, log=None, shared=None):
    """(P, Z) in bytes for one thread's unpadded ELF and runtime configuration, (0, 0) when it has no measured loop;
    relink(P, Z, out_dir) links the thread with pads, shared = (directory, code_key) shares the work between variants.
    """
    if shared:
        cache_dir = Path(shared[0]) / f"{thread}-{shared[1]}"
    return tuple(_choose(elf_path, thread, runtime_bytes, cache_dir, relink, log))


def _choose(elf_path, thread, runtime_bytes, cache_dir, relink, log):
    t_start = time.perf_counter()
    kernel = _kernel(str(elf_path))
    if not kernel.starts:
        return 0, 0
    maps = Maps(elf_path, thread, relink, cache_dir)
    amap = maps.get()
    restarts = {
        int(amap.addr[i])
        for i, models in amap.gaps
        if models[0][:2] == (1, 0) and models[0][4] != 2
    }
    if not restarts:
        return 0, 0
    t_trace = time.perf_counter()
    tr = trace(kernel, runtime_bytes, restarts)
    t_trace = time.perf_counter() - t_trace
    if tr is None:
        return 0, 0
    records, start, warm, lo, hi, complete = tr
    if hi - lo < MIN_WINDOW_INSTR or not records:
        return 0, 0
    cached = Path(cache_dir) / f"{thread}-{flow_key(records, lo)}.json"
    try:
        return tuple(json.loads(cached.read_text())["pads"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    win = Window(kernel, amap, records, start, warm, lo)
    nsets = 16 if thread == "math" else 64
    PP, ZZ = maps.grid()
    # the cost depends on the pads only through the shifts of the segments the window runs in, modulo the predictor
    # and cache periods: candidates with the same shifts are evaluated once
    period = 512 if thread == "math" else 1024
    touched = np.unique(win.seg)
    shifts = amap.segment_shifts(PP, ZZ)[touched] % period
    _, cls = np.unique(shifts, axis=1, return_inverse=True)
    cls = cls.ravel()
    index = {(int(p), int(z)): i for i, (p, z) in enumerate(zip(PP, ZZ))}
    by_cls = {}
    # the smallest pads of each class (Z first) stand for it
    for i in np.lexsort((PP, ZZ)):
        by_cls.setdefault(int(cls[i]), int(i))
    costs = np.full(int(cls.max()) + 1, np.nan)

    def evaluate(cands):
        ids = sorted(
            {int(cls[index[c]]) for c in cands if c in index}
            - set(np.flatnonzero(~np.isnan(costs)).tolist())
        )
        for k in range(0, len(ids), EVAL_CHUNK):
            part = ids[k : k + EVAL_CHUNK]
            rep = [by_cls[c] for c in part]
            costs[part] = win.cost(PP[rep], ZZ[rep], nsets)[0]

    def ranked():
        """the evaluated classes' smallest pads, by cost, ties by the smaller pads (Z first)"""
        reps = [by_cls[c] for c in np.flatnonzero(~np.isnan(costs))]
        reps.sort(key=lambda i: (costs[cls[i]], ZZ[i], PP[i]))
        return [(int(PP[i]), int(ZZ[i])) for i in reps]

    if costs.size * win.n <= EXHAUSTIVE:
        evaluate(list(index))
    else:  # P, then Z from the best few P, then P from the best few Z, then Z again
        evaluate([(p, 0) for p in maps.P.tolist()] + [(0, 0)])
        tops = list(dict.fromkeys(p for p, _ in ranked()))[:STARTS]
        evaluate([(p, z) for p in tops for z in maps.Z.tolist()])
        topz = list(dict.fromkeys(z for _, z in ranked()))[:STARTS]
        evaluate([(p, z) for z in topz for p in maps.P.tolist()])
        p1, _ = ranked()[0]
        evaluate([(p1, z) for z in maps.Z.tolist()])
    # the chosen pads are checked on their real link; where the fitted map is wrong, their cost is taken from the real
    # addresses and the search goes on down the ranking
    exact = {(0, 0): float(costs[cls[index[(0, 0)]]])} if (0, 0) in index else {}
    pads = None
    for cand in ranked()[:MAX_VERIFY]:
        if cand in exact:
            pads = cand
            break
        ok, actual = maps.verify(cand)
        if ok:
            exact[cand] = float(costs[cls[index[cand]]])
            pads = cand
            break
        if actual is not None:
            exact[cand] = float(
                win.cost([cand[0]], [cand[1]], nsets, actual=actual)[0][0]
            )
    if pads is None or (exact and exact.get(pads, np.inf) > min(exact.values())):
        pads = min(exact, key=lambda c: (exact[c], c[1], c[0])) if exact else (0, 0)
    cost_of = lambda pz: exact.get(
        pz, float(costs[cls[index[pz]]]) if pz in index else None
    )
    result = {
        "pads": list(pads),
        "cost": cost_of(pads),
        "cost0": cost_of((0, 0)),
        "window": [lo, hi, int(complete)],
        "classes": int(costs.size),
        "evaluated": int((~np.isnan(costs)).sum()),
    }
    timing = {
        "t": round(time.perf_counter() - t_start, 4),
        "t_trace": round(t_trace, 4),
    }
    write_json(cached, result)
    if log:
        try:
            with open(log, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "elf": str(elf_path),
                            "thread": thread,
                            "bytes": bytes(runtime_bytes).hex(),
                            **result,
                            **timing,
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
    return tuple(pads)


def write_json(path, value):
    """Atomically, as several pytest workers can choose for the same variant at once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)
