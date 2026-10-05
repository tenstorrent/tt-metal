# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Code layout pads for the measured loop of Wormhole perf kernels (LLK_DBG_BARRIER builds, see barrier.h).

A TRISC loop's speed depends on its address: the branch predictor (16 untagged entries, indexed by a hash of PC bits
2..8) and the 2-way instruction cache (16 sets of 16 B on the math thread, 64 on the others) both see the code
address. The TILE_LOOP restart point (pad P) and the code after the TILE_LOOP zone end (pad Z) can take NOPs that
never run in the measured window. This module traces the measured window of one kernel thread for one runtime
configuration with a small RV32IM interpreter, models both structures for every (P, Z) and returns the pads with the
fewest modelled mispredicts and misses. The choice depends only on the code in the window and the runtime arguments.
"""
import functools
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

PAD_STEP = 4
PADS = np.arange(0, 512, PAD_STEP)
WINDOW = 4000  # instructions of the measured window that are modelled
BP_LAG = 8  # instructions between a branch and its predictor update
MISPREDICT_COST, MISS_COST = 3.0, 5.0
CODE_HEADROOM = 2048  # parks and 512 B aligned sections after the pads can each grow the code by up to 511 B
CLOCK_LO = 0xFFB121F0
EBREAK = 0x00100073
NOP = 0x00000013
ZONE_RESERVE = "_ZN12llk_profiler12zone_reserveEv"
ZONE_RECORD = "_ZN12llk_profiler11zone_recordEtyy"
RUN_KERNEL = "_Z10run_kernelRK13RuntimeParams"
PARAMS, STACK, RETURN = 0x70000000, 0x7FFF0000, 0x7EEE0000
M32 = 0xFFFFFFFF


def _sext(x, bits):
    return x - (1 << bits) if x & (1 << (bits - 1)) else x


def _s32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


class Elf:
    """Loaded sections, symbols and the end of the code region of a little endian ELF32."""

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
        self.sections, self.symbols, self.region_end = {}, {}, None
        for h in heads:
            n = name(h[0], heads[shstrndx])
            if n == ".loader_init":  # placed right after the TRISC code region
                self.region_end = h[3]
            if h[2] & 2 and h[1] != 8:  # SHF_ALLOC, not SHT_NOBITS
                self.sections[n] = (h[3], data[h[4] : h[4] + h[5]])
            if h[1] == 2:  # SHT_SYMTAB
                for off in range(h[4], h[4] + h[5], 16):
                    st_name, st_value, st_size = struct.unpack_from("<III", data, off)
                    self.symbols[name(st_name, heads[h[6]])] = (st_value, st_size)
        self.text_end = max(
            a + len(d) for n, (a, d) in self.sections.items() if n.startswith(".text")
        )

    def word(self, addr):
        for a, d in self.sections.values():
            if a <= addr < a + len(d) - 3:
                return struct.unpack_from("<I", d, addr - a)[0]
        return None


class Kernel:
    """One thread's ELF: run_kernel, the measured zone's clock read sites and the parks it re-aligns at."""

    def __init__(self, path):
        self.elf = Elf(path)
        self.rk0, size = self.elf.symbols[RUN_KERNEL]
        self.rk1 = self.rk0 + size
        self.words = {a: self.elf.word(a) for a in range(self.rk0, self.rk1, 4)}
        self.sites = self._sites()
        self.parks = [a for a, w in sorted(self.words.items()) if w == EBREAK]

    def _sites(self):
        """(start, end): the clock read after zone_reserve and the one before zone_record, or None"""
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
        start = [a for a in reads if calls(a, range(-4, 0), reserve)]
        end = [a for a in reads if calls(a, range(1, 14), record)]
        return (start[0], end[0]) if len(start) == 1 and len(end) == 1 else None

    def events(self):
        """[boundary, kind, fill start, takes P]: kind 1 = park restart point (512 B aligned), 2 = zone end pad"""
        ev = [
            [(a + 68 + 511) & ~511, 1, a + 68, 0] for a in self.parks
        ]  # ebreak, 16 words, then .balign 512
        before = [e for e in ev if e[0] <= self.sites[0]]
        if before:
            max(before, key=lambda e: e[0])[3] = 1
        ev.append([self.sites[1] + 8, 2, 0, 0])
        return sorted(ev)


def _jal_offset(w):
    return _sext(
        ((w >> 31) & 1) << 20
        | ((w >> 12) & 0xFF) << 12
        | ((w >> 20) & 1) << 11
        | ((w >> 21) & 0x3FF) << 1,
        21,
    )


def trace(kernel, runtime_bytes, window=WINDOW):
    """Branch records (pc, taken, target, instruction index) of the first `window` instructions of the measured zone.

    Tensix and SFPU instructions do nothing, memory mapped registers read 0, and a backward branch taken again with
    an unchanged register file (a spin wait) falls through, as when the waited-for unit is ready.
    """
    elf = kernel.elf
    mem = {}
    for a, d in elf.sections.values():
        for i, b in enumerate(d):
            mem[a + i] = b
    for i, b in enumerate(runtime_bytes):
        mem[PARAMS + i] = b
    x = [0] * 32
    x[2], x[3], x[10], x[1] = (
        STACK,
        elf.symbols.get("__global_pointer$", (0, 0))[0],
        PARAMS,
        RETURN,
    )
    ld = lambda a, n: sum(mem.get((a + i) & M32, 0) << (8 * i) for i in range(n))

    def st(a, v, n):
        if a < 0x200000 or 0xFFB00000 <= a < 0xFFB10000 or 0x70000000 <= a < 0x80000000:
            for i in range(n):
                mem[(a + i) & M32] = (v >> (8 * i)) & 0xFF

    pc, icount, start, end, rec, seen, words = (
        kernel.rk0,
        0,
        None,
        None,
        [],
        set(),
        dict(kernel.words),
    )
    for _ in range(1_000_000):
        if pc == RETURN or (start is not None and icount >= start + window):
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
            rec.append((pc, 1, nxt, icount))
        elif op == 0x67:
            val, nxt = pc + 4, (a + _sext(w >> 20, 12)) & ~1 & M32
            rec.append((pc, 1, nxt, icount))
        elif op == 0x63:
            t = (
                pc
                + _sext(
                    ((w >> 31) & 1) << 12
                    | ((w >> 7) & 1) << 11
                    | ((w >> 25) & 0x3F) << 5
                    | ((w >> 8) & 0xF) << 1,
                    13,
                )
            ) & M32
            c = {
                0: a == b,
                1: a != b,
                4: _s32(a) < _s32(b),
                5: _s32(a) >= _s32(b),
                6: a < b,
                7: a >= b,
            }.get(f3, False)
            if c and t <= pc:
                key = (pc, tuple(x))
                c = key not in seen
                seen.add(key)
            rec.append((pc, int(c), t, icount))
            nxt = t if c else pc + 4
        elif op == 0x03:
            ea = (a + _sext(w >> 20, 12)) & M32
            if ea == CLOCK_LO and pc == kernel.sites[0] and start is None:
                start = icount
            elif ea == CLOCK_LO and pc == kernel.sites[1] and start is not None:
                end = icount
                break
            n = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}.get(f3, 4)
            mmio = not (
                ea < 0x200000
                or 0xFFB00000 <= ea < 0xFFB10000
                or 0x70000000 <= ea < 0x80000000
            )
            v = 0 if mmio else ld(ea, n)
            val = _sext(v, 8 * n) & M32 if f3 in (0, 1) else v
        elif op == 0x23:
            st(
                (a + _sext(((w >> 25) << 5) | ((w >> 7) & 31), 12)) & M32,
                b,
                {0: 1, 1: 2, 2: 4}.get(f3, 4),
            )
        elif op == 0x13:
            i = _sext(w >> 20, 12)
            sh = (w >> 20) & 31
            val = {
                0: a + i,
                2: int(_s32(a) < i),
                3: int(a < (i & M32)),
                4: a ^ (i & M32),
                6: a | (i & M32),
                7: a & (i & M32),
                1: a << sh,
                5: (_s32(a) >> sh) if (w >> 30) & 1 else (a >> sh),
            }[f3]
        elif op == 0x33:
            f7 = w >> 25
            if f7 == 1:
                sa, sb = _s32(a), _s32(b)
                val = {
                    0: a * b,
                    1: (sa * sb) >> 32,
                    2: (sa * b) >> 32,
                    3: (a * b) >> 32,
                    4: (int(sa / sb) if sb else -1),
                    5: (a // b if b else M32),
                    6: ((abs(sa) % abs(sb)) * (1 if sa >= 0 else -1) if sb else sa),
                    7: (a % b if b else a),
                }[f3]
            else:
                val = {
                    0: (a - b) if f7 == 0x20 else (a + b),
                    1: a << (b & 31),
                    2: int(_s32(a) < _s32(b)),
                    3: int(a < b),
                    4: a ^ b,
                    5: (_s32(a) >> (b & 31)) if f7 == 0x20 else (a >> (b & 31)),
                    6: a | b,
                    7: a & b,
                }[f3]
        elif op == 0x73 and f3:
            val = 0  # CSR reads
        if val is not None and rd:
            x[rd] = val & M32
        pc = nxt
        icount += 1
    if start is None:
        return None
    return (
        [r for r in rec if r[3] >= start],
        start,
        end if end is not None else min(icount, start + window),
    )


def _offsets(events, p, z):
    """the address shift after each event, per candidate pad pair"""
    off = np.zeros((len(p), len(events) + 1), np.int64)
    for e, (bound, kind, fill, takes_p) in enumerate(events):
        if kind == 1:
            off[:, e + 1] = (
                ((fill + off[:, e] + 511) // 512) * 512 + (p if takes_p else 0) - bound
            )
        else:
            off[:, e + 1] = off[:, e] + z
    return off


def _bp_index(a):
    b = lambda i: (a >> i) & 1
    return (
        ((1 - (b(8) ^ b(6))) << 3)
        | ((b(7) ^ b(5)) << 2)
        | ((1 - (b(4) ^ b(3))) << 1)
        | b(2)
    )


def cost(kernel, records, lo, hi, nsets, p, z):
    """Modelled mispredicts * MISPREDICT_COST + misses * MISS_COST per instruction, per candidate pad pair (p[i], z[i])"""
    ev = kernel.events()
    bounds = np.array([e[0] for e in ev], np.int64)
    off = _offsets(ev, np.asarray(p, np.int64), np.asarray(z, np.int64))
    n = off.shape[0]
    rows = np.arange(n)

    def shifted(a):
        seg = (
            len(ev)
            if a >= kernel.rk1
            else 0 if a < kernel.rk0 else int(np.searchsorted(bounds, a, side="right"))
        )
        return a + off[:, seg]

    # instruction cache: 2 ways, first in first out, cold at the window start. A straight run of instructions within
    # one shift segment reads consecutive lines, so it is walked a line at a time for every candidate together.
    ways = np.full((n, nsets, 2), -1, np.int64)
    fill = np.zeros((n, nsets), np.int64)
    last = np.full(n, -1, np.int64)
    misses = np.zeros(n, np.int64)
    edges = sorted({e[0] for e in ev} | {kernel.rk0, kernel.rk1})

    def access(ln, active):
        new = active & (ln != last)
        if not new.any():
            return
        last[new] = ln[new]
        r, l = rows[new], ln[new]
        st = l % nsets
        hit = (ways[r, st, 0] == l) | (ways[r, st, 1] == l)
        r, l, st = r[~hit], l[~hit], st[~hit]
        f = fill[r, st]
        ways[r, st, f & 1] = l
        fill[r, st] = f + 1
        misses[r] += 1

    cur = kernel.sites[0]
    for pc, taken, target, _ in records:
        a = cur
        while a <= pc:
            b = min(
                [e - 4 for e in edges if e > a] + [pc]
            )  # last instruction before the next shift change
            b = min(b, pc)
            first, end = shifted(a) >> 4, shifted(b) >> 4
            for k in range(int((end - first).max()) + 1):
                access(first + k, first + k <= end)
            a = b + 4
        cur = target if taken else pc + 4

    # branch predictor: 2-bit counter and target per entry, updated BP_LAG instructions after the branch; the shift is
    # one to one, so targets compare by their unshifted addresses and only the entry index depends on the candidate
    cnt = np.zeros((n, 16), np.int64)
    tgt = np.full((n, 16), -1, np.int64)
    mispred = np.zeros(n, np.int64)
    pending, index = [], {}
    for pc, taken, target, ic in records:
        while pending and pending[0][0] <= ic:
            _, idx, c, t = pending.pop(0)
            cnt[rows, idx], tgt[rows, idx] = c, t
        idx = index.get(pc)
        if idx is None:
            idx = index[pc] = _bp_index(shifted(pc))
        c, e = cnt[rows, idx], tgt[rows, idx]
        pred = ((c & 2) == 0) & (e != pc)
        ok = (pred == bool(taken)) & ((not taken) | (e == target))
        mispred += ~ok
        right = e == target
        if taken:
            nc = np.where(~right, 1, np.where(c == 1, c, (c + 1) & 3))
        else:
            nc = np.where(~right, 2, np.where(c == 2, c, (c - 1) & 3))
        pending.append((ic + BP_LAG, idx, nc, target))
    return (
        (MISPREDICT_COST * mispred + MISS_COST * misses) / max(hi - lo, 1),
        mispred,
        misses,
    )


def flow_key(records, lo):
    """Identifies the modelled control flow, so runtime configurations that take the same path share a choice"""
    return hashlib.sha1(
        repr([(pc, t, tg, ic - lo) for pc, t, tg, ic in records]).encode()
    ).hexdigest()[:16]


def choose(elf_path, thread, runtime_bytes, cache_dir=None):
    """(P, Z) in bytes for one kernel thread and runtime configuration, (0, 0) when it has no measured loop.

    On the math thread the search runs over pads below 256 B (its 16 set cache repeats every 256 B) and then picks the
    256 B multiples for the predictor; the other threads search P and Z in turn.
    Runtime configurations with the same modelled control flow share a choice, cached as json in cache_dir.
    """
    kernel = _kernel(str(elf_path))
    if not kernel.parks or kernel.sites is None:
        return 0, 0
    tr = trace(kernel, runtime_bytes)
    if tr is None or tr[2] - tr[1] < 200 or not tr[0]:
        return 0, 0
    records, lo, hi = tr
    cached = (
        Path(cache_dir) / f"{thread}-{flow_key(records, lo)}.json"
        if cache_dir
        else None
    )
    if cached:
        try:
            return tuple(json.loads(cached.read_text()))
        except (
            OSError,
            ValueError,
        ):  # not chosen yet, or being written by another worker
            pass
    free = max(
        (kernel.elf.region_end or kernel.elf.text_end)
        - kernel.elf.text_end
        - CODE_HEADROOM,
        0,
    )

    def best(p, z, nsets):
        c = cost(kernel, records, lo, hi, nsets, p, z)[0]
        c = np.where(p + z <= free, c, np.inf)
        if not np.isfinite(c.min()):
            return 0, 0
        i = min(
            np.flatnonzero(c <= c.min() * (1 + 1e-9)).tolist(),
            key=lambda k: (z[k], p[k]),
        )
        return int(p[i]), int(z[i])

    if thread != "math":
        # 64 set cache: P alone, then Z, then P again (as good as the full search on the matmul unpack loops)
        p, _ = best(PADS, np.zeros_like(PADS), 64)
        _, z = best(np.full_like(PADS, p), PADS, 64)
        pads = best(PADS, np.full_like(PADS, z), 64)
    else:
        low = PADS[PADS < 256]
        p, z = best(np.repeat(low, len(low)), np.tile(low, len(low)), 16)
        pads = best(
            np.array([p, p + 256, p, p + 256]), np.array([z, z, z + 256, z + 256]), 16
        )
    if cached:
        write_json(cached, pads)
    return pads


def write_json(path, value):
    """Atomically, as several pytest workers can choose for the same variant at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


@functools.lru_cache(maxsize=64)
def _kernel(elf_path):
    return Kernel(elf_path)
