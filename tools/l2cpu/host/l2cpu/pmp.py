# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""PMP policy of one L2CPU tile (README "PMP policy"): the host computes the locked entries, the firmware writes them.

    pol = build(region_pa, region_size)          # default: the runtime's own TLB windows 0..31 (uncached)
    pol.table()                                  # bytes for the resident page at L2CPU_RES_PMP
    pol.allows(pa, 4, R)                         # the same first-match decision as the hardware (and fw_pmp_allows)
    print(pol.describe())

Entries, in priority order (lowest index wins; every entry LOCKED, so it also checks M-mode; the last one denies
everything else; the x280 has 8 entries and no Smepmp). Every base and size is a multiple of the PMP granule
measured on the chip (HW_GRANULE).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

from . import layout as A

R, W, X, L = 0x01, 0x02, 0x04, 0x80
A_OFF, A_TOR, A_NA4, A_NAPOT = 0 << 3, 1 << 3, 2 << 3, 3 << 3

# x280 PMP, measured on the chip (README "Hardware facts", scripts/pmp_probe.py, logs pr8_probe*.log)
HW_ENTRIES = 8  # pmpaddr0..7; pmpaddr8..15 read 0, pmpaddr16+ trap; no Smepmp (mseccfg traps)
HW_GRANULE = 4096  # G = 10
HW_ADDR_MASK = (1 << 45) - 1  # pmpaddr bits 44..0 (PA < 2^47); NAPOT read of all-ones = the deny-all value

UNCACHED_BASE = 0x3000_0000  # System Port (uncached) alias of local DRAM offset 0
UNCACHED_DELTA = 0x4000_0000_0000  # cached Memory Port PA - uncached System Port PA of the same local DRAM byte
DRAM_BANK_BYTES = 4 << 30  # one GDDR channel; the uncached block must stay inside it
TLB2M_UNCACHED, TLB2M_CACHED, TLB2M_SIZE, TLB2M_COUNT = 0x4_3000_0000, 0x4004_3000_0000, 2 << 20, 224
RUNTIME_WINDOWS = ((0, 32, False),)  # platform_bh.c: 16 mapping slots = small windows 0..31, uncached

# Address ranges the firmware and the applications use (register survey, README "PMP policy"), merged to fit 8.
CORE_COMPLEX = 0x2000_1000  # [0, this) RW: CLINT 0x0200_0000, L3 controller 0x0201_0000 (WAYENABLE, FLUSH64),
#   L2 prefetchers 0x0203_0000, PLIC 0x0C00_0000 (internal buses, not the NoC) + the TLB window config page
PERIPH_RO_END = 0x2008_0000  # [CORE_COMPLEX, this) R: control page (reset vectors, boot record, hart status, RNMI),
#   watchdogs, NIUs, MSI catcher (the doorbell drain is a read); every word read without a fault or a hang in the
#   register survey. The DW_ahb_dmac at 0x2008_0000 and everything above is outside.


class PmpError(ValueError):
    pass


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def napot(base, size):
    if not _pow2(size) or size < 8 or base % size:
        raise PmpError(f"NAPOT needs a power-of-two size >= 8 aligned to itself: 0x{base:x} + 0x{size:x}")
    return (base >> 2) | ((size >> 3) - 1)


@dataclass(frozen=True)
class Entry:
    name: str
    addr: int  # pmpaddr value (reads back unchanged)
    cfg: int  # pmpcfg byte


@dataclass(frozen=True)
class Policy:
    region: int
    region_size: int
    windows: tuple
    uncached: bool
    entries: tuple = field(compare=True)

    def table(self) -> bytes:
        n = len(self.entries)
        addrs = [e.addr for e in self.entries] + [0] * (A.L2CPU_PMP_MAX - n)
        cfgs = [e.cfg for e in self.entries] + [0] * (A.L2CPU_PMP_MAX - n)
        return struct.pack(f"<II{A.L2CPU_PMP_MAX}Q{A.L2CPU_PMP_MAX}B", A.L2CPU_PMP_MAGIC, n, *addrs, *cfgs)

    def ranges(self):
        """(name, lo, hi_exclusive, perm, index) per active entry."""
        out = []
        for i, e in enumerate(self.entries):
            mode = e.cfg & 0x18
            if mode == A_OFF:
                continue
            if mode == A_TOR:
                lo, hi = (self.entries[i - 1].addr << 2 if i else 0), e.addr << 2
            elif mode == A_NA4:
                lo, hi = e.addr << 2, (e.addr << 2) + 4
            else:
                ones = (~e.addr & (e.addr + 1)).bit_length() - 1
                if ones >= 61:
                    lo, hi = 0, 1 << 64
                else:
                    lo = (e.addr & ~((1 << ones) - 1)) << 2
                    hi = lo + (1 << (ones + 3))
            out.append((e.name, lo, hi, e.cfg & 7, i))
        return out

    def allows(self, pa, length, perm):
        last = pa + length - 1
        for _, lo, hi, p, _ in self.ranges():
            first_in, last_in = lo <= pa < hi, lo <= last < hi
            if not first_in and not last_in and not (pa < lo and last >= hi):
                continue
            return first_in and last_in and (p & perm) == perm
        return True

    def describe(self):
        lines = []
        for i, e in enumerate(self.entries):
            mode = {A_OFF: "OFF", A_TOR: "TOR", A_NA4: "NA4", A_NAPOT: "NAPOT"}[e.cfg & 0x18]
            perm = "".join(c if e.cfg & b else "-" for c, b in (("R", R), ("W", W), ("X", X)))
            rng = next(((lo, hi) for _, lo, hi, _, j in self.ranges() if j == i), None)
            span = f"[0x{rng[0]:x}, 0x{rng[1]:x})" if rng else ""
            lines.append(f"{i:2d} {mode:5s} {perm} {'L' if e.cfg & L else '-'} {e.name:34s} {span}")
        return "\n".join(lines)


def _covering_napot(lo, hi):
    """Smallest naturally aligned power-of-two block containing [lo, hi)."""
    size = 1 << max(12, (hi - lo - 1).bit_length())
    while lo // size != (hi - 1) // size:
        size <<= 1
    return lo // size * size, size


def build(
    region,
    region_size,
    windows=RUNTIME_WINDOWS,
    uncached=True,
    entries=HW_ENTRIES,
    granule=HW_GRANULE,
    addr_mask=HW_ADDR_MASK,
):
    """region: cached Memory Port PA of the region; region_size: bytes the firmware and the application use;
    windows: ((first small TLB window, count, cached),) the application maps (one range: 8 entries);
    uncached: allow READS of the region through its uncached alias (a naturally aligned block around it)."""
    if region_size < A.L2CPU_REGION_MIN_SIZE or region_size % granule or region % granule:
        raise PmpError(f"region 0x{region:x} + 0x{region_size:x}: below the minimum or not granule aligned")
    if len(windows) != 1:
        raise PmpError(f"one TLB window range fits in {entries} entries, got {windows}")
    res = region + A.L2CPU_OFF_RESIDENT
    ents = [
        Entry("core complex + TLB window config", CORE_COMPLEX >> 2, L | A_TOR | R | W),
        Entry("external peripherals (read only)", PERIPH_RO_END >> 2, L | A_TOR | R),
        Entry("resident page code + control", napot(res, A.L2CPU_RES_PROTECTED), L | A_NAPOT | R | X),
        Entry("region (TOR base)", region >> 2, L | A_OFF),
        Entry("region (Memory Port, cached)", (region + region_size) >> 2, L | A_TOR | R | W | X),
    ]
    if uncached:
        lo = region - UNCACHED_DELTA
        b, sz = _covering_napot(lo, lo + region_size)
        if b < UNCACHED_BASE or b + sz > UNCACHED_BASE + DRAM_BANK_BYTES:
            raise PmpError(
                f"uncached alias [0x{lo:x}, 0x{lo + region_size:x}) needs the block [0x{b:x}, 0x{b + sz:x}), which "
                "leaves the local DRAM: allocate the region elsewhere or use uncached=False"
            )
        ents.append(Entry("region, uncached alias (read only)", napot(b, sz), L | A_NAPOT | R))
    ((first, count, cached),) = windows
    if first < 0 or count <= 0 or first + count > TLB2M_COUNT:
        raise PmpError(f"TLB windows {first}+{count} outside 0..{TLB2M_COUNT - 1}")
    wb, wsz = (TLB2M_CACHED if cached else TLB2M_UNCACHED) + first * TLB2M_SIZE, count * TLB2M_SIZE
    if not _pow2(wsz) or wb % wsz:
        raise PmpError(f"TLB windows {first}..{first + count - 1}: not a naturally aligned power-of-two block")
    ents.append(
        Entry(
            f"TLB windows {first}..{first + count - 1}{' cached' if cached else ''}",
            napot(wb, wsz),
            L | A_NAPOT | R | W,
        )
    )
    while len(ents) < entries - 1:
        ents.append(Entry("unused", 0, L | A_OFF))
    ents.append(Entry("deny everything else", addr_mask, L | A_NAPOT))
    if len(ents) > entries:
        raise PmpError(f"policy needs {len(ents)} PMP entries, the x280 has {entries}")
    return Policy(region, region_size, tuple(tuple(w) for w in windows), uncached, tuple(ents))


def default_enabled():
    """L2CPU_PMP=0 in the environment turns the policy off (default on)."""
    return os.environ.get("L2CPU_PMP", "1") not in ("0", "off", "no", "")
