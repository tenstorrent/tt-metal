# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Host access to the Blackhole L2CPU tile (SiFive x280 CPUs) of one PCIe chip.

Two NoC backends with the same interface (read32 / write32 / read / write at raw NoC0 coordinates):
  * "ttnn" : ttnn.cluster of the calling process's open device (in-process with a tt-metal session);
  * "umd"  : tt_umd.TTDevice (a bare UMD handle; also a clock holder while it is open).
L2cpuHw(guard=True) wraps the backend with ClockGuard: an ARC read of the L2CPU PLL before EVERY access to the
tile, because a NoC access to the tile while its clock is off hangs the chip (see ../README.md).
Run every process that touches a chip under the lock used by scripts/l2cpu_run.sh.
"""
from __future__ import annotations

import os
import struct
import time

from . import layout as A

L2CPU_TILES = {0: (8, 3), 1: (8, 9), 2: (8, 5), 3: (8, 7)}
ARC = (8, 0)

# ARC tile registers (NoC addresses inside the ARC tile)
L2CPU_RESET = 0x80030014
PLL4_BASE = 0x80020500
PLL_CNTL_1 = 0x4
PLL_CNTL_5 = 0x14
# tt-bh-linux clock.py solutions: mhz -> (fbdiv, [postdiv0..3])
PLL_SOLUTIONS = {200: (128, [15, 15, 15, 15]), 1750: (140, [1, 1, 1, 1])}

# x280 physical addresses (reachable over NoC at the L2CPU tile: NoC addr == PA, passthrough); C side: l2cpu_hw.h
MEMPORT_BASE = 0x4000_3000_0000  # local GDDR (D5 for tile 0), cached, coherent with the x280 caches
CCACHE0_WAYENABLE = 0x0201_0008
L2PF1_BASE = 0x0203_0000  # + 0x2000*hart: BASIC_CONTROL +0, USER_CONTROL +4
RESET_VECTOR_PA = 0x2001_0000  # + 8*hart, u64
SCRATCH_PA = A.L2CPU_BOOT_RECORD_PA  # 64 bytes; the boot record lives here
HART_STATUS_PA = A.L2CPU_HART_STATUS  # u16 cease/halt/wfi/debug
MSI_CATCHER_PA = 0x2006_0000  # doorbell: a non-zero u32 written here wakes hart 0 (PLIC source 6)
PERIPH_HIGH_DELTA = 0xFFFF_F7FE_DFF0_0000  # NoC high alias of external peripherals = PA + delta
NIU0_NODE_ID = 0xFFFF_F7FE_FFF5_6000 + 0x44
REGION_ALIGN = 0x10000  # region base alignment inside a ttnn buffer (ttnn DRAM buffers are only 64 B aligned)

BOOT_MAGIC = A.L2CPU_BOOT_RECORD_MAGIC  # "L2CPBOOT", boot record at scratch +0x08


def check_clock(c5):
    """Raise unless PLL4 CNTL_5 (read through ARC, always safe) says the L2CPU clock runs. A NoC access to the tile
    while its clock is off hangs the chip until a reset (README "Hardware facts"). UMD and ttnn opens turn the clock
    on (driver power flag); pyluwen does not."""
    if c5 == 0xFFFFFFFF:
        raise RuntimeError("ARC reads 0xffffffff: chip is hung, reset the chip (tt-smi -r)")
    if (c5 & 0xFF) == 0:
        raise RuntimeError(
            "L2CPU clock is off (PLL4 CNTL_5 = 0): refusing to touch the L2CPU tile "
            "(open the chip with UMD or ttnn first)"
        )
    return c5


class UmdBackend:
    name = "umd"

    def __init__(self, pcie_id: int = 0):
        import tt_umd

        self.dev = tt_umd.TTDevice.create(pcie_id)
        try:
            self.dev.init_tt_device()
        except Exception as e:  # already initialised / not needed on some versions
            self.init_error = repr(e)

    def read32(self, x, y, addr):
        return self.dev.noc_read32(x, y, addr)

    def write32(self, x, y, addr, val):
        self.dev.noc_write32(x, y, addr, val & 0xFFFFFFFF)

    def read(self, x, y, addr, n):
        return bytes(self.dev.noc_read(x, y, addr, n))

    def write(self, x, y, addr, data):
        self.dev.noc_write(x, y, addr, bytes(data))


class TtnnClusterBackend:
    """In-process backend that shares ttnn's own UMD instance (ttnn.cluster.*). Needs an open ttnn device.
    Coordinates are TRANSLATED; ARC (8,0) and the L2CPU tiles are identity-mapped (checked by node_id())."""

    name = "ttnn"

    def __init__(self, device_id: int = 0):
        import ttnn

        self.c = ttnn.cluster
        self.dev = device_id

    def read32(self, x, y, addr):
        return self.c.read_reg(self.dev, x, y, addr)

    def write32(self, x, y, addr, val):
        self.c.write_to_core_immediate(self.dev, x, y, addr, struct.pack("<I", val & 0xFFFFFFFF))

    def read(self, x, y, addr, n):
        return bytes(self.c.read_from_core(self.dev, x, y, addr, n))

    def write(self, x, y, addr, data):
        self.c.write_to_core(self.dev, x, y, addr, bytes(data))


def make_backend(kind: str | None = None, pcie_id: int | None = None):
    kind = kind or os.environ.get("L2CPU_BACKEND", "umd")
    if pcie_id is None:
        pcie_id = int(os.environ.get("TT_DEV", "0"))
    if kind == "umd":
        return UmdBackend(pcie_id)
    if kind == "ttnn":
        return TtnnClusterBackend(0)  # logical id 0 = the one visible chip
    raise ValueError(kind)


class ClockGuard:
    """Backend wrapper: before EVERY access to the L2CPU tile, an ARC-only read checks that the L2CPU clock is on
    (PLL4 CNTL_5 low byte != 0). A NoC access to (8,3) with the clock off hangs the chip (README "Hardware facts")."""

    def __init__(self, inner, tile_xy):
        self.inner = inner
        self.tile_xy = tuple(tile_xy)
        self.name = getattr(inner, "name", "?") + "+guard"
        self.checks = 0

    def _check(self, x, y):
        if (x, y) != self.tile_xy:
            return
        self.checks += 1
        check_clock(self.inner.read32(*ARC, PLL4_BASE + PLL_CNTL_5))

    def read32(self, x, y, addr):
        self._check(x, y)
        return self.inner.read32(x, y, addr)

    def write32(self, x, y, addr, val):
        self._check(x, y)
        self.inner.write32(x, y, addr, val)

    def read(self, x, y, addr, n):
        self._check(x, y)
        return self.inner.read(x, y, addr, n)

    def write(self, x, y, addr, data):
        self._check(x, y)
        self.inner.write(x, y, addr, data)

    def __getattr__(self, k):  # telemetry() etc.
        return getattr(self.inner, k)


class L2cpuHw:
    """L2CPU tile `tile` (default 0 = CPUs 0-3 at NoC0 (8,3)) of one chip.
    guard=True: check the L2CPU clock before every access to the tile (ClockGuard)."""

    def __init__(self, backend=None, tile: int = 0, log=print, guard: bool = False):
        self.b = backend if backend is not None else make_backend()
        self.tile = tile
        self.x, self.y = L2CPU_TILES[tile]
        if guard and not isinstance(self.b, ClockGuard):
            self.b = ClockGuard(self.b, (self.x, self.y))
        self.log = log or (lambda *a, **k: None)
        self.assert_clock_on()

    def assert_clock_on(self):
        """Raise unless the L2CPU clock runs (see check_clock)."""
        return check_clock(self.arc_read32(PLL4_BASE + PLL_CNTL_5))

    # ---- raw NoC0 ----
    def noc_read32(self, x, y, addr):
        return self.b.read32(x, y, addr)

    def noc_write32(self, x, y, addr, val):
        self.b.write32(x, y, addr, val)

    def noc_read(self, x, y, addr, n):
        return self.b.read(x, y, addr, n)

    def noc_write(self, x, y, addr, data):
        self.b.write(x, y, addr, data)

    # ---- ARC (what pyluwen calls axi_read32/axi_write32 on Blackhole) ----
    def arc_read32(self, addr):
        return self.b.read32(*ARC, addr)

    def arc_write32(self, addr, val):
        self.b.write32(*ARC, addr, val)

    # ---- x280 physical address space through the tile's passthrough ----
    def pa_read32(self, pa):
        return self.b.read32(self.x, self.y, pa)

    def pa_write32(self, pa, val):
        self.b.write32(self.x, self.y, pa, val)

    def pa_read64(self, pa):
        lo, hi = struct.unpack("<II", self.b.read(self.x, self.y, pa, 8))
        return lo | (hi << 32)

    def pa_write64(self, pa, val):
        self.b.write(self.x, self.y, pa, struct.pack("<Q", val & (2**64 - 1)))

    def pa_read(self, pa, n):
        return self.b.read(self.x, self.y, pa, n)

    def pa_write(self, pa, data):
        self.b.write(self.x, self.y, pa, data)

    # external peripherals through the high alias (bypasses the x280 core complex)
    def periph_read32(self, pa):
        return self.b.read32(self.x, self.y, pa + PERIPH_HIGH_DELTA)

    def periph_write32(self, pa, val):
        self.b.write32(self.x, self.y, pa + PERIPH_HIGH_DELTA, val)

    def doorbell(self, value):
        """Ring the tile's MSI catcher doorbell (wakes hart 0). value must be non-zero."""
        self.pa_write32(MSI_CATCHER_PA, value)

    def node_id(self):
        v = self.b.read32(self.x, self.y, NIU0_NODE_ID)
        return v & 0x3F, (v >> 6) & 0x3F

    # ---- reset / clock ----
    def read_l2cpu_reset(self):
        return self.arc_read32(L2CPU_RESET)

    def read_pll(self):
        c1 = self.arc_read32(PLL4_BASE + PLL_CNTL_1)
        c5 = self.arc_read32(PLL4_BASE + PLL_CNTL_5)
        return c1, c5

    @staticmethod
    def pll_mhz(c1, c5, ref_mhz=50.0):
        refdiv, fbdiv = c1 & 0xFF, c1 >> 16
        pd0 = c5 & 0xFF
        return ref_mhz / max(refdiv, 1) * fbdiv / (pd0 + 1)

    def set_l2cpu_pll(self, mhz=None, fbdiv=None, postdivs=None):
        """Exactly tt-bh-linux clock.py set_l2cpu_pll: one-unit steps, postdiv increases, fbdiv, postdiv decreases."""
        if mhz is not None:
            fbdiv, postdivs = PLL_SOLUTIONS[mhz]
        c1, c5 = self.read_pll()
        pd = list(struct.pack("<I", c5))
        refdiv, postdiv, fb = c1 & 0xFF, (c1 >> 8) & 0xFF, c1 >> 16
        self.log(
            f"PLL4 before: CNTL_1=0x{c1:08x} CNTL_5=0x{c5:08x} (~{self.pll_mhz(c1, c5):.0f} MHz) -> fbdiv {fbdiv} postdivs {postdivs}"
        )

        def w5():
            self.arc_write32(PLL4_BASE + PLL_CNTL_5, struct.unpack("<I", bytes(pd))[0])
            time.sleep(1e-9)

        for i, t in enumerate(postdivs):
            while pd[i] < t:
                pd[i] += 1
                w5()
        while fb != fbdiv:
            fb += 1 if fbdiv > fb else -1
            self.arc_write32(PLL4_BASE + PLL_CNTL_1, refdiv | (postdiv << 8) | (fb << 16))
            time.sleep(1e-9)
        for i, t in enumerate(postdivs):
            while pd[i] > t:
                pd[i] -= 1
                w5()
        c1, c5 = self.read_pll()
        self.log(f"PLL4 after:  CNTL_1=0x{c1:08x} CNTL_5=0x{c5:08x} (~{self.pll_mhz(c1, c5):.0f} MHz)")
        return c1, c5

    def set_reset_vectors(self, entry, harts=range(4)):
        for h in harts:
            pa = RESET_VECTOR_PA + 8 * h
            self.periph_write32(pa, entry & 0xFFFFFFFF)
            self.periph_write32(pa + 4, entry >> 32)
        got = [
            self.periph_read32(RESET_VECTOR_PA + 8 * h) | (self.periph_read32(RESET_VECTOR_PA + 8 * h + 4) << 32)
            for h in harts
        ]
        if any(g != entry for g in got):
            raise RuntimeError(f"reset vector readback {[hex(g) for g in got]} != 0x{entry:x}")
        return got

    def read_scratch(self):
        return self.b.read(self.x, self.y, SCRATCH_PA + PERIPH_HIGH_DELTA, 64)

    def write_scratch_u64(self, off, val):
        self.periph_write32(SCRATCH_PA + off, val & 0xFFFFFFFF)
        self.periph_write32(SCRATCH_PA + off + 4, (val >> 32) & 0xFFFFFFFF)

    def set_wayenable(self, n=15):
        self.pa_write32(CCACHE0_WAYENABLE, n)
        return self.pa_read32(CCACHE0_WAYENABLE)

    def set_prefetchers(self):
        for h in range(4):
            self.pa_write32(L2PF1_BASE + 0x2000 * h, 0x15811)
            self.pa_write32(L2PF1_BASE + 0x2000 * h + 4, 0x38C84E)
        return [
            (self.pa_read32(L2PF1_BASE + 0x2000 * h), self.pa_read32(L2PF1_BASE + 0x2000 * h + 4)) for h in range(4)
        ]

    def release(self, low_mhz=200, high_mhz=1750):
        """tt-bh-linux boot.py reset_x280: PLL to low, set bit (4+tile) RMW, read back, PLL to high.
        Refuses if the tile is already released (harts can leave reset only once per chip reset)."""
        r = self.read_l2cpu_reset()
        bit = 1 << (4 + self.tile)
        if r & bit:
            raise RuntimeError(
                f"L2CPU_RESET=0x{r:08x}: tile {self.tile} already released in this reset epoch; reset the chip (tt-smi -r)"
            )
        c1, c5 = self.read_pll()
        orig = (c1 >> 16, list(struct.pack("<I", c5)))
        self.set_l2cpu_pll(low_mhz)
        self.arc_write32(L2CPU_RESET, r | bit)
        r2 = self.read_l2cpu_reset()
        self.log(f"L2CPU_RESET 0x{r:08x} -> 0x{r2:08x}")
        if high_mhz is None:  # back to whatever ARC had programmed (800 MHz after reset on our chips)
            self.set_l2cpu_pll(fbdiv=orig[0], postdivs=orig[1])
        else:
            self.set_l2cpu_pll(high_mhz)
        return r2

    def hart_status(self):
        return self.periph_read32(HART_STATUS_PA) & 0xFFFF

    # ---- memory port helpers ----
    def load_image(self, data: bytes, pa: int, verify=True):
        if len(data) % 4:
            data = data + b"\0" * (4 - len(data) % 4)
        self.pa_write(pa, data)
        if verify:
            got = self.pa_read(pa, len(data))
            if got != data:
                bad = next(i for i in range(len(data)) if got[i] != data[i])
                raise RuntimeError(f"image readback mismatch at +0x{bad:x}")
        return len(data)
