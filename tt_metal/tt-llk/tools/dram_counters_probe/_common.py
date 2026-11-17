# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the read-only DRAM tile probes: tile listing, register reads, card busy check.
Nothing here writes a device register.
"""

import os
import subprocess
import sys

NIU_BASE = 0xFFB20000
NOC_STRIDE = 0x10000
NIU_COUNTERS = {
    "RD_REQ_RECEIVED": 0x2D4,
    "RD_DATA_WORD_SENT": 0x2CC,
    "NONPOSTED_WR_REQ_RECEIVED": 0x2E8,
    "POSTED_WR_REQ_RECEIVED": 0x2EC,
    "NONPOSTED_WR_DATA_WORD_RECEIVED": 0x2E0,
    "POSTED_WR_DATA_WORD_RECEIVED": 0x2E4,
}


def niu_addr(noc, offset):
    return NIU_BASE + noc * NOC_STRIDE + offset


def card_busy(device_path):
    """True when another process has the device node open."""
    try:
        out = subprocess.run(
            ["fuser", device_path], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(ch.isdigit() for ch in out)


class Reader:
    """One way of reading a 32-bit register of a DRAM tile addressed by NoC0 coordinates."""

    name = "?"

    def tiles(self):
        raise NotImplementedError

    def read32(self, tile, addr):
        raise NotImplementedError

    def close(self):
        pass


class UmdReader(Reader):
    """tt_umd TTDevice directly, no exalens: init_tt_device only."""

    name = "umd"

    def __init__(self, device_number=0):
        import tt_umd

        self._umd = tt_umd
        self._dev = tt_umd.TTDevice.create(device_number)
        self._dev.init_tt_device()
        self._soc = tt_umd.SocDescriptor(self._dev)

    def tiles(self):
        cores = self._soc.get_cores(self._umd.CoreType.DRAM, self._umd.CoordSystem.NOC0)
        return [(c.x, c.y) for c in cores]

    def read32(self, tile, addr):
        return self._dev.noc_read32(tile[0], tile[1], addr)


class ExalensReader(Reader):
    """ttexalens context. reinit=True builds a new context for every read batch (see begin_batch)."""

    def __init__(self, reinit=False, safe_mode=True):
        self.name = "exalens-reinit" if reinit else "exalens"
        self._reinit = reinit
        self._safe = safe_mode
        self._ctx = None
        self._open()

    def _open(self):
        from ttexalens.tt_exalens_init import init_ttexalens

        self._ctx = init_ttexalens(safe_mode=self._safe)
        self._dev = self._ctx.devices[0]

    def begin_batch(self):
        if self._reinit:
            self._open()

    def tiles(self):
        out = []
        for loc in self._dev.get_block_locations("dram"):
            x, y = loc.to_str("noc0").replace(",", "-").split("-")
            out.append((int(x), int(y)))
        return out

    def read32(self, tile, addr):
        from ttexalens.coordinate import OnChipCoordinate

        loc = OnChipCoordinate.create(f"{tile[0]},{tile[1]}", self._dev, "noc0")
        return loc.noc_read32(addr)


def make_reader(mode, device_path=None):
    if mode == "umd":
        return UmdReader(int(os.path.basename(device_path or default_device_path())))
    if mode == "exalens":
        return ExalensReader()
    if mode == "exalens-reinit":
        return ExalensReader(reinit=True)
    if mode == "exalens-unsafe":
        return ExalensReader(safe_mode=False)
    raise SystemExit(f"unknown mode {mode}")


def snapshot(reader, tiles):
    """{(tile, noc, counter): value} of every NIU counter on both NoCs."""
    begin = getattr(reader, "begin_batch", None)
    if begin:
        begin()
    return {
        (t, noc, name): reader.read32(t, niu_addr(noc, off))
        for t in tiles
        for noc in (0, 1)
        for name, off in NIU_COUNTERS.items()
    }


def wrap_delta(after, before):
    return (after - before) & 0xFFFFFFFF


def fail(msg):
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def default_device_path():
    return os.environ.get("TT_DEVICE_PATH", "/dev/tenstorrent/1")
