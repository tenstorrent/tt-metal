# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the read-only DRAM tile probes: tile listing, register reads, card busy check.
Nothing here writes a device register.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from tt_llk_perf import headers  # noqa: E402

NIU_COUNTERS = {
    "RD_REQ_RECEIVED": 0x2D4,
    "RD_DATA_WORD_SENT": 0x2CC,
    "NONPOSTED_WR_REQ_RECEIVED": 0x2E8,
    "POSTED_WR_REQ_RECEIVED": 0x2EC,
    "NONPOSTED_WR_DATA_WORD_RECEIVED": 0x2E0,
    "POSTED_WR_DATA_WORD_RECEIVED": 0x2E4,
}


def niu_addr(tile, noc, offset, arch="blackhole"):
    """Address of an NIU register of a DRAM tile given by its noc0 (x, y)."""
    return headers.dram_niu_address(arch, noc, tile[1], offset)


def card_busy(device_path):
    """True when another process has the device node open."""
    target = os.path.realpath(device_path)
    me = str(os.getpid())
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == me:
            continue
        fd_dir = os.path.join("/proc", pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(os.path.join(fd_dir, fd)) == target:
                    return True
            except OSError:
                continue
    return False


def executable(path):
    """Absolute path of an existing executable file, or exit."""
    resolved = Path(path).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise SystemExit(f"not an executable file: {path}")
    return str(resolved)


class Reader:
    """One way of reading a 32-bit register of a DRAM tile addressed by NoC0 coordinates."""

    name = "?"

    def tiles(self):
        raise NotImplementedError

    def read32(self, tile, addr):
        raise NotImplementedError

    def read_bytes(self, tile, addr, size):
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

    def read_bytes(self, tile, addr, size):
        return bytes(self._dev.noc_read(tile[0], tile[1], addr, size))


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

    def read_bytes(self, tile, addr, size):
        from ttexalens.coordinate import OnChipCoordinate

        loc = OnChipCoordinate.create(f"{tile[0]},{tile[1]}", self._dev, "noc0")
        buf = bytearray(size)
        loc.noc_read(addr, buf)
        return bytes(buf)


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


def snapshot(reader, tiles, arch="blackhole"):
    """{(tile, noc, counter): value} of every NIU counter on both NoCs."""
    begin = getattr(reader, "begin_batch", None)
    if begin:
        begin()
    return {
        (t, noc, name): reader.read32(t, niu_addr(t, noc, off, arch))
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
