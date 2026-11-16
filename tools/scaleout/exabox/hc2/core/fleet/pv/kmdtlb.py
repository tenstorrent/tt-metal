# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""kmdtlb: run-ttx's PCIe TLB windows come from the driver (tt-kmd), never from fixed indices.

Stock run-ttx programs fixed PCIe TLB windows itself (init_static_tlbs: windows 0-199) and so overwrote the windows
tt_telemetry_collector had allocated from the driver (0 and 1): chips hung and the host took an MCE.
This file runs inside run-ttx as a PyInstaller runtime hook (repack.py puts it before run-ttx's main script; it is
Python 3.7 code, as run-ttx is). It changes only where run-ttx's windows live:

- run-ttx keeps its own window numbers (0-209) as logical numbers for its bookkeeping (_tlb_state, _tlb_targets, its
  dynamic pools and locks); programming one (PCIe._program_tlb) only records the wanted address and flags.
- At each access (PCIe._tlb_access_info, inside get_tlb's with-block) the logical window is bound to a window this
  process allocated from the driver (TENSTORRENT_IOCTL_ALLOCATE_TLB on its own fd of the same chip), programmed there
  with run-ttx's own register math (its _program_tlb at that window's number) when its config differs, and the access
  goes through that window's offset. Up to KMDTLB_WINDOWS windows (default 32) are allocated; past that the least
  recently used window not inside an open with-block is re-programmed (a window in use is never moved).
- Every write to the PCIe TLB config registers is checked: one that is not the programming of a window this process
  owns raises RuntimeError before it reaches the chip (no fixed index is ever programmed).
- The windows are freed (TENSTORRENT_IOCTL_FREE_TLB) when run-ttx closes the chip, at exit, and on SIGTERM or SIGHUP
  (made a SystemExit when no handler was set); the driver also frees them when the fd closes (any death, SIGKILL too).

run-ttx's TtxRun preloads, noc_init and every other write are unchanged: they reach the chip through these windows.
Without a driver window (no ioctl, no free window) run-ttx stops with an error before any write; it never falls back to a
fixed index.
"""

import atexit
import ctypes
import os
import signal
import struct
import sys
import threading
from collections import OrderedDict

TLB_REG_BASE = 0x1FC00000
COUNT_2M = 202
COUNT_4G = 8
COUNT = COUNT_2M + COUNT_4G
REG_END = TLB_REG_BASE + COUNT * 12  # config regs: 12 bytes per window
EXT_END = REG_END + COUNT * 4  # ext regs (window 32 and up): 4 bytes per window
SIZE_2M = 1 << 21
SIZE_4G = 1 << 32

# tt-kmd ioctl.h: _IO(TENSTORRENT_IOCTL_MAGIC 0xFA, nr)
IOCTL_ALLOCATE_TLB = (0xFA << 8) | 11
IOCTL_FREE_TLB = (0xFA << 8) | 12


def log(msg):
    try:
        sys.stderr.write("kmdtlb: %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass


class Kmd(object):
    """the driver's TLB allocator on one fd of one chip"""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.ioctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)

    def _ioctl(self, req, buf):
        b = ctypes.create_string_buffer(bytes(buf), len(buf))
        if self._libc.ioctl(self.fd, req, b) != 0:
            e = ctypes.get_errno()
            raise OSError(e, os.strerror(e))
        return b.raw

    def allocate(self, size):
        """the window id (= its hardware index) of a free window of this size, or OSError (ENOMEM: none free)"""
        out = self._ioctl(IOCTL_ALLOCATE_TLB, struct.pack("<QQ", size, 0) + b"\0" * 32)
        return struct.unpack_from("<I", out, 16)[0]

    def free(self, wid):
        self._ioctl(IOCTL_FREE_TLB, struct.pack("<I", wid))

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def bdf_path(bdf, sysroot="/sys"):
    """the /dev/tenstorrent/<n> node whose PCI device is bdf ("bus:slot.func" in hex as run-ttx's get_pci_bdf prints it,
    e.g. "c1:00.0", with or without a "0000:" domain), or ''"""
    try:
        b, rest = bdf.strip().split(":")[-2:]
        s, f = rest.split(".")
        want = (int(b, 16), int(s, 16), int(f, 16))
    except ValueError:
        return ""
    d = os.path.join(sysroot, "class", "tenstorrent")
    try:
        names = os.listdir(d)
    except OSError:
        return ""
    for n in sorted(names):
        try:
            real = os.path.basename(os.path.realpath(os.path.join(d, n, "device")))
            _dom, bus, sf = real.split(":")
            slot, fn = sf.split(".")
            got = (int(bus, 16), int(slot, 16), int(fn, 16))
        except (OSError, ValueError):
            continue
        if got == want:
            return "/dev/tenstorrent/" + n.split("!")[-1]
    return ""


class Windows(object):
    """one chip's logical -> driver window map"""

    def __init__(self, kmd, cap, describe, program):
        self.kmd = kmd
        self.cap = cap
        self._describe = describe  # run-ttx's own PCIe._describe_tlb (unpatched)
        self._program = program  # run-ttx's own PCIe._program_tlb (unpatched)
        self.lock = threading.RLock()
        self.want = {}  # logical -> (address, flags) run-ttx programmed
        self.phys = OrderedDict()  # logical -> driver window, least recently used first
        self.free_ids = {SIZE_2M: [], SIZE_4G: []}  # allocated, bound to no logical window
        self.owned = set()  # every window id this process holds from the driver
        self.state = {}  # driver window -> (address, flags) programmed there
        self.pins = {}  # logical -> open with-blocks using it
        self.programming = None  # the window whose config regs are being written now
        self.closed = False
        self.allocs = 0
        self.reprograms = 0

    @staticmethod
    def size_of(logical):
        return SIZE_2M if logical < COUNT_2M else SIZE_4G

    def _take(self, logical):
        size = self.size_of(logical)
        if self.free_ids[size]:
            return self.free_ids[size].pop()
        n = sum(1 for lg in self.phys if self.size_of(lg) == size)
        if n < self.cap:
            try:
                wid = self.kmd.allocate(size)
            except OSError as e:
                wid = None
                if not n:
                    raise RuntimeError("kmdtlb: no driver TLB window of %d bytes for run-ttx (%s)" % (size, e))
            if wid is not None:
                if wid in self.owned or not (0 <= wid < COUNT) or self.size_of(wid) != size:
                    raise RuntimeError("kmdtlb: the driver returned window %r for a %d byte request" % (wid, size))
                self.owned.add(wid)
                self.allocs += 1
                return wid
        for lg in self.phys:  # least recently used first
            if self.size_of(lg) == size and not self.pins.get(lg):
                return self.phys.pop(lg)
        raise RuntimeError("kmdtlb: every driver TLB window is in use (%d)" % n)

    def bind(self, pcie, logical):
        """the driver window logical lives in now, programmed with run-ttx's wanted config"""
        if self.closed:
            raise RuntimeError("kmdtlb: chip closed, run-ttx accessed window %d after it" % logical)
        if logical not in self.want:
            raise RuntimeError("kmdtlb: run-ttx accessed window %d it never programmed" % logical)
        wid = self.phys.get(logical)
        if wid is None:
            wid = self._take(logical)
            self.phys[logical] = wid
        else:
            self.phys.move_to_end(logical)
        want = self.want[logical]
        if self.state.get(wid) != want:
            self.programming = wid
            try:
                self._program(pcie, self._describe(pcie, wid), want[0], want[1])
            finally:
                self.programming = None
            self.state[wid] = want
            self.reprograms += 1
        return wid

    def check_write(self, offset, length):
        """RuntimeError unless [offset, offset+length) of BAR0 misses the TLB config regs or is the window being programmed"""
        lo, hi = offset, offset + max(length, 1)
        if hi <= TLB_REG_BASE or lo >= EXT_END:
            return
        ok = set()
        if self.programming is not None:
            w = self.programming
            ok = {(TLB_REG_BASE + w * 12, TLB_REG_BASE + w * 12 + 12), (REG_END + w * 4, REG_END + w * 4 + 4)}
        if not any(a <= lo and hi <= b for a, b in ok):
            raise RuntimeError(
                "kmdtlb: refused a write to the PCIe TLB config regs at 0x%x (+%d): not a driver window of ours"
                % (lo, hi - lo)
            )

    def close(self):
        """free every window (once); later accesses raise"""
        with self.lock:
            if self.closed:
                return 0
            self.closed = True
            n = 0
            for wid in sorted(self.owned):
                try:
                    self.kmd.free(wid)
                    n += 1
                except OSError as e:
                    log("free of window %d failed (%s); the driver frees it when the fd closes" % (wid, e))
            self.owned.clear()
            self.phys.clear()
            self.kmd.close()
            return n


_all = []  # every Windows, for the exit and signal frees
_tls = threading.local()


def close_all(why):
    for w in list(_all):
        n = w.close()
        if n:
            log(
                "%s: freed %d driver TLB windows of %s (%d allocated, %d programmings)"
                % (why, n, w.kmd.path, w.allocs, w.reprograms)
            )


def _on_term(n, _f):
    raise SystemExit(128 + n)


def install(PCIe, HWConnectionPci, cap=None, sysroot="/sys", trap=True):
    """patch run-ttx's PCIe and HWConnectionPci classes (the module objects are passed in, so tests can use stand-ins)"""
    if getattr(PCIe, "_kmdtlb", False):
        return
    cap = cap or int(os.environ.get("KMDTLB_WINDOWS", "32") or 32)
    o_describe = PCIe._describe_tlb
    o_program = PCIe._program_tlb
    o_access = PCIe._tlb_access_info
    o_common = PCIe._get_tlb_common

    def windows(pcie):
        hw = getattr(pcie, "_hw_connection", None)
        if hw is None:
            raise RuntimeError("kmdtlb: run-ttx's PCIe has no PCI connection")
        w = getattr(hw, "_kmdtlb", None)
        if w is None:
            bdf = hw.get_pci_bdf()
            path = bdf_path(bdf, sysroot)
            if not path:
                raise RuntimeError("kmdtlb: no /dev/tenstorrent node for chip %s" % bdf)
            w = Windows(Kmd(path), cap, o_describe, o_program)
            hw._kmdtlb = w
            _all.append(w)
            log("chip %s (%s): run-ttx's TLB windows come from the driver (up to %d)" % (bdf, path, cap))
        return w

    def _program_tlb(self, tlb, address, flags):
        w = windows(self)
        with w.lock:
            w.want[tlb.index] = (address, flags)
            wid = w.phys.get(tlb.index)
            if wid is not None and not w.pins.get(tlb.index):  # re-bound at its next access, programmed there
                w.phys.pop(tlb.index)
                w.free_ids[w.size_of(tlb.index)].append(wid)
            elif wid is not None:  # in use by an open with-block: program it in place, as run-ttx would
                w.programming = wid
                try:
                    o_program(self, o_describe(self, wid), address, flags)
                finally:
                    w.programming = None
                w.state[wid] = (address, flags)
                w.reprograms += 1

    def _tlb_access_info(self, tlb, address):
        w = windows(self)
        with w.lock:
            wid = w.bind(self, tlb.index)
            stack = getattr(_tls, "frames", None)
            if stack:
                stack[-1].append(tlb.index)
                w.pins[tlb.index] = w.pins.get(tlb.index, 0) + 1
        return o_access(self, o_describe(self, wid), address)

    def _get_tlb_common(self, address, tlb_flags, register_access):
        w = windows(self)
        frame = []
        if not hasattr(_tls, "frames"):
            _tls.frames = []
        _tls.frames.append(frame)
        try:
            return (yield from o_common(self, address, tlb_flags, register_access))
        finally:
            try:
                _tls.frames.remove(frame)
            except ValueError:
                pass
            with w.lock:
                for lg in frame:
                    w.pins[lg] -= 1

    PCIe._program_tlb = _program_tlb
    PCIe._tlb_access_info = _tlb_access_info
    PCIe._get_tlb_common = _get_tlb_common
    PCIe._kmdtlb = True

    def guard(fn, kind):
        def wrapped(self, *a):
            w = getattr(self, "_kmdtlb", None)
            if kind == "bar":
                bar, off = a[0], a[1]
                n = 8 if fn.__name__ == "bar_write64" else 4
                if bar == 0:
                    (w or _NOWIN).check_write(off, n)
            else:  # write32/write64/block_write: BAR0 offsets
                off = a[0]
                n = a[1] if kind == "block" else (8 if fn.__name__ == "write64" else 4)
                (w or _NOWIN).check_write(off, n)
            return fn(self, *a)

        wrapped.__name__ = fn.__name__
        return wrapped

    for name, kind in (
        ("bar_write32", "bar"),
        ("bar_write64", "bar"),
        ("write32", "w"),
        ("write64", "w"),
        ("block_write", "block"),
    ):
        if hasattr(HWConnectionPci, name):
            setattr(HWConnectionPci, name, guard(getattr(HWConnectionPci, name), kind))

    o_release = getattr(HWConnectionPci, "release", None)
    if o_release is not None:

        def release(self, *a, **k):
            try:
                return o_release(self, *a, **k)
            finally:
                w = getattr(self, "_kmdtlb", None)
                if w is not None:
                    n = w.close()
                    if n:
                        log(
                            "chip closed: freed %d driver TLB windows of %s (%d allocated, %d programmings)"
                            % (n, w.kmd.path, w.allocs, w.reprograms)
                        )

        HWConnectionPci.release = release

    if trap:
        atexit.register(close_all, "exit")
        for s in (signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(s) == signal.SIG_DFL:
                signal.signal(s, _on_term)


class _NoWindows(object):
    programming = None

    def check_write(self, offset, length):
        Windows.check_write(self, offset, length)


_NOWIN = _NoWindows()


def main():
    """the runtime hook's entry: patch run-ttx's own classes before its main script runs"""
    from tenstorrent.chip.pcie import PCIe
    from tenstorrent.hw_connection import HWConnectionPci

    install(PCIe, HWConnectionPci)
