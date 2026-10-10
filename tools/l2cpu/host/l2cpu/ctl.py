# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Bring-up, stop, restart and status of the x280 harts of one L2CPU tile, WITHOUT a chip reset.

    ctl = L2cpuCtl(L2cpuHw(backend, tile=t), region_pa)   # tile t (default 0); region in the tile's local DRAM
    ctl.start(image)                      # cold: fresh chip epoch only (a tile is released once per chip reset)
    start_tiles([ctl0, ctl1, ...], image) # several tiles of the chip, released by one L2CPU_RESET write
    ctl.stop()                            # every hart parked in the resident page -> per-hart records
    ctl.restart(image=None, warm=True, slot=None)
    ctl.status(); ctl.is_alive(); ctl.log_text(); ctl.ensure_clock()

Levels: L1 = mailbox PARK (cooperative), L2 = RNMI (host trigger bits), L3 = chip reset (outside this module).
PMP: start() locks a per-epoch policy (pmp.py, README "PMP policy"); restart() keeps it and refuses another one.
Layout: ../../include/l2cpu_boot.h (mirror: layout.py). Every wait below is bounded (README "Waits and their bounds").
"""
from __future__ import annotations

import os
import struct
import time

from . import layout as A
from . import pmp as P
from .hw import ClockGuard, L2cpuHw

HERE = os.path.dirname(os.path.abspath(__file__))
L2CPU_DIR = os.path.dirname(os.path.dirname(HERE))  # tools/l2cpu
RESIDENT_BLOB = os.environ.get("L2CPU_RESIDENT_BLOB") or os.path.join(
    L2CPU_DIR, "fw", "build", "resident", "resident-bh.bin"
)
REC_FIELDS = (
    "state",
    "parked_epoch",
    "kind",
    "rnmi_count",
    "mncause",
    "mnepc",
    "mnstatus",
    "mcause",
    "mepc",
    "mtval",
    "mstatus",
)
PMP_STATE = {0: "OFF", 1: "APPLIED", 2: "REUSED"}
STATE = {0: "RUNNING", 1: "IN_RNMI", 2: "PARKED", 3: "LEAVING"}
KIND = {0: "-", 1: "SOFT", 2: "RNMI", 3: "RNMI_EXC", 4: "ERROR"}


class L2cpuCtlError(RuntimeError):
    pass


class L2cpuCtl:
    def __init__(self, hw: L2cpuHw, base: int, log=print, mhz: int = 1750, region_size: int = A.L2CPU_REGION_MIN_SIZE):
        if not isinstance(hw.b, ClockGuard):  # every access to the tile goes through the clock guard
            hw.b = ClockGuard(hw.b, (hw.x, hw.y))
        self.hw = hw
        self.base = base
        self.res = base + A.L2CPU_OFF_RESIDENT
        self.log = log or (lambda *a, **k: None)
        self.mhz = mhz
        self.mbreq = None
        self.region_size = region_size
        self.policy = None  # the PMP policy locked in this chip epoch (start), None = off

    def make_policy(self, pmp=None):
        """pmp: None = environment default (L2CPU_PMP, on), False = off, True = the default policy for this region,
        or a pmp.Policy for this region."""
        if pmp is None:
            pmp = P.default_enabled()
        if pmp is False:
            return None
        if pmp is True:
            return P.build(self.base, self.region_size)
        if pmp.region != self.base:
            raise L2cpuCtlError(f"PMP policy for region 0x{pmp.region:x}, this tile's region is 0x{self.base:x}")
        return pmp

    # ---- low level ----
    def r32(self, pa):
        return self.hw.pa_read32(pa)

    def w32(self, pa, v):
        self.hw.pa_write32(pa, v)

    def r64(self, pa):
        return self.hw.pa_read64(pa)

    def w64(self, pa, v):
        self.hw.pa_write64(pa, v)

    def trigger(self):
        return self.hw.periph_read32(A.L2CPU_RNMI_TRIGGER)

    def set_trigger(self, v):
        self.hw.periph_write32(A.L2CPU_RNMI_TRIGGER, v)
        return self.trigger()

    def handler_addrs(self):
        out = []
        for h in range(4):
            a = A.L2CPU_RNMI_HANDLER + 16 * h
            out.append(tuple(self.hw.periph_read32(a + o) | (self.hw.periph_read32(a + o + 4) << 32) for o in (0, 8)))
        return out

    def record(self, h):
        d = self.hw.pa_read(self.res + A.L2CPU_RES_REC + A.L2CPU_REC_SIZE * h, A.L2CPU_REC_PMP + 4)
        v = struct.unpack_from("<4I7Q", d, 0)
        rec = dict(zip(REC_FIELDS, v))
        rec["park_count"] = struct.unpack_from("<I", d, A.L2CPU_REC_PARK_COUNT)[0]
        rec["pmp"] = struct.unpack_from("<I", d, A.L2CPU_REC_PMP)[0]
        rec["pmp_name"] = PMP_STATE.get(
            rec["pmp"], f"ERROR entry {rec['pmp'] & 0xFF}" if rec["pmp"] >> 8 else rec["pmp"]
        )
        rec["state_name"] = STATE.get(rec["state"], rec["state"])
        rec["kind_name"] = KIND.get(rec["kind"], rec["kind"])
        return rec

    def records(self):
        return [self.record(h) for h in range(4)]

    def ensure_clock(self):
        """Re-apply the target L2CPU clock when a power-state change reprogrammed PLL4 (ARC resets it to 800 MHz)."""
        c1, c5 = self.hw.read_pll()
        mhz = self.hw.pll_mhz(c1, c5)
        if abs(mhz - self.mhz) > 1:
            self.log(f"l2cpu ctl: L2CPU clock {mhz:.0f} MHz, re-applying {self.mhz}")
            self.hw.set_l2cpu_pll(self.mhz)
        return mhz

    # ---- resident page ----
    def write_resident(self, entry_pa, blob=None, policy=None):
        blob = blob if blob is not None else open(RESIDENT_BLOB, "rb").read()
        assert len(blob) == A.L2CPU_RES_SIZE, len(blob)
        b = bytearray(blob)
        if policy is not None:
            t = policy.table()
            b[A.L2CPU_RES_PMP : A.L2CPU_RES_PMP + len(t)] = t
        struct.pack_into("<Q", b, A.L2CPU_RES_ENTRY, entry_pa)
        struct.pack_into("<II", b, A.L2CPU_RES_BOOT_MODE, A.L2CPU_BOOT_COLD, 1)
        struct.pack_into("<I", b, A.L2CPU_RES_GO_EPOCH, 0)
        struct.pack_into("<I", b, A.L2CPU_RES_RNMI_MODE, A.L2CPU_RNMI_PARK)
        self.hw.pa_write(self.res, bytes(b))
        if self.hw.pa_read(self.res, len(b)) != bytes(b):
            raise L2cpuCtlError("resident page read-back mismatch")

    def write_rnmi_handlers(self):
        for h in range(4):
            a = A.L2CPU_RNMI_HANDLER + 16 * h
            for o, ent in ((0, A.L2CPU_RES_RNMI_ENTRY), (8, A.L2CPU_RES_RNMI_EXC)):
                pa = self.res + ent + A.L2CPU_RES_STRIDE * h
                self.hw.periph_write32(a + o, pa & 0xFFFFFFFF)
                self.hw.periph_write32(a + o + 4, pa >> 32)
        got = self.handler_addrs()
        want = [
            (
                self.res + A.L2CPU_RES_RNMI_ENTRY + A.L2CPU_RES_STRIDE * h,
                self.res + A.L2CPU_RES_RNMI_EXC + A.L2CPU_RES_STRIDE * h,
            )
            for h in range(4)
        ]
        return got, want

    # ---- generic firmware status (header block, mailbox, log) ----
    def fw_status(self):
        return self.r32(self.base + A.L2CPU_OFF_FW_STATUS)

    def boot_epoch(self):
        return self.r32(self.base + A.L2CPU_OFF_BOOT_EPOCH)

    def ready(self, epoch=None):
        ok = self.fw_status() == A.L2CPU_FW_STATUS_READY and self.r32(self.base + A.L2CPU_OFF_MAGIC) == A.L2CPU_MAGIC
        return ok and (epoch is None or self.boot_epoch() == epoch)

    def wait_ready(self, epoch=None, timeout=5.0):
        t0 = time.time()
        while not self.ready(epoch):
            if time.time() - t0 > timeout:
                raise L2cpuCtlError(
                    f"not READY (epoch {epoch}): status {self.fw_status()} epoch {self.boot_epoch()}; "
                    f"records {self.records()}"
                )
            time.sleep(0.0005)
        self.mbreq = self.r32(self.base + A.L2CPU_OFF_MB_ACK)
        return time.time() - t0

    def heartbeats(self):
        return [self.r64(self.base + A.L2CPU_OFF_HEARTBEAT + 64 * h) for h in range(4)]

    def mb(self, cmd, *args, timeout=1.0, doorbell=True):
        if self.mbreq is None:
            self.mbreq = self.r32(self.base + A.L2CPU_OFF_MB_ACK)
        a = list(args) + [0] * (7 - len(args))
        self.hw.pa_write(self.base + A.L2CPU_OFF_MB_CMD, struct.pack("<II7Q", cmd, 0, *[x & (2**64 - 1) for x in a]))
        self.mbreq = (self.mbreq + 1) & 0xFFFFFFFF
        self.w32(self.base + A.L2CPU_OFF_MB_REQ, self.mbreq)
        if doorbell:
            self.hw.doorbell(0x4D420000 | (self.mbreq & 0xFFFF))
        t0 = time.time()
        while self.r32(self.base + A.L2CPU_OFF_MB_ACK) != self.mbreq:
            if time.time() - t0 > timeout:
                raise L2cpuCtlError(f"mailbox cmd {cmd}: no ack")
        st, _, *rep = struct.unpack("<II7Q", self.hw.pa_read(self.base + A.L2CPU_OFF_MB_STATUS, 64))
        return st, rep

    def error(self):
        code, hart, ctx, _, arg = struct.unpack("<IIIIQ", self.hw.pa_read(self.base + A.L2CPU_OFF_ERROR_CODE, 24))
        return dict(code=code, hart=hart, context=ctx, arg=arg) if code else None

    def hart_state(self, h):
        d = self.hw.pa_read(self.base + A.L2CPU_OFF_HART_STATE + 64 * h, 48)
        st, err, mc, mepc, mtval, tc, lw, ea = struct.unpack("<IIQQQIIQ", d)
        return dict(status=st, error=err, mcause=mc, mepc=mepc, mtval=mtval, trap_count=tc, last_work=lw, error_arg=ea)

    def inject(self, hart, kind=0):
        """Test hook (fault injection): `hart` executes an illegal instruction (kind 0), or L2CPU_INJECT_SPIN /
        L2CPU_INJECT_WFIPARK (a hart only an RNMI can reach). Other kinds are refused by the firmware.
        Raises L2cpuCtlError on any non-OK mailbox status."""
        st, rep = self.mb(A.L2CPU_MB_INJECT, hart, kind)
        if st != A.L2CPU_MB_OK:
            raise L2cpuCtlError(f"inject hart {hart} kind {kind}: mailbox status {st}")
        return st, rep

    def log_text(self, since=0):
        wr = self.r64(self.base + A.L2CPU_OFF_LOG_WR)
        start = max(since, wr - A.L2CPU_LOG_DATA_SIZE)
        out = bytearray()
        pos = start
        while pos < wr:
            o = pos % A.L2CPU_LOG_DATA_SIZE
            n = min(wr - pos, A.L2CPU_LOG_DATA_SIZE - o)
            out += self.hw.pa_read(self.base + A.L2CPU_OFF_LOG_DATA + o, n)
            pos += n
        return out.decode("latin1"), wr

    # ---- API ----
    def _pre_release(self, entry, probe, policy=None):
        def pre(hw):
            probe["trigger_reset_value"] = self.trigger()
            probe["handlers_reset_value"] = self.handler_addrs()
            self.write_resident(entry, policy=policy)
            if policy is not None:  # boot record flag (bringup.prepare wrote 0 there): every hart applies the table
                hw.write_scratch_u64(A.L2CPU_BOOT_PMP, A.L2CPU_PMP_MAGIC)
            probe["pmp_entries"] = len(policy.entries) if policy is not None else 0
            got, want = self.write_rnmi_handlers()
            probe["handlers_written"] = got
            probe["handlers_ok"] = got == want
            if not probe["handlers_ok"]:
                raise L2cpuCtlError(f"RNMI handler addresses read back {got} != {want}")
            return probe

        return pre

    def start(self, image: bytes, slot=A.L2CPU_SLOT_A, ready_timeout=5.0, pmp=None):
        """Cold bring-up in a fresh chip epoch (tile not yet released): image, resident page (with the PMP table),
        RNMI handler addresses, boot record PMP flag, release. pmp: see make_policy (default on). Returns a dict incl.
        the reset values of the RNMI registers seen before they were written."""
        from .bringup import bringup

        entry = self.base + slot
        probe = {}
        policy = self.make_policy(pmp)
        pre = self._pre_release(entry, probe, policy)
        info = bringup(
            image,
            entry,
            hw=self.hw,
            region_pa=self.base,
            high_mhz=self.mhz,
            log=self.log,
            pre_release=pre,
            ready=lambda hw: self.ready(1),
            ready_timeout=ready_timeout,
        )
        self.mbreq = self.r32(self.base + A.L2CPU_OFF_MB_ACK)
        self.policy = policy
        self.check_pmp(A.L2CPU_PMP_STATE_APPLIED if policy is not None else A.L2CPU_PMP_STATE_OFF)
        info["probe"] = probe
        return info

    def locked_table(self):
        """The PMP table the harts locked in this chip epoch (boot record flag set), as bytes; None = policy off."""
        if self.hw.read_scratch()[A.L2CPU_BOOT_PMP : A.L2CPU_BOOT_PMP + 4] != struct.pack("<I", A.L2CPU_PMP_MAGIC):
            return None
        n = len(P.Policy(0, 0, (), False, ()).table())
        return self.hw.pa_read(self.res + A.L2CPU_RES_PMP, n)

    def check_pmp(self, want):
        """After READY: every hart reports the PMP state `want` (APPLIED after start, REUSED after a restart); with the
        policy off, OFF. Raises otherwise."""
        recs = self.records()
        got = [r["pmp"] for r in recs]
        if any(g != want for g in got):
            raise L2cpuCtlError(f"PMP state {[r['pmp_name'] for r in recs]}, expected {PMP_STATE[want]} on every hart")
        return got

    def check_restart_policy(self, image, entry, pmp=None):
        """Restart rule (README "PMP policy"): the locked set stays until the next chip reset, so a restart keeps the
        region, its size, the TLB windows and the policy; the image must lie in one of the region's image slots.
        Returns the locked table (None = policy off)."""
        locked = self.locked_table()
        if pmp is not None:
            want = self.make_policy(pmp)
            if (want.table() if want is not None else None) != locked:
                raise L2cpuCtlError(
                    "PMP policy is locked for this chip epoch: a restart cannot change the region, its size, the TLB "
                    "windows or turn the policy on/off (chip reset + start with the new policy instead)"
                )
        if self.policy is not None and locked != self.policy.table():
            raise L2cpuCtlError("resident PMP table or boot record flag differs from the policy locked at start")
        if entry not in (self.base + A.L2CPU_SLOT_A, self.base + A.L2CPU_SLOT_B):
            raise L2cpuCtlError(f"entry 0x{entry:x} is not an image slot of region 0x{self.base:x}")
        if image is not None and len(image) > A.L2CPU_SLOT_SIZE:
            raise L2cpuCtlError("image larger than a slot")
        return locked

    def stop(self, timeout_l1=0.05, timeout_l2=0.1, force=True):
        """Park every hart in the resident page. L1 (mailbox PARK) first if the firmware is READY, then L2 (RNMI)
        for the harts that did not park. Returns (records, levels_used)."""
        self.hw.assert_clock_on()
        used = []
        recs = self.records()
        if all(r["state"] == A.L2CPU_STATE_PARKED for r in recs):
            return recs, used
        if self.ready():
            try:
                st, _ = self.mb(A.L2CPU_MB_PARK, timeout=0.2)
                used.append("L1")
                t0 = time.time()
                while time.time() - t0 < timeout_l1:
                    if all(self.record(h)["state"] == A.L2CPU_STATE_PARKED for h in range(4)):
                        break
            except L2cpuCtlError:
                pass
        mask = sum(1 << h for h in range(4) if self.record(h)["state"] != A.L2CPU_STATE_PARKED)
        if mask and force:
            used.append(f"L2(mask 0x{mask:x})")
            self.rnmi(mask, A.L2CPU_RNMI_PARK, timeout=timeout_l2)
        recs = self.records()
        bad = [h for h in range(4) if recs[h]["state"] != A.L2CPU_STATE_PARKED]
        if bad:
            raise L2cpuCtlError(f"harts {bad} not parked (chip reset needed): {recs}")
        return recs, used

    def rnmi(self, mask, mode=None, timeout=0.1):
        """Raise RNMI on the harts in `mask` (level: set, wait for the handler, clear). mode: L2CPU_RNMI_*."""
        if mode is not None:
            self.w32(self.res + A.L2CPU_RES_RNMI_MODE, mode)
            if self.r32(self.res + A.L2CPU_RES_RNMI_MODE) != mode:  # order the Memory Port write before the trigger
                raise L2cpuCtlError("rnmi_mode read-back")
        before = [self.record(h)["rnmi_count"] for h in range(4)]
        t = self.trigger()
        self.set_trigger(t | mask)
        t0 = time.time()
        while True:
            done = all(self.record(h)["rnmi_count"] != before[h] for h in range(4) if mask & (1 << h))
            if done or time.time() - t0 > timeout:
                break
        self.set_trigger(self.trigger() & ~mask)
        if mode == A.L2CPU_RNMI_PARK:
            t0 = time.time()
            while time.time() - t0 < timeout:
                if all(self.record(h)["state"] == A.L2CPU_STATE_PARKED for h in range(4) if mask & (1 << h)):
                    break
        return [self.record(h)["rnmi_count"] - before[h] for h in range(4)]

    def restart(self, image: bytes | None = None, warm=True, slot=None, timeout=5.0, pmp=None):
        """stop -> (load image) -> go. Never touches L2CPU_RESET. Returns timing and the stop records. pmp: the policy
        the caller expects; a restart refuses one that differs from the policy locked at start."""
        t0 = time.time()
        entry = self.base + slot if slot is not None else self.r64(self.res + A.L2CPU_RES_ENTRY)
        locked = self.check_restart_policy(image, entry, pmp)
        recs, used = self.stop()
        t1 = time.time()
        if image is not None:
            self.hw.load_image(image, entry)
        ep = self.r32(self.res + A.L2CPU_RES_BOOT_EPOCH) + 1
        self.w32(self.base + A.L2CPU_OFF_FW_STATUS, 0)
        self.w64(self.res + A.L2CPU_RES_ENTRY, entry)
        self.w32(self.res + A.L2CPU_RES_BOOT_MODE, A.L2CPU_BOOT_WARM if warm else A.L2CPU_BOOT_COLD)
        self.w32(self.res + A.L2CPU_RES_BOOT_EPOCH, ep)
        if self.r32(self.res + A.L2CPU_RES_BOOT_EPOCH) != ep:
            raise L2cpuCtlError("boot epoch read-back")
        self.w32(self.res + A.L2CPU_RES_GO_EPOCH, self.r32(self.res + A.L2CPU_RES_GO_EPOCH) + 1)  # last
        t2 = time.time()
        self.wait_ready(ep, timeout)
        t3 = time.time()
        self.check_pmp(A.L2CPU_PMP_STATE_REUSED if locked is not None else A.L2CPU_PMP_STATE_OFF)
        return dict(
            epoch=ep,
            entry=entry,
            levels=used,
            stop_records=recs,
            t_stop=t1 - t0,
            t_load=t2 - t1,
            t_ready=t3 - t2,
            t_total=t3 - t0,
        )

    def status(self):
        c1, c5 = self.hw.read_pll()
        r = self.hw.read_l2cpu_reset()
        return dict(
            released=bool((r >> (4 + self.hw.tile)) & 1),
            clock_mhz=self.hw.pll_mhz(c1, c5),
            hart_status=self.hw.hart_status(),
            trigger=self.trigger(),
            fw_status=self.fw_status(),
            boot_epoch=self.boot_epoch(),
            heartbeats=self.heartbeats(),
            records=self.records(),
            guard_checks=getattr(self.hw.b, "checks", None),
        )

    def is_alive(self, window=0.01):
        recs = self.records()
        a = self.heartbeats()
        time.sleep(window)
        b = self.heartbeats()
        return all(b[h] > a[h] for h in range(4) if recs[h]["state"] != A.L2CPU_STATE_PARKED)


def start_tiles(ctls, image, slot=A.L2CPU_SLOT_A, ready_timeout=5.0, mhz=None, log=print, pmp=None):
    """Cold bring-up of several tiles of one chip in a fresh chip epoch: per tile (its own L2cpuCtl, hw.tile and
    region) image, resident page and RNMI handler addresses, then ONE L2CPU_RESET write releasing all of them, then
    each tile's READY. image: bytes for every tile, or {tile: bytes}. pmp: as L2cpuCtl.make_policy, for every tile
    (each gets its own policy for its own region). Returns one info dict per ctl."""
    from .bringup import bringup_tiles

    specs, probes, policies = [], [], []
    for c in ctls:
        probe = {}
        entry = c.base + slot
        policies.append(c.make_policy(pmp))
        img = image[c.hw.tile] if isinstance(image, dict) else image
        specs.append(
            dict(
                hw=c.hw,
                image=img,
                load_pa=entry,
                region_pa=c.base,
                pre_release=c._pre_release(entry, probe, policies[-1]),
                ready=lambda hw, c=c: c.ready(1),
            )
        )
        probes.append(probe)
    infos = bringup_tiles(specs, high_mhz=mhz or ctls[0].mhz, ready_timeout=ready_timeout, log=log)
    for c, info, probe, pol in zip(ctls, infos, probes, policies):
        c.mbreq = c.r32(c.base + A.L2CPU_OFF_MB_ACK)
        c.policy = pol
        c.check_pmp(A.L2CPU_PMP_STATE_APPLIED if pol is not None else A.L2CPU_PMP_STATE_OFF)
        info["probe"] = probe
        info["tile"] = c.hw.tile
    return infos
