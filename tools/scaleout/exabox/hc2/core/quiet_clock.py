#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""quiet_clock.py: the clock check at every quiet run.sh takes.

Every quiet (C4's cold half and its open-mesh idle, the after-sustained step's quiet, the sustained stage's lead quiet and
the quiet after each chunk) first asks every chip above idle for its idle clock (AICLK_GO_LONG_IDLE, ARC 0x54, as
ub0_repro.py's end-of-run go_long_idle does) and polls every chip's tt_aiclk until all read <= 800 MHz: the quiet counts
only from there. The wait is capped at 5 min (healthy chips: under 1 s, go_long_idle's 0.34-0.39 s for
32 chips). A chip still above 800 MHz then is one QUIET FAIL line (a finding) naming it by BDF and tray 1-4 with its last
reading; the quiet and the run go on without it, and later quiets of the same process do not wait for it again.
After a quiet taken with the mesh open, busy() asks every chip for its busy clock again (AICLK_GO_BUSY, ARC 0x52, what
tt-metal's mesh open sends), so the next load runs at full clock, and logs every chip's clock at the load start.
Nothing here reads or changes a power limit: only the two clock-state messages and sysfs clock reads.
"""
import os
import threading
import time

IDLE_MHZ = 800                     # cold idle AICLK (ub0_repro.IDLE_MHZ)
LOST_AICLK = 100000                # tt_aiclk reads 0xFFFFFFFF (or fails) once a chip is gone
MSG_GO_BUSY, MSG_GO_LONG_IDLE = 0x52, 0x54
CAP_S = float(os.environ.get("TRAY_QUIET_CLK_CAP_S") or 300)   # 5 min, then a FINDING
RESEND_S = 30.0                    # a chip still above idle gets the idle request again this often
SAY_S = 30.0                       # a progress line naming the chips still above idle (the broker reaps 300 s of silence)
POLL_S = 0.25
BUSY_WAIT_S = 5.0                  # after GO_BUSY: wait at most this long for the clocks to rise, then log them
SEND_TIMEOUT_S = 60.0
BUS_TO_TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}   # PCI bus & 0xF0 -> tray 1-4 (ub0_repro.BUS_TO_TRAY)


def tray_of(pci):
    try:
        return BUS_TO_TRAY.get(int(pci.split(":")[-2], 16) & 0xF0)
    except (IndexError, ValueError, AttributeError):
        return None


def sysfs_aiclk(cid):
    try:
        with open(os.path.join(os.environ.get("UB0_SYS_ROOT", "/sys"), "class", "tenstorrent",
                               f"tenstorrent!{cid}", "tt_aiclk")) as f:
            return int(f.read().strip() or "0")
    except (OSError, ValueError):
        return None


def arc_send(cids, code, timeout_s=SEND_TIMEOUT_S):
    """ARC message `code` (no args) to each chip, each through its own short-lived tt-umd device (go_long_idle's way),
    under a timeout: a chip that stops answering never blocks the quiet. Returns dict(sent, failed[, error])."""
    rec = dict(sent=0, failed=[])

    def go():
        import tt_umd
        for c in cids:
            try:
                d = tt_umd.TTDevice.create(c)
                d.init_tt_device()
                rc = d.arc_msg(code, True, [], 1000)
                del d
                if rc[0] == 0:
                    rec["sent"] += 1
                else:
                    rec["failed"].append(dict(chip=c, error=f"exit code {rc[0]:#x}"))
            except Exception as e:  # noqa: BLE001
                rec["failed"].append(dict(chip=c, error=f"{type(e).__name__}: {str(e)[:120]}"))

    th = threading.Thread(target=go, name="quiet-clock-arc", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        rec["error"] = f"timeout: ARC {code:#x} did not finish within {timeout_s:g} s"
    return rec


def alive(mhz):
    return mhz is not None and 0 < mhz < LOST_AICLK


def chip_list(cs):
    out = []
    for c in sorted(cs):
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def grouped(clocks):
    """{chip: MHz} -> '1350 MHz chips 0-31' or '800 MHz chips 0-3; 1350 MHz chips 4-31' (ub0_repro.clocks_line's shape)."""
    by = {}
    for c, v in clocks.items():
        by.setdefault(f"{v} MHz" if alive(v) else "lost", []).append(c)
    return "; ".join(f"{k} chips {chip_list(cs)}" for k, cs in sorted(by.items(), key=lambda kv: min(kv[1])))


class QuietClock:
    """ids: {chip id: PCI address}. read(chip) -> MHz or None; send(chips, code) -> dict; say(line) logs one line;
    tick(what), if given, is called on each poll (a heartbeat for the stage's watchdog). Each settle() is one dict in
    self.log (the result JSON / rows keep it)."""

    def __init__(self, ids, read=sysfs_aiclk, send=arc_send, say=print, tick=None, cap_s=CAP_S,
                 clock=time.monotonic, sleep=time.sleep):
        self.ids, self.read, self.send, self.say, self.tick = dict(ids), read, send, say, tick
        self.cap_s, self.clock, self.sleep = cap_s, clock, sleep
        self.stuck = {}   # chip -> its last MHz: named in a QUIET FAIL, not waited for again
        self.log = []

    def name(self, c, mhz=None):
        pci = self.ids.get(c)
        return f"{pci} (tray {tray_of(pci) or '?'}) chip {c}" + (f" {mhz} MHz" if mhz is not None else "")

    def clocks(self):
        return {c: self.read(c) for c in self.ids}

    def hot(self, clocks):
        return sorted(c for c, v in clocks.items() if alive(v) and v > IDLE_MHZ and c not in self.stuck)

    def settle(self, label):
        """Idle request to every chip above idle, then poll until every chip reads <= IDLE_MHZ (capped at cap_s).
        Returns the record; its 'settle_s' is the time to idle clock, from which the caller counts the quiet."""
        t0 = self.clock()
        clocks = self.clocks()
        hot = self.hot(clocks)
        rec = dict(label=label, chips=len(self.ids), requested=len(hot), cap_s=self.cap_s)
        if hot:
            r = self.send(hot, MSG_GO_LONG_IDLE)
            if r.get("failed") or r.get("error"):
                rec["send"] = r
        last_send = last_say = t0
        while hot:
            clocks = self.clocks()
            hot = self.hot(clocks)
            now = self.clock()
            if not hot or now - t0 >= self.cap_s:
                break
            if self.tick:
                self.tick(f"quiet clock {label}")
            if now - last_send >= RESEND_S:
                last_send = now
                self.send(hot, MSG_GO_LONG_IDLE)
            if now - last_say >= SAY_S:
                last_say = now
                self.say(f"QUIET CLOCK {label}: {now - t0:.0f} s, still above {IDLE_MHZ} MHz: "
                         + ", ".join(self.name(c, clocks[c]) for c in hot))
            self.sleep(POLL_S)
        el = self.clock() - t0
        never = {c: clocks[c] for c in hot}
        lost = sorted(c for c, v in clocks.items() if not alive(v))
        earlier = sorted(c for c in self.stuck if alive(clocks.get(c)) and clocks[c] > IDLE_MHZ)
        rec.update(settle_s=round(el, 2), at_idle=sum(1 for v in clocks.values() if alive(v) and v <= IDLE_MHZ),
                   never=[dict(chip=c, pci=self.ids.get(c), tray=tray_of(self.ids.get(c)), mhz=v) for c, v in never.items()],
                   still_above_named_earlier=earlier, lost=lost)
        self.stuck.update(never)
        self.say(f"QUIET CLOCK {label}: {rec['at_idle']}/{len(self.ids)} chips at <= {IDLE_MHZ} MHz after {el:.1f} s"
                 + (f"; lost {chip_list(lost)}" if lost else "")
                 + (f"; still above, named in an earlier quiet: {', '.join(self.name(c, clocks[c]) for c in earlier)}"
                    if earlier else "") + "; quiet counts from here")
        if never:
            self.say(fail_line(label, [(self.ids.get(c), c, v) for c, v in never.items()], el, self.cap_s))
        self.log.append(rec)
        return rec

    def busy(self, label):
        """After a quiet with the mesh open: the busy request to every chip still on the bus (the next load runs at
        full clock), a short wait for the clocks to rise, then every chip's clock at the load start, logged."""
        on = [c for c, v in self.clocks().items() if alive(v)]
        r = self.send(on, MSG_GO_BUSY) if on else dict(sent=0, failed=[])
        t0 = self.clock()
        while True:
            clocks = self.clocks()
            if all(not alive(v) or v > IDLE_MHZ for v in clocks.values()) or self.clock() - t0 >= BUSY_WAIT_S:
                break
            self.sleep(POLL_S)
        self.say(f"AICLK at load start ({label}): {grouped(clocks)}"
                 + (f"; busy request: {r}" if r.get("failed") or r.get("error") else ""))
        if self.log and self.log[-1].get("label") == label:
            self.log[-1]["load_start_aiclk"] = clocks
        return clocks


def fail_line(label, never, waited_s, cap_s=CAP_S):
    """never: [(pci, chip, MHz)]. The finding: every chip by BDF and tray 1-4 with its last reading."""
    return (f"QUIET FAIL {label}: chip(s) not at idle clock (<= {IDLE_MHZ} MHz) after {waited_s:.0f} s (cap "
            f"{cap_s:.0f} s): " + ", ".join(f"{p} (tray {tray_of(p) or '?'}) chip {c} at {v} MHz" for p, c, v in never)
            + "; the quiet went on without them")


def kmd_pci():
    """/dev/tenstorrent id -> PCI address (sysfs, read-only)."""
    import glob
    out = {}
    for d in glob.glob(os.path.join(os.environ.get("UB0_SYS_ROOT", "/sys"), "class", "tenstorrent", "tenstorrent!*")):
        out[int(d.rsplit("!", 1)[1])] = os.path.basename(os.path.realpath(os.path.join(d, "device")))
    return out


if __name__ == "__main__":
    # quiet_clock.py settle <label>: the clock check on every chip of this host (run.sh's idle_clocks, before each try)
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "settle":
        sys.exit("usage: quiet_clock.py settle <label>")
    QuietClock(kmd_pci(), say=lambda s: print(s, flush=True)).settle(" ".join(sys.argv[2:]))
