#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole galaxy whole-tray drop repro ("ub0"): one fresh process opens the 4x8 mesh, uploads one bf16 weight
(8192 x 16384, 256 MiB per chip, replicated), idles, then runs a matmul burst on all 32 chips at once.

On a weak tray the first matmul of a COLD start trips the tray: all 8 chips of one tray leave PCIe within ~1-20 s.
A sysfs watchdog logs the first chip(s) lost with PCI address and tray, then exits 3 at once
(closing the device after a drop only hangs). Needs only tt-metal's python_env (ttnn, tt_umd) and sysfs.

Before any load a preflight checks that all 32 chips of the 4 trays are enumerated and answer, and refuses (exit 2)
otherwise: it never loads a galaxy with a tray missing. It logs KMD, firmware and BIOS versions
and how cold the box is (a trip needs a cold start: the first open after a chassis power cycle, or >= 30 min with
no process on the device), and warns loudly when the cold state is not confirmed.

The default, --load ub0: set_fabric_config(FABRIC_1D_RING) before the open, so the open starts the fabric
routers on the ethernet cores and they stay up for the whole run, but no CCL op is ever issued: the burst is bf16
matmuls only (2048x8192 @ 8192x16384, replicated, all 32 chips). It is "fabric up and idle + matmul".
--load matmul: the same upload, 60 s idle and 30 s burst with NO fabric (no fabric config, no routers started),
for a box whose inter-chip cabling is broken. The result JSON's "traffic" block says what ran.
--load allgather: the same ring fabric, then a burst of ttnn.all_gather CCL ops instead of matmuls (AG_VARIANTS
below: the heaviest gathers the 4x8 mesh takes, back to back). Each variant runs once before the idle (its kernel
compile, and any halving), so the burst after the idle is a sudden full load with nothing left to compile. A CCL op
that does not finish within --ccl-op-timeout-s with every chip still on PCIe ends the run as CCL_HANG (exit 4),
apart from a tray drop (exit 3).

Every step prints  STRESS|<UTC ISO time>|message.  Every exit writes the --result JSON first.
"""
import argparse
import csv
import faulthandler
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

EXIT_NO_TRIP, EXIT_ERROR, EXIT_REFUSED, EXIT_TRIP, EXIT_CCL_HANG = 0, 1, 2, 3, 4
EXIT_HELP = """exit codes:
  0  NO TRIP: the run completed and no chip left the bus
  1  ERROR: script or environment error, timeout of a phase, or interrupted (SIGINT/SIGTERM); no chip lost
  2  PREFLIGHT REFUSED: a chip or tray was missing (or would not answer) before any load; nothing was loaded
  3  TRIP: one or more chips left the bus (the result JSON names the first lost chip, its PCI address and tray)
  4  CCL_HANG (--load allgather): a CCL op hung past its limit with every chip still on the bus; no tray drop

The verdict, the first lost chip, timings, a clock/power summary and versions go to the --result JSON on every exit."""

BUS_TO_TRAY = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}   # PCI bus & 0xF0 -> tray; BMC sensor UBB<tray-1>
TRAYS, CHIPS_PER_TRAY = 4, 8
CPLD_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}      # tray -> in-band tray CPLD I2C bus (addr 0x80); UBB0 0x11 and
#                                                      UBB1 0x13 seen in drops, UBB2 0x0D, UBB3 0x0F
CPLD_REGS = ("1e", "19")                             # 0x1E alive byte (ff healthy, 00 dead), 0x19 VR power good (ff)
LOSS_CAPTURE_MAX_S = 90                              # on_loss captures (lspci, CPLD, SEL), each bounded by a timeout
TEARDOWN_MAX_S = int(os.environ.get("UB0_TEARDOWN_MAX_S") or 60)   # clean-exit tt-metal teardown deadline
LOST_AICLK = 100000                                  # tt_aiclk reads 0xFFFFFFFF (or fails) once a chip is gone
IDLE_MHZ = 800                                       # cold idle AICLK; > 800 means the device ran recently
COLD_S = 1800                                        # 30 min with no process on the device
MSG_SET_ASIC_HOST_FMAX = 0x23                        # arg0 = MHz ceiling, arg1 bit0 = restore fw default
# no args; at a normal exit tt-metal's atexit ~Cluster -> UMD LocalChip::close_device -> BlackholeDeviceFirmware::
MSG_AICLK_GO_LONG_IDLE = 0x54                        # set_clock_state(IDLE) sends it
# per-load settings: the 4x8 mesh, the 2048x8192 @ 8192x16384 bf16 matmul, a 30 s burst; matmul has no fabric
LOADS = {"ub0": dict(m=2048, k=8192, n=16384, topo="ring", mesh="4x8"),
         "matmul": dict(m=2048, k=8192, n=16384, topo="none", mesh="4x8"),
         "allgather": dict(m=2048, k=8192, n=16384, topo="ring", mesh="4x8")}
# --load allgather: (name, cluster_axis, gather dim, per-chip input shape), run in turn, back to back. Why these are
# the heaviest: each output is 2^30 bf16 elements = 2 GiB per chip. The cap is on the element count: 2^30 is half
# of INT32_MAX (2^31 - 1), a margin for 32-bit volume/page maths. 2 GiB is not an int32 limit: as a byte count
# (2^31) it is already one past INT32_MAX. So every op moves 1.75-1.97 GiB into each chip
# over the ethernet fabric. The three cover both mesh axes (8-way rows, 4-way columns) and the full 32-way gather
# across every chip. ttnn refuses cluster_axis=None on a 2D mesh with a 1D fabric, so full_32way is two chained
# gathers on the same ring: 8-way along axis 1, then 4-way along axis 0 (64 MiB -> 512 MiB -> 2 GiB). Live DRAM per
# chip: 832 MiB of inputs + one 2 GiB output and a 512 MiB step (freed after each op), well inside the 32 GiB of a
# Blackhole chip. A variant whose first call runs out of DRAM (no chip lost) is halved, up to twice, then dropped;
# any other first-call error drops it at once. The result JSON's burst.cut says what was cut.
AG_VARIANTS = (("axis1_8way", 1, 3, (1, 1, 8192, 16384)),    # 256 MiB in -> [1,1,8192,131072] 2 GiB out
               ("axis0_4way", 0, 2, (1, 1, 16384, 16384)),   # 512 MiB in -> [1,1,65536,16384] 2 GiB out
               ("full_32way", None, 2, (1, 1, 2048, 16384)))  # 64 MiB in -> axis 1, axis 0 -> [1,1,65536,16384] 2 GiB
AG_WAYS = {1: 8, 0: 4, None: 32}
# sysfs and /dev roots; the offline tests point these at a fake tree (tests/fakesys.py)
SYS = os.environ.get("UB0_SYS_ROOT", "/sys")
DEV = os.environ.get("UB0_DEV_ROOT", "/dev/tenstorrent")


def utc(t=None):
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


def mark(msg):
    line = f"STRESS|{utc()}|{msg}"
    try:
        print(line, flush=True)
    except (OSError, ValueError):  # stdout gone (tee killed): keep going, the result JSON still gets written
        pass
    except RuntimeError:  # a signal handler's mark inside the main thread's print: "reentrant call inside ..."
        try:
            os.write(1, (line + "\n").encode())
        except OSError:
            pass


def tt_class():
    return os.path.join(SYS, "class", "tenstorrent")


def chips():
    """/dev/tenstorrent id -> PCI address."""
    out = {}
    for d in glob.glob(os.path.join(tt_class(), "tenstorrent!*")):
        out[int(d.rsplit("!", 1)[1])] = os.path.basename(os.path.realpath(os.path.join(d, "device")))
    return dict(sorted(out.items()))


def tray_of(pci):
    try:
        return BUS_TO_TRAY.get(int(pci.split(":")[-2], 16) & 0xF0)
    except (IndexError, ValueError, AttributeError):
        return None


def fail_line(lost, total, trays, cls, pci=None):  # BDF + trays 1-4 only, never a UBB number
    return (f"RESULT FAIL lost={len(lost)}/{total} trays={trays} " + (f"pci={pci} " if pci is not None else "")
            + f"chips={lost} loss={cls}")


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def sysfs(cid, name):
    v = read(os.path.join(tt_class(), f"tenstorrent!{cid}", name))
    try:
        return int(v or "0")
    except ValueError:
        return None


def aiclk(cid):
    return sysfs(cid, "tt_aiclk")


def chip_list(cs):
    """[0, 1, 2, 5] -> '0-2,5'"""
    out, cs = [], sorted(cs)
    for c in cs:
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def clocks_line(ids):
    """Every chip's AICLK now, grouped: '1350 MHz chips 0-31', or '800 MHz chips 0-3; 1350 MHz chips 4-31'."""
    by = {}
    for c in ids:
        v = aiclk(c)
        by.setdefault("lost" if v is None or v >= LOST_AICLK else f"{v} MHz", []).append(c)
    return "; ".join(f"{k} chips {chip_list(cs)}" for k, cs in sorted(by.items(), key=lambda kv: min(kv[1])))


def alive(cid, clk):
    return os.path.exists(os.path.join(DEV, str(cid))) and clk is not None and clk < LOST_AICLK


def present(cid):
    """On the bus and the firmware answers: tt_aiclk reads 0xFFFFFFFF (or fails) once the chip is gone."""
    return alive(cid, aiclk(cid))


def hwmon(cid, name, scale):
    for h in glob.glob(os.path.join(tt_class(), f"tenstorrent!{cid}", "device", "hwmon", "hwmon*", name)):
        try:
            return int(read(h)) / scale
        except (TypeError, ValueError):
            return None
    return None


def hwmon_power_w(cid):
    return hwmon(cid, "power1_input", 1e6)


TELEM_COLS = ["utc", "dev", "pci", "tray", "aiclk_mhz", "power_w", "phase"]
TELEM_FULL_COLS = ["current_a", "vcore_mv", "temp_c"]   # --telem-full: appended, so readers of the first 7 still work
LINK_COLS = ["utc", "dev", "pci", "root_port", "heartbeat", "hb_still_s", "rp_speed", "rp_width", "rp_dllla",
             "rp_training", "ep_node", "phase"]


def telem_full(cid):
    """--telem-full extras from the chip's hwmon: current (A), VCORE (mV), ASIC temperature (C); None if unreadable."""
    return [hwmon(cid, "curr1_input", 1e3), hwmon(cid, "in0_input", 1), hwmon(cid, "temp1_input", 1e3)]


def heartbeat(cid):
    """The ARC heartbeat counter (KMD sysfs tt_heartbeat, mode 444); None when unreadable or the chip is gone."""
    v = read(os.path.join(tt_class(), f"tenstorrent!{cid}", "tt_heartbeat"))   # not sysfs(): missing would read 0
    try:
        v = int(v, 0)
    except (TypeError, ValueError):
        return None
    return None if v == 0xFFFFFFFF else v


PCIE_SPEED = {1: "2.5", 2: "5", 3: "8", 4: "16", 5: "32", 6: "64"}   # LnkSta current link speed code -> GT/s


def lnksta(d):
    """Link Status of the PCI function at sysfs dir d (a root port: host side, no chip traffic). From the config space's
    PCIe capability when readable (root reads all of it; others get 64 bytes), so DLLLA and training are known; else from
    the kernel's current_link_speed/width (also a live LnkSta read). dict(speed GT/s, width, dllla, training, src)."""
    cfg = None
    try:
        with open(os.path.join(d, "config"), "rb") as f:
            cfg = f.read(4096)
    except OSError:
        pass
    if cfg and len(cfg) > 0x40:
        ptr, seen = cfg[0x34] & 0xFC, set()
        while 0x40 <= ptr < len(cfg) - 1 and ptr not in seen:
            seen.add(ptr)
            if cfg[ptr] == 0x10 and ptr + 0x14 <= len(cfg):
                v = int.from_bytes(cfg[ptr + 0x12:ptr + 0x14], "little")
                if v == 0xFFFF:
                    break           # the function itself does not answer
                return dict(speed=PCIE_SPEED.get(v & 0xF, f"code{v & 0xF}"), width=(v >> 4) & 0x3F,
                            dllla=(v >> 13) & 1, training=(v >> 11) & 1, src="config")
            ptr = cfg[ptr + 1] & 0xFC
    sp, w = read(os.path.join(d, "current_link_speed")), read(os.path.join(d, "current_link_width"))
    try:
        w = int(w)
    except (TypeError, ValueError):
        w = None
    return dict(speed=(sp or "").split(" ")[0] or None, width=w, dllla=None, training=None, src="sysfs")


def link_down(ls):
    return not ls or ls.get("dllla") == 0 or not ls.get("width") or ls.get("speed") in (None, "Unknown")


# ---------------------------------------------------------------- loss evidence (read-only; no write, no re-power)
def run_cmd(cmd, timeout):
    """stdout of a bounded read-only command, or None if it failed, timed out or is not installed."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def ipmi(tool, args, timeout):
    """One bounded ipmitool call through --ipmitool / --sel-ipmitool ('none': off)."""
    if not tool or tool == "none":
        return None
    return run_cmd(tool.split() + args, timeout)


def cpld_read(tool):
    """Tray CPLD reg 0x1E (alive) and 0x19 (VR power good) of every tray: {tray: {"1e": "ff", "19": "ff"}}; None
    where the read failed. Same in-band reads as run_repro.sh's cpld_dump (raw 0x06 0x52 <bus> 0x80 0x01 <reg>)."""
    out = {}
    for tray, bus in CPLD_BUS.items():
        out[tray] = {}
        for reg in CPLD_REGS:
            s = ipmi(tool, ["raw", "0x06", "0x52", f"{bus:#04x}", "0x80", "0x01", "0x" + reg], 8)
            out[tray][reg] = s.split()[0].lower() if s and s.split() else None
    return out


def sel_tail(tool, n=40):
    """The last n BMC SEL lines (ipmitool sel elist last n), or None if the SEL could not be read."""
    s = ipmi(tool, ["sel", "elist", "last", str(n)], 30)
    return None if s is None else [ln.strip() for ln in s.splitlines() if ln.strip()]


def load_sel_baseline(path):
    """run.sh's once-a-run SEL read (--sel-baseline): its lines, or None when unread or empty (then no loss is a
    CHIP_HANG and old FRB lines never count as new)."""
    try:
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()] or None
    except (OSError, UnicodeDecodeError):
        return None


def sel_new(before, after):
    """SEL lines in after whose record id was not in before (all of after if before was not read)."""
    seen = {ln.split("|")[0].strip() for ln in before or []}
    return [ln for ln in after or [] if ln.split("|")[0].strip() not in seen]


def pci_dir(cid, pci):
    """The chip's PCI directory in the sysfs device tree (via the tenstorrent class link, else /sys/bus/pci)."""
    for link in (os.path.join(tt_class(), f"tenstorrent!{cid}", "device"), os.path.join(SYS, "bus", "pci", "devices", pci)):
        if os.path.exists(link):
            return os.path.realpath(link)
    return None


def aer_counts(d):
    """Non-zero AER counters of one PCI function from sysfs aer_dev_{correctable,nonfatal,fatal}: {"file": {...}}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "aer_dev_*"))) if d else []:
        nz = {}
        for ln in (read(f) or "").splitlines():
            k, _, v = ln.strip().rpartition(" ")
            if k and v.isdigit() and int(v):
                nz[k] = int(v)
        out[os.path.basename(f)] = nz
    return out


def link_capture(cid, pci, lspci):
    """Right after a loss: the lost chip's root port (its parent in the device tree), the port's lspci -vv
    LnkSta/LnkCtl/status lines and the AER counters of the port and the chip."""
    d = pci_dir(cid, pci)
    up = os.path.dirname(d) if d else None
    rp = os.path.basename(up) if up and os.path.basename(up).count(":") == 2 else None
    rec = dict(chip=cid, pci=pci, root_port=rp, aer_root_port=aer_counts(up) if rp else None, aer_chip=aer_counts(d))
    if rp and lspci and lspci != "none":
        out = run_cmd(lspci.split() + ["-vv", "-s", rp], 10)
        rec["lspci_root_port"] = None if out is None else [
            ln.strip() for ln in out.splitlines() if not ln[:1].isspace() or
            any(k in ln for k in ("LnkSta", "LnkCtl", "UESta", "CESta", "DevSta", "Status:"))]
    return rec


def port_dir(rp, known):
    """The sysfs directory of root port rp: as found before the run, else /sys/bus/pci/devices/<rp>."""
    return known.get(rp) or os.path.realpath(os.path.join(SYS, "bus", "pci", "devices", rp))


def aer_rose(now, pre):
    """AER counters of the port that are higher than at the start of the run. A port with no baseline (pre is None)
    is unknown: no rise, so the loss is not a CHIP_HANG."""
    if pre is None:
        return {}
    out = {}
    for f, cnt in (now or {}).items():
        base = (pre or {}).get(f, {})
        up = {k: v for k, v in cnt.items() if v > base.get(k, 0)}
        if up:
            out[f] = up
    return out


def classify_loss(lost, ids, cpld, new_sel, sel_read, mmio, aer_up, lost_ports=None):
    """CHIP_HANG, DROP or UNKNOWN. CHIP_HANG only when every sign of a single-chip
    PCIe hang is there and nothing points at power: a UMD MMIO timeout, AER counters up on every lost chip's root
    port, every tray CPLD alive (each lost tray reads ff/ff on reg 0x1E/0x19, no readable tray reads otherwise), the
    SEL read both before the run and after the loss with no new FRB on a lost chip's UBB, and no whole tray lost.
    DROP only when the evidence shows power or PCIe gone: a whole tray lost, a lost tray's CPLD reg 0x1E/0x19 read
    not ff, or a new FRB on a lost chip's UBB. Everything else (an unread CPLD or SEL, no SEL baseline, a lost chip's
    root port unknown or without an AER rise) is UNKNOWN: a chip that stopped answering is never called a drop for
    want of a BMC read. All three are a loss: exit 3, and the ladder ends. aer_up: {root port: counters that rose};
    lost_ports: {chip: root port or None}."""
    trays = sorted({tray_of(ids[c]) for c in lost}, key=str)
    per_tray = {}
    for c in lost:
        per_tray[tray_of(ids[c])] = per_tray.get(tray_of(ids[c]), 0) + 1
    in_tray = {t: sum(1 for p in ids.values() if tray_of(p) == t) for t in per_tray}
    whole = sorted((t for t, n in per_tray.items() if t is not None and n >= max(in_tray[t], 1)), key=str)
    ubbs = [f"UBB{t - 1}_ASIC" for t in trays if isinstance(t, int)]
    cpld = cpld or {}
    bad = {t: v for t, v in cpld.items() if any(x not in (None, "ff") for x in v.values())}
    frb = [ln for ln in new_sel or [] if "FRB" in ln and any(u in ln for u in ubbs)]
    alive_ff = bool(ubbs) and all(cpld.get(t) and all(x == "ff" for x in cpld[t].values())
                                  for t in trays if isinstance(t, int))
    aer_up = aer_up or {}
    lost_ports = lost_ports if lost_ports is not None else {c: None for c in lost}
    no_aer = sorted(c for c in lost if not aer_up.get(lost_ports.get(c)))
    basis = dict(whole_trays_lost=whole, cpld_not_ff={str(t): v for t, v in bad.items()}, lost_trays_cpld_ff=alive_ff,
                 sel_frb=frb, sel_read=bool(sel_read), mmio_timeout=bool(mmio), root_port_aer_up=aer_up,
                 lost_chips_without_aer_up=no_aer)
    hang = (bool(mmio) and bool(aer_up) and not no_aer and alive_ff and not bad and bool(sel_read) and not frb
            and not whole)
    basis["missing_for_chip_hang"] = [w for w, ok in (("mmio_timeout", bool(mmio)),
                                                      ("root_port_aer_up", bool(aer_up) and not no_aer),
                                                      ("cpld_alive", alive_ff and not bad), ("sel_read", bool(sel_read)),
                                                      ("no_sel_frb", not frb), ("not_whole_tray", not whole)) if not ok]
    lost_bad = sorted((t for t in bad if t in trays), key=str)
    basis["drop_evidence"] = [w for w, ok in (("whole_tray_lost", bool(whole)), ("lost_tray_cpld_not_ff", bool(lost_bad)),
                                              ("sel_frb", bool(frb))) if ok]
    return ("CHIP_HANG" if hang else "DROP" if basis["drop_evidence"] else "UNKNOWN"), basis


# ---------------------------------------------------------------- preflight (read-only: sysfs and /proc)

def preflight(ids):
    """Problems that refuse the run: the box must show 4 trays x 8 chips, every one answering. Returns a list."""
    problems = []
    if len(ids) != TRAYS * CHIPS_PER_TRAY:
        problems.append(f"{len(ids)} chips enumerated in {tt_class()}, expected {TRAYS * CHIPS_PER_TRAY}")
    by_tray = {}
    for c, pci in ids.items():
        by_tray.setdefault(tray_of(pci), []).append(c)
    for t in range(1, TRAYS + 1):
        n = len(by_tray.get(t, []))
        if n != CHIPS_PER_TRAY:
            problems.append(f"tray {t} has {n}/{CHIPS_PER_TRAY} chips enumerated")
    if by_tray.get(None):
        problems.append(f"chips on an unknown PCI bus (not in the tray map): "
                        f"{[(c, ids[c]) for c in by_tray[None]]}")
    gone = [c for c in ids if not present(c)]
    if gone:
        problems.append(f"chips not answering (no /dev node, or tt_aiclk unreadable / 0xFFFFFFFF): {gone} "
                        f"trays {sorted({tray_of(ids[c]) for c in gone}, key=str)}")
    return problems


def holders():
    """pid -> command of processes holding a /dev/tenstorrent node open, and how many /proc entries were unreadable
    (other users' processes, when not root)."""
    found, hidden = {}, 0
    prefix = os.path.realpath(DEV) + "/"
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = int(fd_dir.split("/")[2])
        if pid == os.getpid():
            continue
        try:
            fds = os.listdir(fd_dir)
        except (FileNotFoundError, ProcessLookupError):  # exited during the scan
            continue
        except OSError:
            hidden += 1
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith(prefix):
                found[pid] = (read(f"/proc/{pid}/cmdline") or "").replace("\0", " ")[:120]
                break
    return found, hidden


def uptime_s():
    v = read("/proc/uptime")
    return float(v.split()[0]) if v else None


def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(path, **kw):
    if not path:
        return
    st = load_state(path)
    st.update(kw, host=os.uname().nodename)
    try:
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, path)
    except OSError:
        pass


def cold_state(clocks, holding, up_s, last_use_age_s, watched_s):
    """How cold the box is. clocks: chip -> tt_aiclk now. holding: pid -> cmd. up_s: host uptime. last_use_age_s:
    seconds since this tool last used the device (None: no record). watched_s: seconds every chip was seen at
    <= 800 MHz in a row (--wait-quiet-min). Returns dict(status=cold|likely_cold|unconfirmed|warm, reasons=[...])."""
    warm, notes = [], []
    hot = sorted(c for c, v in clocks.items() if v is not None and IDLE_MHZ < v < LOST_AICLK)
    if holding:
        warm.append(f"processes hold the device now: {holding}")
    if hot:
        warm.append(f"chips {hot} run above the {IDLE_MHZ} MHz idle clock now (the device was used recently)")
    if last_use_age_s is not None and last_use_age_s < COLD_S and (up_s is None or last_use_age_s < up_s):
        warm.append(f"this tool used the device {last_use_age_s / 60:.1f} min ago (< {COLD_S // 60} min)")
    if warm:
        return dict(status="warm", reasons=warm)
    if watched_s is not None and watched_s >= COLD_S:
        return dict(status="cold", reasons=[f"all chips at <= {IDLE_MHZ} MHz for {watched_s / 60:.1f} min in a row"])
    since_boot_only = last_use_age_s is None or (up_s is not None and last_use_age_s >= up_s)
    if up_s is not None and up_s < COLD_S and since_boot_only:
        return dict(status="likely_cold", reasons=[
            f"host up {up_s / 60:.1f} min and no use of the device by this tool since boot: cold if the boot was a "
            "chassis (BMC) power cycle and nothing else opened the device since"])
    if watched_s:
        notes.append(f"all chips at <= {IDLE_MHZ} MHz for only {watched_s / 60:.1f} min")
    else:
        notes.append(f"all chips at <= {IDLE_MHZ} MHz now, but how long they have been quiet is not known")
    notes.append("use --wait-quiet-min 31, or power-cycle the chassis, to get a confirmed cold start")
    return dict(status="unconfirmed", reasons=notes)


def versions(ids):
    v = {"kmd": read(os.path.join(SYS, "module", "tenstorrent", "version")), "kernel": os.uname().release,
         "python": sys.version.split()[0], "host": os.uname().nodename}
    for name in ("bios_vendor", "bios_version", "bios_date", "board_vendor", "board_name", "product_name"):
        v[name] = read(os.path.join(SYS, "class", "dmi", "id", name))
    attrs = set()
    for c in ids:
        attrs.update(os.path.basename(p) for p in glob.glob(os.path.join(tt_class(), f"tenstorrent!{c}", "tt_*_ver")))
    for name in sorted(attrs | {"tt_card_type"}):
        vals = sorted({read(os.path.join(tt_class(), f"tenstorrent!{c}", name)) or "" for c in ids})
        if vals != [""]:
            v[name] = vals
    home = os.environ.get("TT_METAL_HOME")
    v["tt_metal_home"] = home
    if home:
        try:
            r = subprocess.run(["git", "-C", home, "describe", "--always", "--dirty", "--abbrev=40"],
                               capture_output=True, text=True, timeout=10)
            v["tt_metal_commit"] = r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            v["tt_metal_commit"] = None
    return v


# ---------------------------------------------------------------- the run: watchdog, phase timeouts, one exit

def pg_record(pid):
    """Append "pgid starttime" of a start_new_session child to $TRAY_PGID_FILE (run.sh's lib/pgroups.sh reaps it)."""
    f = os.environ.get("TRAY_PGID_FILE")
    if not f:
        return
    try:
        with open(f"/proc/{pid}/stat") as st:
            start = st.read().rsplit(")", 1)[1].split()[19]
        with open(f, "a") as out:
            out.write(f"{pid} {start}\n")
    except (OSError, IndexError):
        pass


def pg_term(pgid, grace_s=5.0):
    """TERM a process group now, KILL it after grace_s (a daemon timer), so it never outlives a preemption."""
    def kill():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
    t = threading.Timer(grace_s, kill)
    t.daemon = True
    t.start()


class Interrupted(BaseException):
    pass


def bursting(phase):
    return phase.partition("@")[0] == "burst"


def sample_into(stats, ids, clocks, phase, burst_started, writer, full=False):
    """One 1-s telemetry sample: a CSV row per chip and the pre_burst/burst/post_burst summary in stats. full: the
    row also gets current, VCORE and temperature (--telem-full)."""
    now = utc()
    cyc = phase.partition("@")[2]   # --cycles: cycle 2 on has pre_burst2 and burst2 (cyc_phase)
    bucket = ("burst" if bursting(phase) else "pre_burst" if cyc else "post_burst" if burst_started else "pre_burst") + cyc
    s = stats.setdefault(bucket, dict(samples=0, aiclk_min=None, aiclk_max=None, chip_power_max_w=None,
                                      box_power_max_w=None, box_power_last_w=None, last_utc=None))
    total, clk_ok = 0.0, []
    for c, pci in ids.items():
        clk = clocks.get(c)
        pw = hwmon_power_w(c)
        if writer:
            writer.writerow([now, c, pci, tray_of(pci), clk, pw, phase] + (telem_full(c) if full else []))
        if clk is not None and clk < LOST_AICLK:
            clk_ok.append(clk)
        if pw is not None:
            total += pw
            s["chip_power_max_w"] = max(s["chip_power_max_w"] or 0.0, round(pw, 1))
    s["samples"] += 1
    if clk_ok:
        s["aiclk_min"] = min(clk_ok + ([s["aiclk_min"]] if s["aiclk_min"] is not None else []))
        s["aiclk_max"] = max(clk_ok + ([s["aiclk_max"]] if s["aiclk_max"] is not None else []))
    s["box_power_last_w"] = round(total, 1)
    s["box_power_max_w"] = max(s["box_power_max_w"] or 0.0, round(total, 1))
    s["last_utc"] = now


def first_loss_line(f):
    return (f"FIRST LOSS at {f['utc']}: chip(s) {f['chips'][:8]}{'...' if len(f['chips']) > 8 else ''} "
            f"pci {f['pci'][:8]}" + (f" (+{f['s_after_burst_begin']}s after burst begin)"
                                     if "s_after_burst_begin" in f else f" (phase {f['phase']})"))


class LinkPoll:
    """--link-poll: every 1 s, each chip's ARC heartbeat (tt_heartbeat) and its root port's LnkSta, one CSV row per
    chip. Gives the order of a hang: heartbeat still (ARC stuck), then the link down. While the run holds a hang
    (hold=True) only the root ports are read (host side): no read reaches a chip until the snapshot is done."""

    def __init__(self, path, ids, ports):
        self.ids, self.ports = ids, ports             # ports: chip -> (root port name, its sysfs dir)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f, lineterminator="\n")
        self.w.writerow(LINK_COLS)
        self.f.flush()
        self.last = {}                                # chip -> (heartbeat value, time it last changed)

    def poll(self, phase, hold=False):
        now, t = utc(), time.time()
        still, down = {}, []
        for c, pci in self.ids.items():
            rp, rpd = self.ports.get(c, (None, None))
            hb, hs = None, None
            if not hold:
                hb = heartbeat(c)
                prev = self.last.get(c)
                if hb is not None and (prev is None or prev[0] != hb):
                    self.last[c] = (hb, t)
                elif prev is not None:
                    hs = round(t - prev[1], 1)
                    still[c] = hs
            ls = lnksta(rpd) if rpd else None
            if rpd and link_down(ls):
                down.append(c)
            ls = ls or {}
            self.w.writerow([now, c, pci, rp, hb, hs, ls.get("speed"), ls.get("width"), ls.get("dllla"),
                             ls.get("training"), int(os.path.exists(os.path.join(DEV, str(c)))), phase])
        self.f.flush()
        return dict(utc=now, hb_still_s=still, link_down=down, hold=hold)

    def close(self):
        self.f.close()


def sentinel_main(ids, poll_s, telem, shm, conn, lspci, ppid, full=False, link=None, ports=None):
    """The chip watchdog and telemetry, in their own process (forked by start_watchdog before torch/ttnn load).
    A device call that holds the GIL blinds the watchdog thread (about 4.8 s inside open_mesh_device);
    this process keeps polling sysfs every --poll-s, writes the --telem rows, prints FIRST LOSS the moment a
    chip goes and captures that chip's root port link and AER at once, then hands the loss to the main process over
    the pipe. It only reads sysfs and runs lspci; it exits when told to or when the main process is gone.
    With --link-poll it also writes the 1 s heartbeat/LnkSta rows (LinkPoll). While shm hold is set (--hold-on-hang,
    the snapshot is running) it reads no chip at all: only the root ports' LnkSta, every 1 s."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)    # the main process owns Ctrl-C / TERM handling
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    f = open(telem, "a", newline="") if telem else None
    w = csv.writer(f, lineterminator="\n") if f else None
    lp = LinkPoll(link, ids, ports or {}) if link else None
    stats, last_t, last_ok, reported = {}, 0.0, None, False
    try:
        while not shm["stop"].value and os.getppid() == ppid:
            if shm["hold"].value:
                if lp and time.time() - last_t >= 1.0:
                    last_t = time.time()
                    conn.send(("link", lp.poll(shm["phase"].value.decode(errors="replace"), hold=True)))
                time.sleep(poll_s)
                continue
            clocks = {c: aiclk(c) for c in ids}
            phase, burst_t0 = shm["phase"].value.decode(errors="replace"), shm["burst"].value or None
            gone = [c for c in ids if not alive(c, clocks[c])]
            if gone and not reported:
                t = time.time()
                first = dict(utc=utc(t), chips=gone, pci=[ids[c] for c in gone], phase=phase)
                if burst_t0:
                    first["s_after_burst_begin"] = round(t - burst_t0, 2)
                mark(first_loss_line(first) + " [sentinel]")
                cap = link_capture(gone[0], ids[gone[0]], lspci)
                conn.send(("loss", dict(epoch=t, chips=gone, phase=phase, last_ok=last_ok, capture=cap)))
                reported = True
            elif not gone:
                last_ok = time.time()
            if time.time() - last_t >= 1.0:
                last_t = time.time()
                sample_into(stats, ids, clocks, phase, burst_t0 is not None, w, full)
                if f:
                    f.flush()
                conn.send(("stats", stats))
                if lp:
                    conn.send(("link", lp.poll(phase)))
            time.sleep(poll_s)
    except (OSError, ValueError, EOFError, BrokenPipeError):
        pass
    finally:
        if f:
            f.close()
        if lp:
            lp.close()
    os._exit(0)


class Run:
    """Owns every exit. The watchdog thread polls the chips every --poll-s, samples clock/power every 1 s and
    enforces the phase deadlines; whichever path ends the run calls finish(), which writes the result JSON and
    leaves with os._exit (no interpreter teardown: it can hang in the device library after a drop). After a clean
    close it first sends GO_LONG_IDLE itself (go_long_idle), then leaves through libc exit() so tt-metal's C++
    teardown runs, bounded by TEARDOWN_MAX_S (teardown_exit)."""

    def __init__(self, a, ids):
        self.a, self.ids = a, ids
        self.root_ports, self.port_dirs = {}, {}   # chip -> root port name, root port -> sysfs dir (before the run)
        self.lock = threading.Lock()
        self.finishing = False
        self.trip_pending = False
        self.abandoned = False            # a hard exit: a device call may still be running (hung phase, signals)
        self.printed = False              # the RESULT line is out
        self.stop = threading.Event()
        self.stop_at = None               # monotonic time of the first SIGINT/SIGTERM
        self.phase, self.deadline = "start", None
        self.phase_t0 = time.monotonic()  # monotonic start of the current phase
        self.op = None                    # (name, monotonic deadline, started) of the CCL op in flight (allgather)
        self.t_start = time.time()
        import multiprocessing
        self.mp = multiprocessing.get_context("fork")       # explicit: the default changes in Python 3.14
        self.shm = dict(phase=self.mp.Array("c", 64, lock=False), burst=self.mp.Value("d", 0.0, lock=False),
                        stop=self.mp.Value("b", 0, lock=False),   # read by the sentinel process
                        hold=self.mp.Value("b", 0, lock=False))   # --hold-on-hang: snapshot running, read no chip
        self.holding = False              # --hold-on-hang: a CCL hang is held and its snapshot is running
        self.link_state = None            # the sentinel's last --link-poll summary (heartbeat still, links down)
        self.sp = self.sconn = None                         # the sentinel process and our end of its pipe
        self._burst_t0 = None
        self.last_ok = None               # time of the last sweep with every chip present
        self.mesh = self.ttnn = None
        self.fmax_set = False
        self.close_ok = self.restore_ok = None   # set by close_quietly: False if the mesh close or fmax restore failed
        self.result = dict(verdict=None, exit_code=None, args={k: v for k, v in vars(a).items()},
                           start_utc=utc(self.t_start), chips={c: dict(pci=p, tray=tray_of(p)) for c, p in ids.items()})
        self.stats = {}
        self.csv_f = self.csv_w = None
        self.wd = None

    # phases ------------------------------------------------------------------------------------------------
    def set_phase(self, name, timeout_s=None):
        with self.lock:
            self.phase, self.phase_t0 = name, time.monotonic()
            self.deadline = self.phase_t0 + timeout_s if timeout_s else None
        self.shm["phase"].value = name.encode()[:63]
        self.result.setdefault("phases", []).append(dict(phase=name, utc=utc(), timeout_s=timeout_s))

    @property
    def burst_t0(self):
        return self._burst_t0

    @burst_t0.setter
    def burst_t0(self, t):
        self._burst_t0 = t
        self.shm["burst"].value = t or 0.0

    def step_mark(self, what):
        """Timestamp a UMD start or fabric step (log line and result["steps"]): a loss is placed against them."""
        self.result.setdefault("steps", []).append(dict(step=what, utc=utc()))
        mark(f"step: {what}")

    # watchdog ----------------------------------------------------------------------------------------------
    def start_watchdog(self):
        if self.a.telem:
            self.csv_f = open(self.a.telem, "w", newline="")
            self.csv_w = csv.writer(self.csv_f, lineterminator="\n")
            self.csv_w.writerow(TELEM_COLS + (TELEM_FULL_COLS if self.a.telem_full else []))
            self.csv_f.flush()
        self.start_sentinel()
        self.wd = threading.Thread(target=self.watch, name="ub0-watchdog", daemon=True)
        self.wd.start()

    def start_sentinel(self):
        """Fork the sentinel process (no torch/ttnn loaded yet). If it cannot start or dies, the watchdog thread polls
        and samples by itself, as before."""
        try:
            rd, wr = self.mp.Pipe(duplex=False)
            self.sp = self.mp.Process(target=sentinel_main, name="ub0-sentinel", daemon=True,
                                      args=(self.ids, self.a.poll_s, self.a.telem, self.shm, wr, self.a.lspci, os.getpid(),
                                            self.a.telem_full, self.a.link_poll or None,
                                            {c: (rp, self.port_dirs.get(rp)) for c, rp in self.root_ports.items()}))
            self.sp.start()
            wr.close()
            self.sconn = rd
            if self.csv_f:                     # the sentinel writes the rows now
                self.csv_f.close()
                self.csv_f = self.csv_w = None
            self.result["sentinel"] = dict(pid=self.sp.pid, started_utc=utc())
            mark(f"sentinel process {self.sp.pid} polls the chips every {self.a.poll_s}s and writes the telemetry"
                 + (f"; heartbeat/LnkSta every 1 s to {self.a.link_poll}" if self.a.link_poll else ""))
        except Exception as e:
            self.sp = self.sconn = None
            self.result["sentinel"] = dict(error=f"{type(e).__name__}: {e}")
            mark(f"sentinel process did not start ({type(e).__name__}: {e}); the watchdog thread polls alone")

    def sentinel_alive(self):
        if self.sp is None:
            return False
        if self.sp.is_alive():
            return True
        self.drain()
        mark(f"sentinel process exited (code {self.sp.exitcode}); the watchdog thread polls alone")
        self.result.setdefault("sentinel", {})["exited_utc"] = utc()
        self.sp = None
        if self.a.telem and self.csv_f is None:
            self.csv_f = open(self.a.telem, "a", newline="")
            self.csv_w = csv.writer(self.csv_f, lineterminator="\n")
        return False

    def drain(self):
        """Messages from the sentinel: the latest stats; a loss returns its record."""
        loss = None
        try:
            while self.sconn is not None and self.sconn.poll():
                kind, body = self.sconn.recv()
                if kind == "stats":
                    self.stats = body
                elif kind == "link":
                    self.link_state = body
                elif kind == "loss" and loss is None:
                    loss = body
        except (OSError, EOFError):
            self.sconn = None
        return loss

    def stop_sentinel(self):
        if self.sp is None:
            return
        self.shm["stop"].value = 1
        self.sp.join(timeout=2 * self.a.poll_s + 2)
        self.drain()

    def sweep(self):
        return {c: aiclk(c) for c in self.ids}

    def watch(self):
        last_t = 0.0
        while not self.stop.is_set():
            try:
                if self.holding:                  # a held hang: its snapshot runs; no chip reads, no deadlines
                    self.stop.wait(self.a.poll_s)
                    continue
                if self.sentinel_alive():         # it polls and samples; we read its messages
                    loss = self.drain()
                    if loss:
                        self.on_loss(loss["chips"], t_loss=loss["epoch"], detected_by="sentinel",
                                     capture=loss.get("capture"), last_ok=loss.get("last_ok"))   # does not return
                else:
                    clocks = self.sweep()
                    gone = [c for c in self.ids if not alive(c, clocks[c])]
                    if gone:
                        self.on_loss(gone)            # does not return
                    self.last_ok = time.time()
                    if time.time() - last_t >= 1.0:
                        last_t = time.time()
                        self.sample(clocks)
                now = time.monotonic()
                with self.lock:
                    phase, deadline = self.phase, self.deadline
                with self.lock:
                    op = self.op
                if op is not None and now > op[1]:
                    self.ccl_hang(op)                     # does not return
                if deadline is not None and now > deadline and self.a.load == "allgather" and phase == "burst":
                    self.ccl_hang(op, overrun=(now - self.phase_t0, round(deadline - self.phase_t0)))
                if deadline is not None and now > deadline:
                    self.abandoned = True
                    self.finish("ERROR", EXIT_ERROR, error=f"timeout: phase '{phase}' ran past its limit "
                                f"(hung device call?); exiting without closing the device")
                if self.stop_at is not None and now > self.stop_at + self.a.stop_grace_s:
                    self.abandoned = True
                    self.finish("INTERRUPTED", EXIT_ERROR,
                                error=f"stop requested; cleanup did not end within {self.a.stop_grace_s}s")
            except Exception as e:  # never let the watchdog die: it is the only exit for a hung device call
                mark(f"watchdog error {type(e).__name__}: {e}")
            self.stop.wait(self.a.poll_s)

    def sample(self, clocks):
        with self.lock:
            phase = self.phase
        sample_into(self.stats, self.ids, clocks, phase, self.burst_t0 is not None, self.csv_w, self.a.telem_full)
        if self.csv_f:
            self.csv_f.flush()

    def on_loss(self, gone, t_loss=None, detected_by="watchdog", capture=None, last_ok=None):
        """First chip(s) lost: log at once (tripcap.sh triggers on FIRST LOSS; the sentinel process prints it itself
        when it saw the loss), capture the lost chip's root port link and AER, wait --settle-s from the loss (the rest
        of the tray follows), re-check the chips, read the tray CPLDs, the SEL and the lost chips' root-port AER again,
        classify the loss (CHIP_HANG / DROP / UNKNOWN), log every lost chip, exit 3. Any loss is a TRIP, exit 3, even when
        every chip answers again after --settle-s (that is recorded as transient_chips, loss_confirmed false)."""
        t_loss = t_loss or time.time()
        with self.lock:
            if self.finishing or self.trip_pending:   # another path is exiting / reporting already
                threading.Event().wait()
            self.trip_pending = True
            phase = self.phase
        last_ok = last_ok or self.last_ok
        first = dict(utc=utc(t_loss), chips=gone, pci=[self.ids[c] for c in gone],
                     trays=sorted({tray_of(self.ids[c]) for c in gone}, key=str), phase=phase,
                     last_all_present_utc=utc(last_ok) if last_ok else None, detected_by=detected_by,
                     reported_utc=utc())
        if self.burst_t0 is not None:
            first["s_after_burst_begin"] = round(t_loss - self.burst_t0, 2)
        if self.result.get("steps"):
            first["last_step"] = self.result["steps"][-1]
        if detected_by != "sentinel":
            mark(first_loss_line(first))
        cap = self.result["loss_capture"] = dict(link=capture or link_capture(gone[0], self.ids[gone[0]], self.a.lspci))
        time.sleep(max(0.0, self.a.settle_s - (time.time() - t_loss)))
        now_lost = [c for c in self.ids if not present(c)]
        transient = [c for c in gone if c not in now_lost]
        lost = now_lost or gone
        trays = sorted({tray_of(self.ids[c]) for c in lost}, key=str)
        if transient:
            mark(f"TRANSIENT LOSS: chip(s) {transient} answer again {self.a.settle_s}s after the first loss "
                 f"(still a TRIP: any loss is a drop)")
        cap["cpld"] = cpld_read(self.a.ipmitool)
        sel = sel_tail(self.a.sel_ipmitool or self.a.ipmitool)
        # no baseline: no line is known to be new (an old FRB must not read as this loss's drop)
        cap["sel_new"] = sel_new(self.result.get("sel_pre"), sel) \
            if sel is not None and self.result.get("sel_pre") is not None else None
        sel_read = sel is not None and self.result.get("sel_pre") is not None   # no baseline: unknown, never CHIP_HANG
        # root-port AER again after the settle (the AER can land seconds after the loss); a rise in either
        # sample counts, on every lost chip's root port
        link, pre = cap["link"], self.result.get("aer_pre") or {}
        ports = {c: self.root_ports.get(c) for c in lost}
        if link.get("root_port") and link.get("chip") in ports and not ports[link["chip"]]:
            ports[link["chip"]] = link["root_port"]
        cap["aer_after_settle"] = {rp: aer_counts(port_dir(rp, self.port_dirs)) for rp in sorted(set(filter(None, ports.values())))}
        rise_at_loss = {link["root_port"]: aer_rose(link.get("aer_root_port"), pre.get(link["root_port"]))} \
            if link.get("root_port") else {}
        rise_after = {rp: aer_rose(cnt, pre.get(rp)) for rp, cnt in cap["aer_after_settle"].items()}
        cap["aer_rise"] = dict(at_loss=rise_at_loss, after_settle=rise_after)
        aer_up = {}
        for sample in (rise_at_loss, rise_after):
            for rp, rise in sample.items():
                for f, cnt in rise.items():
                    for k, v in cnt.items():
                        d = aer_up.setdefault(rp, {}).setdefault(f, {})
                        d[k] = max(d.get(k, 0), v)
        cls, basis = classify_loss(lost, self.ids, cap["cpld"], cap["sel_new"], sel_read,
                                   self.result.get("mmio_timeout"), aer_up, ports)
        mark(f"loss class {cls}: whole trays lost {basis['whole_trays_lost'] or 'none'}, CPLD not ff "
             f"{basis['cpld_not_ff'] or 'none'}, lost trays CPLD ff/ff {basis['lost_trays_cpld_ff']}, SEL FRB on lost "
             f"tray {len(basis['sel_frb']) if sel_read else 'unread' if sel is None else 'no baseline'}, MMIO timeout "
             f"{basis['mmio_timeout']}, root ports {sorted(set(filter(None, ports.values())))} AER up "
             f"{aer_up or 'none'} (at loss {'+'.join(rise_at_loss) or 'unread'}, after settle "
             f"{'+'.join(rise_after) or 'unread'})"
             + (f"; not CHIP_HANG for want of {basis['missing_for_chip_hang']}" if cls != "CHIP_HANG" else "")
             + (f"; DROP on {basis['drop_evidence']}" if cls == "DROP" else ""))
        mark(fail_line(lost, len(self.ids), trays, cls, [self.ids.get(c) for c in lost]))
        self.printed = True
        self.finish("TRIP", EXIT_TRIP, first_loss=first, lost_chips=lost, lost_trays=trays, _no_wait=True,
                    loss_class=cls, loss_class_basis=basis, transient_chips=transient, loss_confirmed=bool(now_lost))

    # the one exit ------------------------------------------------------------------------------------------
    def ccl_hang(self, op, error=None, overrun=None):
        """A CCL op hung (or tt-metal timed it out) past its limit. A chip gone is a trip; else CCL_HANG, exit 4.
        op: self.op, the (name, deadline, start) of the all_gather in flight, or None. overrun: (seconds, limit) when
        the burst phase as a whole ran past its limit. Every chip is still on PCIe and a hung gather does not say
        which chip stalled, so the line names the variant and its mesh axis, and says the chip is unknown.
        Does not return: the device call may still be running, so it exits without closing the device."""
        gone = [c for c in self.ids if not present(c)]
        if gone:
            self.on_loss(gone)            # does not return
        self.abandoned = True
        n = len(self.ids)
        v = next((x for x in AG_VARIANTS if op and x[0] == op[0]), None)
        s_in_op = round(time.monotonic() - op[2], 1) if op else None
        if v:
            what = f"op {ag_scope(v[0], v[1], v[2], self.a.mesh)} did not finish in {s_in_op:.0f}s"
        elif op:
            what = f"op {op[0]} did not finish in {s_in_op:.0f}s"
        else:
            what = "no all_gather in flight (op unknown)"
        if overrun:
            what += f"; the burst phase ran {overrun[0]:.0f}s, past its {overrun[1]}s limit"
        line = f"CCL hang: chip unknown ({n}/{n} still on PCIe), {what}"
        hang = dict(op=op[0] if op else None, cluster_axis=v[1] if v else None, dim=v[2] if v else None,
                    s_in_op=s_in_op, burst_phase_s=round(overrun[0], 1) if overrun else None, phase=self.phase,
                    utc=utc(), chips="unknown", chips_present=n, tt_metal_error=error, line=line)
        if self.a.hold_on_hang:
            self.result["ccl_hang"] = hang
            self.result["noc_snapshot"] = self.hang_snapshot(op[0] if op else "burst")   # a second caller waits there
            gone = [c for c in self.ids if not present(c)]
            if gone:                      # the chip left the bus during the hold: a trip, snapshot kept in the result
                mark(f"chip(s) {gone} gone after the hang snapshot: reporting the loss")
                self.on_loss(gone, detected_by="after_hang_snapshot")   # does not return
        self.finish("CCL_HANG", EXIT_CCL_HANG, ccl_hang=hang, error=line + (f"; {error}" if error else ""))

    def hang_suspects(self):
        """Chips for the snapshot's lspci: --suspect-pci, plus any chip whose heartbeat stood still >= 2 s or whose
        root port link is down in the last --link-poll sample; every chip when none is known."""
        ls, sus, why = self.link_state or {}, {}, []
        for want in filter(None, (self.a.suspect_pci or "").split(",")):
            want = want.strip().lower()
            want = want if want.count(":") == 2 else "0000:" + want      # 02:00.0 -> 0000:02:00.0
            for c, pci in self.ids.items():
                if pci.lower() == want:
                    sus[c] = "suspect-pci"
        for c, st in (ls.get("hb_still_s") or {}).items():
            if st is not None and st >= 2 and c in self.ids:
                sus.setdefault(c, f"heartbeat still {st}s")
        for c in ls.get("link_down") or []:
            if c in self.ids:
                sus.setdefault(c, "root port link down")
        if not sus:
            why.append("no suspect chip known (no --suspect-pci, no heartbeat stall or link down seen): every chip")
            sus = {c: "all" for c in self.ids}
        return sus, why

    def hang_snapshot(self, what):
        """--hold-on-hang: a CCL op hung with every chip on the bus. Keep the process (and the hang) alive and run
        --snapshot-cmd (noc_snapshot.sh: lspci -vvv of the suspects' root ports and endpoints, dmesg tail, tt-triage
        NOC0 then NOC1), each step under its own timeout, then hold --hang-hold-s more. No chip read from this process
        or the sentinel meanwhile (shm hold), only root-port LnkSta. Returns the record for the result JSON."""
        with self.lock:
            busy = self.holding or self.finishing or self.trip_pending
            if not busy:
                self.holding = True
        if busy:                          # another path holds or reports already; it ends the process
            threading.Event().wait()
        self.shm["hold"].value = 1
        t0 = time.time()
        self.drain()
        sus, why = self.hang_suspects()
        base = os.path.dirname(os.path.abspath(self.a.result)) if self.a.result else os.getcwd()
        d = self.a.snapshot_dir or os.path.join(base, f"noc_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        pairs = [f"{self.root_ports.get(c) or ''}={self.ids[c]}" for c in sorted(sus)]
        rec = dict(dir=d, op=what, begin_utc=utc(t0), suspects={c: dict(pci=self.ids[c], root_port=self.root_ports.get(c),
                   why=w) for c, w in sorted(sus.items())}, notes=why, link_state=self.link_state,
                   cmd=self.a.snapshot_cmd, rc=None)
        mark(f"HOLD ON HANG: op {what} hung with {len(self.ids)}/{len(self.ids)} chips on PCIe; process {os.getpid()} "
             f"stays alive, no reset. Snapshot to {d}: suspects {chip_list(sus)}"
             + (f" ({'; '.join(f'{c}: {w}' for c, w in sorted(sus.items()) if w != 'all')})" if any(
                 w != "all" for w in sus.values()) else ""))
        env = dict(os.environ, SNAP_TRIAGE_TIMEOUT_S=str(self.a.snapshot_timeout_s))
        if self.a.lspci and self.a.lspci != "none":
            env["SNAP_LSPCI"] = self.a.lspci
        if self.a.triage:
            env["SNAP_TRIAGE"] = self.a.triage
        limit = 20 * len(pairs) + 40 + 2 * (self.a.snapshot_timeout_s + 15) + 40   # its steps' own timeouts, plus slack
        p = None
        try:
            p = subprocess.Popen([*self.a.snapshot_cmd.split(), d, *pairs], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.snap_p = p
            pg_record(p.pid)              # run.sh's TERM trap reaps it too (lib/pgroups.sh)
            timer = threading.Timer(limit, lambda: os.killpg(p.pid, signal.SIGKILL))
            timer.daemon = True
            timer.start()
            for ln in p.stdout:
                mark(ln.rstrip())
            rec["rc"] = p.wait()
            timer.cancel()
            if rec["rc"] == -signal.SIGKILL:
                rec["error"] = f"snapshot killed after {limit}s"
        except BaseException as e:    # cannot start it, or interrupted: stop its process group, keep the record
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
            if not isinstance(e, Exception):
                raise
        rec["steps"] = read_steps(os.path.join(d, "steps.tsv"))
        rec["seconds"] = round(time.time() - t0, 1)
        mark(f"snapshot end rc {rec['rc']} in {rec['seconds']}s: " + (", ".join(
            f"{st['step']} {st['rc']}" for st in rec["steps"]) or rec.get("error") or "no steps.tsv"))
        if self.a.hang_hold_s > 0:
            end = time.monotonic() + self.a.hang_hold_s
            mark(f"holding the hang {self.a.hang_hold_s}s more (pid {os.getpid()}, no chip reads): attach "
                 "exalens / tt-triage by hand now")
            while time.monotonic() < end:
                time.sleep(min(60.0, max(0.0, end - time.monotonic())))
                mark(f"holding, {max(0, end - time.monotonic()):.0f}s left")
            rec["held_s"] = self.a.hang_hold_s
        self.drain()
        rec["link_state_end"] = self.link_state
        return rec

    def finish(self, verdict, code, _no_wait=False, **extra):
        if not _no_wait and self.trip_pending:   # the watchdog is reporting a trip: let it exit
            threading.Event().wait(self.a.settle_s + LOSS_CAPTURE_MAX_S + 10)
        with self.lock:
            if self.finishing:                          # another path is exiting already
                busy = True
            else:
                busy, self.finishing = False, True
        if busy:
            threading.Event().wait(30)
            os._exit(code)
        self.stop.set()
        if self.wd is not None and threading.current_thread() is not self.wd:
            self.wd.join(timeout=2 * self.a.poll_s + 5)
        self.stop_sentinel()
        if verdict != "TRIP" and self.closed_cleanly():
            skip = [w for w, ok in (("close failed", self.close_ok), ("fmax restore failed", self.restore_ok))
                    if ok is False]
            if skip:      # a device call failed: do not send it more messages
                self.result["long_idle"] = "skipped: " + ", ".join(skip)
                mark(f"GO_LONG_IDLE {self.result['long_idle']}")
            else:
                self.go_long_idle()
        r = self.result
        r.update(extra, verdict=verdict, exit_code=code, end_utc=utc(), phase_at_end=self.phase,
                 burst_begin_utc=utc(self.burst_t0) if self.burst_t0 else None, telemetry=self.stats)
        try:
            if self.csv_f:
                self.csv_f.flush()
        except (OSError, ValueError, RuntimeError):   # RuntimeError: reentrant, from a signal handler
            pass
        if self.result.get("device_used"):
            save_state(self.a.state_file, last_end_utc=utc(), last_end_epoch=time.time())
        write_result(self.a.result, r)
        if not self.printed:      # dit-stress and our harness read RESULT PASS / RESULT FAIL
            if verdict == "TRIP":
                lost = extra.get("lost_chips", [])
                trays = sorted({tray_of(self.ids.get(c)) for c in lost}, key=str)
                mark(fail_line(lost, len(self.ids), trays, extra.get("loss_class", "UNKNOWN"),
                               [self.ids.get(c) for c in lost]))
            else:
                word = dict(NO_TRIP="PASS", PREFLIGHT_REFUSED="REFUSED", DRY_RUN_OK="DRY-RUN-OK",
                            CCL_HANG="CCL_HANG").get(verdict, "ERROR")
                mark(f"RESULT {word} verdict={verdict} exit={code}" +
                     (f" error={extra['error']}" if extra.get("error") else ""))
        mark(f"result json {self.a.result}")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (OSError, ValueError, RuntimeError):
            pass
        if verdict != "TRIP" and self.closed_cleanly() and self.close_ok is not False and self.restore_ok is not False:
            self.teardown_exit(code)
        os._exit(code)

    def teardown_exit(self, code):
        """Clean exit only: let tt-metal's C++ teardown run (MetalContext/~Cluster, which puts the active ERISC cores
        back on base firmware). os._exit skips it, and after a fabric session the broker's post-job eth-heartbeat
        then times out and its fabric check fails on the first active eth core. libc exit() runs the static
        destructors from any thread. A watchdog timer bounds it: after TEARDOWN_MAX_S it prints TEARDOWN TIMEOUT,
        records it in the result JSON (teardown.status) and hard-exits with the same code. The timer needs the GIL,
        so SIGALRM (default action, no handler is set) ends a teardown hung holding it 10 s later; the result JSON,
        written already, keeps the exit code for run_repro.sh."""
        self.result["teardown"] = dict(status="exit_started", max_s=TEARDOWN_MAX_S, start_utc=utc())
        write_result(self.a.result, self.result)
        mark(f"teardown: libc exit({code}), hard exit after {TEARDOWN_MAX_S} s")
        signal.alarm(TEARDOWN_MAX_S + 10)
        threading.Timer(TEARDOWN_MAX_S, self.teardown_timeout, (code,)).start()
        try:
            import ctypes
            ctypes.CDLL(None).exit(code)
        except (ImportError, OSError, AttributeError):
            pass

    def teardown_timeout(self, code):
        try:
            self.result["teardown"].update(status="TIMEOUT", end_utc=utc())
            write_result(self.a.result, self.result)
            mark(f"TEARDOWN TIMEOUT after {TEARDOWN_MAX_S}s: tt-metal teardown did not finish, hard exit {code}")
            sys.stdout.flush()
        except BaseException:
            pass
        finally:
            os._exit(code)

    # back to idle clock on a clean exit ----------------------------------------------------------------------
    def closed_cleanly(self):
        """The device was used, no device call is left running (no hard exit, the mesh closed, fmax restored) and no
        chip is lost."""
        return (bool(self.result.get("device_used")) and not self.abandoned and self.mesh is None
                and not self.fmax_set and not self.trip_pending and all(present(c) for c in self.ids))

    def go_long_idle(self):
        """AICLK_GO_LONG_IDLE (ARC 0x54) to every chip, under --device-timeout-s. tt-metal sends it only from its
        atexit teardown (~Cluster -> UMD LocalChip::close_device -> BlackholeDeviceFirmware::set_clock_state(IDLE);
        ttnn.close_mesh_device keeps the default context), which os._exit skips:
        without this the chips stay at the GO_BUSY clock (1350 MHz) and the next cold gate (<= 800 MHz) never
        passes without a power cycle. Recorded in result["long_idle"]; a failure here does not change the verdict.

        The watchdog has stopped by now, so a chip lost in this window (up to --device-timeout-s + 5 s) would go
        unseen: the sysfs readback below checks for it, warns and sets result["long_idle_chip_missing"]. The verdict
        and exit code stay as the run ended (the load was over and every chip was present after the close), so a
        lost chip here is not counted as a load trip; the flag lets the reader of the result decide."""
        t0 = time.monotonic()
        rec = self.result["long_idle"] = dict(sent=0, chips=len(self.ids), failed=[], timeout_s=self.a.device_timeout_s)

        def send():
            import tt_umd
            for c in self.ids:
                try:
                    dev = tt_umd.TTDevice.create(c)
                    dev.init_tt_device()
                    rc = dev.arc_msg(MSG_AICLK_GO_LONG_IDLE, True, [], 1000)
                    del dev
                    if rc[0] == 0:
                        rec["sent"] += 1
                        continue
                    rec["failed"].append(dict(chip=c, error=f"exit code {rc[0]:#x}"))
                except Exception as e:
                    rec["failed"].append(dict(chip=c, error=f"{type(e).__name__}: {str(e)[:120]}"))

        th = threading.Thread(target=send, name="ub0-long-idle", daemon=True)
        th.start()
        th.join(self.a.device_timeout_s)
        if th.is_alive():
            rec["error"] = f"timeout: GO_LONG_IDLE did not finish within {self.a.device_timeout_s}s"
        give_up = time.monotonic() + 5                  # the clock follows within ~1 s; record what sysfs reads
        while True:
            clocks = [aiclk(c) for c in self.ids]
            if all(v is not None and v <= IDLE_MHZ for v in clocks) or time.monotonic() > give_up:
                break
            time.sleep(0.25)
        known = [v for v in clocks if v is not None and v < LOST_AICLK]
        at_idle = [v is not None and v <= IDLE_MHZ for v in clocks]
        rec.update(aiclk_max_after=max(known) if known else None, chips_at_idle=sum(at_idle),
                   seconds=round(time.monotonic() - t0, 2))
        mark(f"GO_LONG_IDLE sent to {rec['sent']}/{len(self.ids)} chips; {rec['chips_at_idle']}/{len(self.ids)} "
             f"at <= {IDLE_MHZ} MHz (max {rec['aiclk_max_after']})" + (f"; {rec['error']}" if "error" in rec else ""))
        missing = [c for c, idle in zip(self.ids, at_idle) if not idle and not present(c)]
        if missing:
            trays = sorted({tray_of(self.ids[c]) for c in missing}, key=str)
            rec.update(missing_chips=missing, missing_pci=[self.ids[c] for c in missing], missing_trays=trays)
            self.result["long_idle_chip_missing"] = True
            mark(f"WARNING chip(s) {missing} trays {trays} missing after GO_LONG_IDLE (lost after the load ended, "
                 f"while the watchdog was stopped); verdict unchanged, see long_idle_chip_missing in the result")

    # signals -----------------------------------------------------------------------------------------------
    def on_signal(self, signum, frame):
        name = signal.Signals(signum).name
        if self.finishing:                # already writing the result and exiting
            return
        if self.stop_at is not None:
            self.abandoned = True
            self.finish("INTERRUPTED", EXIT_ERROR, error=f"second signal {name}: exiting at once")
        self.stop_at = time.monotonic()
        mark(f"{name} received: stopping (burst stops, mesh closes; hard exit in {self.a.stop_grace_s}s)")
        snap = getattr(self, "snap_p", None)   # the snapshot runs in its own session: TERM it, KILL in 5 s
        if snap is not None and snap.poll() is None:
            pg_term(snap.pid)
        raise Interrupted(name)

    # cleanup when no chip was lost -------------------------------------------------------------------------
    def close_quietly(self, why):
        """Close the mesh (if open) and restore fmax (if set), each under its own deadline. Either way the mesh and the
        fmax are not touched again; close_ok/restore_ok record whether each step worked (finish() checks them)."""
        if self.mesh is not None:
            self.set_phase("close", self.a.close_timeout_s)
            self.close_ok = False
            try:
                self.ttnn.close_mesh_device(self.mesh)
                self.ttnn.set_fabric_config(self.ttnn.FabricConfig.DISABLED)
                mark(f"mesh closed ({why})")
                self.close_ok = True
            except Exception as e:
                mark(f"mesh close failed ({why}): {type(e).__name__}: {str(e)[:200]}")
            self.mesh = None
        if self.fmax_set:
            self.set_phase("fmax_restore", self.a.device_timeout_s)
            self.restore_ok = False
            try:
                ok, arb, _ = set_fmax(0)
                mark(f"fmax restored on {ok}/{len(self.ids)} chips; arb_max {arb}")
                self.restore_ok = ok == len(self.ids)
            except Exception as e:
                mark(f"fmax restore failed ({why}): {type(e).__name__}: {str(e)[:200]}")
            self.fmax_set = False


def read_steps(path):
    """noc_snapshot.sh's steps.tsv as [{step, rc, seconds, file}]; [] if missing."""
    try:
        with open(path) as f:
            return [dict(zip(("step", "rc", "seconds", "file"), ln.rstrip("\n").split("\t")))
                    for ln in f.readlines()[1:] if ln.strip()]
    except OSError:
        return []


def write_result(path, r):
    if not path:
        return
    try:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(r, f, indent=1, default=str)
            f.write("\n")
        os.replace(tmp, path)
    except OSError as e:
        mark(f"cannot write result json {path}: {e}")


def set_fmax(mhz):
    """ARC 0x23 on every local chip (mhz <= 0 restores the default). Lives in fw RAM until a chip reset."""
    import tt_umd
    ok = 0
    for i in chips():
        dev = tt_umd.TTDevice.create(i)
        dev.init_tt_device()
        rc = dev.arc_msg(MSG_SET_ASIC_HOST_FMAX, True, [max(mhz, 0), 1 if mhz <= 0 else 0], 1000)
        ok += rc[0] == 0
        del dev
    time.sleep(2)  # telemetry lags the message by 1-2 s
    per = {}  # AICLK_ARB_MAX per chip (None: not readable), for load_test.sh's per-chip cap check
    for i in chips():
        dev = tt_umd.TTDevice.create(i)
        dev.init_tt_device()
        tel = dev.get_arc_telemetry_reader()
        per[i] = (tel.read_entry(tt_umd.TelemetryTag.AICLK_ARB_MAX) & 0xFFFF
                  if tel.is_entry_available(tt_umd.TelemetryTag.AICLK_ARB_MAX) else None)
        del tel, dev
    return ok, sorted({v for v in per.values() if v is not None}), per


class Parser(argparse.ArgumentParser):
    def error(self, message):  # a usage error is exit 1 (argparse's own 2 means PREFLIGHT REFUSED here)
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def parse_args(argv=None):
    ap = Parser(description=__doc__.split("\n\n")[0], epilog=EXIT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fmax", type=int, default=None, help="AICLK ceiling in MHz for this run (restored at exit "
                    "unless a chip was lost); default: clock left as found")
    ap.add_argument("--topo", choices=("ring", "linear"), default=None,
                    help="fabric of a load with one: ring (the default of ub0, allgather) or linear (FABRIC_1D), for a box "
                         "cabled without some wrap links (run.sh passes linear when a wrap port is down)")
    ap.add_argument("--load", choices=sorted(LOADS), default="ub0",
                    help="ub0 (default, the tested repro: fabric ring up but idle, matmul-only burst), matmul "
                         "(the same burst with no fabric at all) or allgather (ring fabric, all_gather CCL burst)")
    ap.add_argument("--idle-s", type=int, default=60, help="open-mesh idle before the burst (default 60)")
    ap.add_argument("--burst-s", type=int, default=30, help="burst length in s (default 30)")
    ap.add_argument("--cycles", type=int, default=1, help="quiet + burst cycles in this one mesh open (default 1; "
                    "the bus-bar qb part's quiet-burst steps in one process; --load ub0 only)")
    ap.add_argument("--heavy", choices=sorted(HEAVY), default=os.environ.get("TRAY_HEAVY") or HEAVY_DEFAULT,
                    help=f"matmul loads: the burst op (default {HEAVY_DEFAULT}, env TRAY_HEAVY; lofi_all = 16k_bf8_lofi + PCIe "
                         "write + all_gather + DRAM add per step; mm = the old 2048x8192 @ 8192x16384)")
    ap.add_argument("--burst-gap-ms", type=float, default=float(os.environ.get("TRAY_BURST_GAP_MS") or BURST_GAP_MS),
                    help=f"pause (not a quiet: no clock check) after every --burst-steps heavy ops (default {BURST_GAP_MS:g}, env TRAY_BURST_GAP_MS; "
                         "0 = back to back, no sync inside the burst)")
    ap.add_argument("--burst-steps", type=int, default=BURST_STEPS, help=f"heavy ops per burst (default {BURST_STEPS})")
    ap.add_argument("--sfpu-par", type=int, choices=(0, 1), default=0,
                    help="1: the heavy matmul carries a fused --sfpu-act SFPU epilogue on every output tile, the SFPU "
                         "working in parallel with the FPU (chunk_load.py's --sfpu-par; run.sh's qb and step "
                         "bursts); 0 (default): the plain matmul")
    ap.add_argument("--sfpu-act", default=None, help="--sfpu-par 1's activation, <ttnn.UnaryOpType>[:<param>] (default "
                    "SFPU_ACT, the accurate gelu)")
    ap.add_argument("--variant", default=os.environ.get("FLEET_LOAD_VARIANT", ""),
                    help="dev load variant (fleet/load_variant.py): host-stress runs host workers for the "
                         "burst; no-gap comes as --burst-gap-ms 0; dram-mm stays out (lofi_all is the burst's link and PCIe traffic)")
    ap.add_argument("--telem", default="", help="CSV: per-chip aiclk and power every 1 s (sysfs only)")
    ap.add_argument("--result", default=None,
                    help="result JSON path (default ./ub0_result_<host>_<UTC>.json; '' = none)")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and cold-state check only (sysfs reads), print the plan, no device open; "
                         "exit 0 if it would run, 2 if refused")
    ap.add_argument("--wait-quiet-min", type=float, default=0,
                    help="before loading, wait until every chip has read <= 800 MHz for this many minutes in a row "
                         "(31 confirms a cold start); gives up (exit 2) after 2x this + 30 min")
    ap.add_argument("--state-file", default=os.environ.get("UB0_STATE", "/var/tmp/ub0_repro.state.json"),
                    help="records when this tool last used the device, for the cold-state check ('' = none)")
    g = ap.add_argument_group("timeouts and watchdog (a hung phase ends the run with exit 1, or 3 if chips are lost)")
    g.add_argument("--device-timeout-s", type=int, default=120, help="per --fmax set/restore pass (default 120)")
    g.add_argument("--import-timeout-s", type=int, default=600, help="import torch + ttnn (default 600)")
    g.add_argument("--open-timeout-s", type=int, default=300, help="mesh open incl. fabric (default 300)")
    g.add_argument("--upload-timeout-s", type=int, default=600, help="weight/activation upload (default 600)")
    g.add_argument("--burst-grace-s", type=int, default=900, help="burst limit = --burst-s + this; covers the first "
                   "run's kernel compile and the last iteration/readback (default 900)")
    g.add_argument("--close-timeout-s", type=int, default=120, help="mesh close (default 120)")
    g.add_argument("--ccl-op-timeout-s", type=int, default=60, help="--load allgather: one all_gather op plus its "
                   "sync; past it with every chip present = CCL_HANG, exit 4 (default 60)")
    g.add_argument("--ccl-first-op-timeout-s", type=int, default=600, help="the same for each variant's first call, "
                   "which compiles its kernels (default 600)")
    g.add_argument("--stop-grace-s", type=int, default=60, help="after SIGINT/SIGTERM, hard exit if cleanup takes "
                   "longer (default 60)")
    g.add_argument("--poll-s", type=float, default=0.25, help="watchdog chip poll period (default 0.25)")
    e = ap.add_argument_group("loss evidence (read-only; nothing here writes to a chip or a power path)")
    e.add_argument("--ipmitool", default="none", metavar="CMD",
                   help="in-band ipmitool command for the tray CPLD reads before the run and after a loss, e.g. "
                   "'timeout 15 sudo -n ipmitool' (run_repro.sh passes it); 'none' (default): no CPLD read, a loss "
                   "with no other drop evidence (a whole tray lost) is UNKNOWN")
    e.add_argument("--sel-baseline", default="", metavar="FILE",
                   help="skip the per-run CPLD/SEL baseline reads (14-31 s); the SEL a loss is read against is FILE "
                   "('ipmitool sel elist last 40' taken once by run.sh). The after-loss CPLD and SEL reads still run "
                   "with --ipmitool. FILE missing or empty: no SEL baseline, so the SEL neither shows nor rules out a drop")
    e.add_argument("--sel-ipmitool", default="", metavar="CMD",
                   help="ipmitool command for the BMC SEL (default: --ipmitool; run_repro.sh passes lanplus when "
                   "BMC_HOST is set, IPMI_PASSWORD from the environment)")
    e.add_argument("--lspci", default="lspci", metavar="CMD",
                   help="lspci command for the lost chip's root port right after a loss ('none': skip)")
    h = ap.add_argument_group("hang debug (--load allgather; read-only: nothing here resets or writes to a chip)")
    h.add_argument("--hold-on-hang", action="store_true",
                   help="on CCL_HANG keep the process (and the hang) alive and run --snapshot-cmd first: lspci -vvv of "
                   "the suspects' root ports and endpoints, dmesg tail, tt-triage --noc-id=0 then 1; then exit 4 as "
                   "before. Turns on --telem-full and --link-poll auto. Ignored for --load ub0/matmul")
    h.add_argument("--snapshot-cmd", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "noc_snapshot.sh"),
                   metavar="CMD", help="the snapshot hook: CMD OUT_DIR ROOT_PORT=ENDPOINT... (default: noc_snapshot.sh "
                   "beside this script)")
    h.add_argument("--snapshot-dir", default="", help="snapshot output (default: noc_snapshot_<UTC> beside --result)")
    h.add_argument("--snapshot-timeout-s", type=int, default=120, help="per tt-triage pass (default 120)")
    h.add_argument("--triage", default=os.environ.get("UB0_TRIAGE", ""), metavar="CMD",
                   help="tt-triage command (default $UB0_TRIAGE, else the hook's own search: "
                   "$TT_METAL_HOME/tools/triage/triage.py, tt-triage on PATH)")
    h.add_argument("--hang-hold-s", type=int, default=0, help="after the snapshot, hold the hang this long for "
                   "exalens/tt-triage by hand (default 0)")
    h.add_argument("--suspect-pci", default="", metavar="BDF[,BDF]",
                   help="chips the snapshot's lspci always covers, e.g. 02:00.0 (with heartbeat-still and link-down "
                   "chips from --link-poll; every chip when none is known)")
    h.add_argument("--telem-full", action="store_true", help="--telem rows also get current_a, vcore_mv, temp_c")
    h.add_argument("--link-poll", default="auto", metavar="CSV",
                   help="1 s ARC heartbeat (tt_heartbeat) + root port LnkSta per chip to CSV; 'auto' (default): "
                   "<--telem stem>_link.csv with --hold-on-hang, else off; '' = off")
    nm = ap.add_argument_group("numeric check (matmul loads only; never stops the load)")
    nm.add_argument("--numeric-every", type=int, default=int(os.environ.get("TRAY_NUMERIC_EVERY") or 500),
                    help="> 0: 16 checks spaced by time over the burst (the first and the last always; 0 = only those; "
                         "-1 = off; default 500, env TRAY_NUMERIC_EVERY; the value is the old op spacing, unused)")
    nm.add_argument("--numeric-budget-s", type=float, default=float(os.environ.get("TRAY_NUMERIC_BUDGET_S") or 90),
                    help="seconds the after-burst readback may spend on its middle checks (default 90, env "
                         "TRAY_NUMERIC_BUDGET_S; the default burst's 18 checks take 32-41 s); op 1 and the last "
                         "checked output are always read; skipped ones give a NUMERIC WARN per tray")
    nm.add_argument("--numeric-cpu-rows", type=int, default=int(os.environ.get("TRAY_NUMERIC_CPU_ROWS") or 32),
                    help="rows of the golden checked against a CPU fp32 matmul (default 32)")
    g.add_argument("--settle-s", type=float, default=2.0, help="wait after the first loss before counting all lost "
                   "chips (default 2)")
    a = ap.parse_args(argv)
    a.heavy_act = (a.sfpu_act or SFPU_ACT) if a.sfpu_par else None   # heavy_op's fused epilogue (as chunk_load.py)
    tp = a.topo
    for key, val in LOADS[a.load].items():
        setattr(a, key, val)
    if tp and a.topo != "none":   # --topo overrides the load's fabric; a load with none keeps none
        a.topo = tp
    if os.environ.get("TRAY_NUMERIC", "1") == "0" or a.load == "allgather":
        a.numeric_every = -1
    if a.cycles < 1 or (a.cycles > 1 and a.load == "allgather"):
        ap.error(f"--cycles {a.cycles}: 1 or more, and 1 with --load allgather")
    a.hold_on_hang = a.hold_on_hang and a.load == "allgather"
    a.telem_full = a.telem_full or a.hold_on_hang
    if a.link_poll == "auto":
        a.link_poll = (os.path.splitext(a.telem)[0] + "_link.csv") if a.hold_on_hang and a.telem else ""
    if a.result is None:
        a.result = f"ub0_result_{os.uname().nodename}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    return a


def quiet_clock(run):
    """The run's QuietClock (quiet_clock.py): its settle() records go to result["quiet_clock"]."""
    if getattr(run, "qc", None) is None:
        import quiet_clock as qcm
        run.qc = qcm.QuietClock(run.ids, say=mark)
        run.result["quiet_clock"] = run.qc.log
    return run.qc


def wait_quiet(run, ids, minutes):
    """Every chip at <= 800 MHz for `minutes` in a row; returns the seconds watched in a row. The clock check
    first asks every chip above idle for its idle clock and waits for all to read <= 800 MHz, at most 5 min:
    the quiet counts only from there. A chip it names in a QUIET FAIL (a finding) is not waited for again,
    and the quiet goes on: no give-up, no refusal for a hot chip (the old minutes*2+30 min give-up is gone). A chip
    that rises again during the count restarts it after a new clock check, once; a second rise is a QUIET WARN naming the
    chips and the count goes on (so the wait is at most 2 clock checks and 2 counts)."""
    RESTARTS = 1
    need, qc = minutes * 60, quiet_clock(run)
    mark(f"waiting for {minutes:g} min of every chip at <= {IDLE_MHZ} MHz in a row (cold state)")
    qc.settle("cold quiet")
    t0, restarts, last_log = time.monotonic(), 0, 0.0
    while True:
        clocks = {c: aiclk(c) for c in ids}
        if any(not alive(c, clocks[c]) for c in ids):
            return time.monotonic() - t0          # preflight re-checks and refuses
        hot = [c for c, v in clocks.items() if v > IDLE_MHZ and c not in qc.stuck]
        if hot and restarts < RESTARTS:
            restarts += 1
            mark(f"chip(s) {hot} back above {IDLE_MHZ} MHz during the quiet: clock check again, quiet restarts "
                 f"({restarts}/{RESTARTS})")
            qc.settle(f"cold quiet restart {restarts}")
            t0 = time.monotonic()
        elif hot and restarts == RESTARTS:
            restarts += 1
            named = ", ".join(f"{ids[c]} (tray {tray_of(ids[c])}) chip {c} {clocks[c]} MHz" for c in hot)
            mark(f"QUIET WARN cold quiet: chip(s) {named} rose above {IDLE_MHZ} MHz again after the restart; "
                 "the quiet goes on")
        quiet = time.monotonic() - t0
        if quiet >= need:
            return quiet
        if time.monotonic() - last_log >= 60:  # the broker reaps a job silent for 300 s
            last_log = time.monotonic()
            mark(f"quiet for {quiet / 60:.1f}/{minutes:g} min")
        time.sleep(min(5.0, max(0.1, need - quiet)))


def cyc_phase(name, cyc):
    """--cycles: cycle 1's phases are "idle" and "burst"; cycle 2 on: "idle@2", "burst@2" (their own telemetry buckets)."""
    return name if cyc == 1 else f"{name}@{cyc}"


def cyc_tag(a, cyc):
    return f" {cyc}/{a.cycles}" if a.cycles > 1 else ""


def open_quiet(run, idle_s, cyc=1):
    """The idle with the mesh open : clock check first (every chip to <= 800 MHz, at most 5 min),
    then idle_s counted from there, then every chip asked for its busy clock again and logged at the burst start.
    cyc > 1 (--cycles): the quiet after the cycle before's burst, the same clock check and idle_s."""
    import quiet_clock as qcm
    qc = quiet_clock(run)
    if cyc > 1:   # run.sh's quiet-burst stall limit reads this line: the quiet's clock check and idle print nothing
        # a loss in this quiet is this cycle's, timed from no burst (phase idle@N), not from the cycle before's burst
        run.result["cycle"], run.burst_t0 = cyc, None
        run.result.pop("burst_t0", None)
        mark(f"cycle {cyc}/{run.a.cycles} begin (clock check, then idle {idle_s}s)")
    run.set_phase(cyc_phase("idle", cyc), idle_s + qc.cap_s + qcm.BUSY_WAIT_S + 3 * qcm.SEND_TIMEOUT_S + 60)
    qc.settle("open quiet")
    mark(f"idle {idle_s}s begin" + cyc_tag(run.a, cyc))
    time.sleep(idle_s)
    qc.busy("open quiet")
    mark(f"AICLK before burst{cyc_tag(run.a, cyc)}: {clocks_line(run.ids)}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    ids = chips()
    run = Run(a, ids)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, run.on_signal)
    # SIGUSR1: every thread's Python stack to stderr from the C handler, no GIL needed (run.sh's per-try wall-clock
    # limit sends it to a hung try before stopping it; a try once hung with the main thread in a futex)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    try:
        guarded(run)
    except BaseException as e:  # a signal or error during the cleanup itself
        run.finish("INTERRUPTED" if isinstance(e, Interrupted) else "ERROR", EXIT_ERROR,
                   error=f"during cleanup: {type(e).__name__}: {str(e)[:200]}")


def guarded(run):
    a, ids = run.a, run.ids
    try:
        body(run)
    except Interrupted as e:
        if run.holding:                         # stopped during a held hang: leave the device as it is
            run.finish("INTERRUPTED", EXIT_ERROR, error=f"stopped by {e} while holding a CCL hang")
        if not run.trip_pending and not [c for c in ids if not present(c)]:
            run.close_quietly(f"after {e}")
        run.finish("INTERRUPTED", EXIT_ERROR, error=f"stopped by {e}")
    except BaseException as e:  # an op failing after a drop: let the watchdog report it if chips are gone
        err = f"{type(e).__name__}: {str(e)[:300]}"
        mark(f"exception {err}")
        if run.holding:                         # a held hang (its snapshot runs): no close, no more device calls
            run.result["exception_during_hold"] = dict(utc=utc(), error=err)
            threading.Event().wait()
        if "MMIO" in str(e) and "timeout" in str(e).lower():   # UMD "MMIO per-op timeout": classified on a loss
            run.result["mmio_timeout"] = dict(utc=utc(), phase=run.phase, error=str(e)[:300],
                                              last_step=(run.result.get("steps") or [None])[-1])
            mark(f"UMD MMIO timeout in phase {run.phase}: loss class (CHIP_HANG / DROP / UNKNOWN) follows if a chip is lost")
        if run.wd is not None and not run.trip_pending:
            time.sleep(a.settle_s + 1)          # the watchdog sees a loss within --poll-s and exits 3 itself
        if a.load == "allgather" and run.phase in ("warmup", "burst") and "TIMEOUT" in str(e) \
                and "hang" in str(e).lower():
            run.ccl_hang(run.op, error=err)     # tt-metal's own op timeout; no return
        if run.trip_pending:                    # it is reporting one (CPLD/SEL reads take a while): let it exit
            threading.Event().wait(a.settle_s + LOSS_CAPTURE_MAX_S)
        lost = [c for c in ids if not present(c)]
        if lost:
            run.result["error"] = err
            if not run.trip_pending:
                run.on_loss(lost, detected_by="main_thread")   # does not return
            run.finish("TRIP", EXIT_TRIP, lost_chips=lost, _no_wait=True, loss_class="UNKNOWN",
                       first_loss=dict(utc=utc(), chips=lost, pci=[ids[c] for c in lost], phase=run.phase,
                                       trays=sorted({tray_of(ids[c]) for c in lost}, key=str)))
        try:
            run.close_quietly("after an error")
        except Interrupted:
            pass
        run.finish("ERROR", EXIT_ERROR, error=f"{type(e).__name__}: {str(e)[:300]}")


# --heavy: the burst's op, enqueued back to back (full load, far above DIDT). "mm" is the old
# 2048x8192 @ 8192x16384 ttnn.matmul; the 16k ones are 16384^3 minimal_matmul with all of their inputs in DRAM.
# Measured on one box (notes/load-baseline.md): box kW and TFLOP/s per chip for each.
HEAVY = {"mm": None,
         "16k_hifi2": ("HiFi2", False, 2, 4, "bfloat16"),
         "16k_hifi4": ("HiFi4", True, 2, 2, "bfloat16"),
         "16k_bf8_lofi": ("LoFi", False, 2, 4, "bfloat8_b"),
         "lofi_all": ("LoFi", False, 2, 4, "bfloat8_b"),
         "16k_bfp4_lofi": ("LoFi", False, 2, 4, "bfloat4_b"),
         "8k_fp32_hifi4": ("HiFi4", True, 2, 2, "float32"),
         "8k_int32_eltwise": ("HiFi4", False, 0, 0, "int32")}
# Number formats the bf16/bf8 matmuls leave out, run as chunk_load.py --heavy-x in
# the heavy chunk's tail. 16k_bfp4_lofi: bfloat4_b inputs, bf16 output. 8k_fp32_hifi4: float32 in, fp32 dest acc, at 8192
# (a 16k float32 tensor is 1 GiB per chip). 8k_int32_eltwise: int32 A*B+C on the SFPU integer path, plus an int8 round trip
# (int32 -> uint8 -> int32 typecast) added in; no matmul. Sizes other than 16384 are in HEAVY_SIZE.
HEAVY_SIZE = {"8k_fp32_hifi4": 8192, "8k_int32_eltwise": 8192}
HEAVY_XB = {"bfloat8_b": 1, "bfloat4_b": 0.5, "float32": 4, "int32": 4}   # bytes per input element; 2 = bf16


def heavy_dram_bytes(h, int8=True):
    """One --heavy op's DRAM bytes per chip. A matmul reads A and B and writes its output (bfloat4_b writes bf16).
    8k_int32_eltwise counts each op it runs, 4 bytes per int32 and 1 per uint8 element: multiply A*B
    (read 8, write 4), add p+C (8, 4), and with the int8 round trip (int8, its default) typecast A to uint8 (4, 1),
    back to int32 (1, 4) and add o+v (8, 4): 46 bytes per element, 24 without the round trip."""
    S, dt = HEAVY_SIZE.get(h, 16384), HEAVY[h][4]
    if h in HEAVY_ELTWISE:
        return S * S * (12 + 12 + (5 + 5 + 12 if int8 else 0))
    return S * S * (2 * HEAVY_XB.get(dt, 2) + (2 if dt == "bfloat4_b" else HEAVY_XB.get(dt, 2)))
HEAVY_ELTWISE = {"8k_int32_eltwise"}
# lofi_all: 16k_bf8_lofi's matmul, then in each step a host -> device PCIe write (on
# command queue 1 when the mesh has 2), an in-box all_gather and a DRAM add (notes/load-baseline.md: the
# only load that went past the 130 W TDP), run in bursts of BURST_STEPS steps with a BURST_GAP_MS pause after each.
HEAVY_EXTRAS = {"lofi_all": ("pcie", "all_gather", "dram")}
# The default: the highest tray kW mean measured (9.31 kW, 328.8 TFLOP/s per chip). 16k_hifi4
# throws in minimal_matmul on ttnn 0.80.0 and falls back to ttnn.matmul at 80.8 TFLOP/s.
# The default: lofi_all with 50 ms gaps. On one box (notes/gap-tuning.md) it
# gave the highest chip and box peaks of the gap sweep: 288 W per chip (122% over TDP), 11.52 kW PSU peak, 297 TFLOP/s per chip.
HEAVY_DEFAULT = "lofi_all"
# chunk_load.py --sfpu-par 1's fused epilogue: the fast gelu (param 1). The A/B on one box
# (3 x 10 min sustained load, notes/sfpu-par.md): gelu:1 kept matmul TFLOP/s per MHz within 1.4% of no epilogue (the
# 2% no-FPU-stall rule) at 11.47 kW box against 11.31; the accurate gelu:0 cost 15% per MHz (an FPU stall) and fell to 11.11 kW
SFPU_ACT = "gelu:1"
# Measured TFLOP/s per chip of each --heavy (same table; lofi_all at 50 ms gaps): the op cap when ttnn has no events
HEAVY_TFLOPS = {"mm": 137.5, "16k_hifi2": 171.0, "16k_hifi4": 80.8, "16k_bf8_lofi": 328.8, "lofi_all": 297.0,
                # measured: heavy_x_rate rows of a smoke run (median of 3 warm runs alone, per chip)
                "16k_bfp4_lofi": 509.08, "8k_fp32_hifi4": 80.27, "8k_int32_eltwise": 0.03}
# lofi_all's PCIe host -> device write per step, MB per chip: 16 in a C4 burst (--pcie-mb 16). The sustained part
# (the bus-bar stage chunks, full's long chunk included) writes 16 too: an earlier setting of 48 came from one box's run
# (11.19 kW mean against 10.90 kW for 16), but another box measured 16 MB at 12.15 kW mean and
# 48 MB at 11.89 kW. chunk_load.py's --heavy-pcie-mb sets it (FLEET_SUSTAIN_PCIE_MB); heavy_op reads a.heavy_pcie_mb.
HEAVY_PCIE_MB = 16
SUSTAINED_PCIE_MB = 16
BURST_STEPS = 8          # heavy ops per burst before a gap (bursts of 8 steps, then the gap)
BURST_GAP_MS = 50.0      # pause after each burst (a gap, not a quiet: no clock check) (the best gap measured; env TRAY_BURST_GAP_MS, 0 = back to back)
# A minimal_matmul fallback runs ttnn.matmul: its measured 16k rate (the 16k_hifi4 fallback, the slowest measured) bounds the
# no-event op cap, so a fallback burst stops queuing within --burst-s of device time
FALLBACK_TFLOPS = HEAVY_TFLOPS["16k_hifi4"]
RUN_AHEAD_OPS = 4        # heavy ops the host may queue ahead of the device
NUMERIC_CHECKS = 16      # checked matmuls per burst, spaced by time, plus the first and the last
MAX_PENDING_CHECKS = 20  # checked outputs (64 MiB each per chip, 1.25 GiB at most) held until the burst's synchronize


class RunAhead:
    """Keeps the host at most k heavy ops ahead of the device, so a --burst-s burst runs --burst-s on the device, not
    2-4x that (60 s queued took 96-258 s to run). An event after each op and a host-only wait
    on the one k ops back: the device always has k ops queued, so it never idles and gets no synchronize. Without
    ttnn events: no wait, the burst stops queuing at --burst-s times the load's measured rate (HEAVY_TFLOPS)."""

    def __init__(self, ttnn, mesh, k, cap):
        self.rec, self.wait = getattr(ttnn, "record_event", None), getattr(ttnn, "event_synchronize", None)
        self.mesh, self.k, self.cap, self.evs, self.why = mesh, k, cap, [], None
        if not (self.rec and self.wait):
            self.why = "ttnn has no record_event/event_synchronize"

    def queued(self, nh):
        """After heavy op nh is queued: True while the burst may queue another."""
        if self.why is None:
            try:
                self.evs.append(self.rec(self.mesh, 0))
                if len(self.evs) > self.k:
                    self.wait(self.evs.pop(0))   # the host waits; the device still has k ops queued
                return True
            except Exception as e:
                self.why, self.evs = f"event: {type(e).__name__}: {str(e)[:120]}", []
        return nh < self.cap

    def desc(self):
        return (f"host at most {self.k} ops ahead of the device (ttnn events)" if self.why is None else
                f"op cap {self.cap} (--burst-s x measured rate; {self.why})")


class BurstGap:
    """--burst-gap-ms: after every --burst-steps heavy ops, wait for the device to finish them
    (one synchronize) and enqueue nothing for the gap, so each burst starts from idle chips: one di/dt step per burst.
    No readback in it. 0 ms: back to back, no synchronize (the old load)."""

    def __init__(self, ttnn, mesh, steps, gap_ms, ra=None):
        self.ttnn, self.mesh, self.steps, self.gap_ms, self.ra = ttnn, mesh, steps, gap_ms, ra
        self.n, self.s = 0, 0.0   # gaps taken, seconds spent in them

    def on(self):
        return self.gap_ms > 0 and self.steps > 0

    def after(self, nh):
        """After heavy op nh is queued: at the end of each burst, synchronize and pause for the gap."""
        if not self.on() or nh <= 0 or nh % self.steps:
            return False
        self.ttnn.synchronize_device(self.mesh)
        if self.ra is not None:
            self.ra.evs = []   # every recorded op has finished
        t = time.time()
        time.sleep(self.gap_ms / 1000.0)
        self.s += time.time() - t
        self.n += 1
        return True

    def desc(self):
        return (f"bursts of {self.steps} steps, {self.gap_ms:g} ms pause after each" if self.on() else
                "no gap (back to back)")


def op_cap(burst_s, heavy, flops, fallback=None):
    """RunAhead's op cap without ttnn events: the heavy ops the device runs in --burst-s at the measured rate of the op
    that runs (a fallback: at most FALLBACK_TFLOPS), so a burst never queues more than --burst-s of work."""
    rate = min(HEAVY_TFLOPS[heavy], FALLBACK_TFLOPS) if fallback else HEAVY_TFLOPS[heavy]
    return int(burst_s * rate * 1e12 / flops) + 1


def read_back(nc, ttnn, pend, mark):
    """The burst's checked outputs, read back and compared after its one synchronize while --numeric-budget-s lasts;
    op 1 (the golden's) and the last checked output always; once the budget is spent or the checks are
    off, the rest are deallocated unread and counted (numeric_report.py makes them a NUMERIC WARN).
    Returns (seconds spent, outputs skipped)."""
    spent, skipped = 0.0, 0
    for i, (pn, k, yc, ts) in enumerate(pend):
        if not nc.off and (pn == 1 or i == len(pend) - 1 or nc.spent < nc.budget_s):
            spent += nc.check(yc, pn, k, ts)
        else:
            skipped += 1
        ttnn.deallocate(yc)
    if skipped:
        mark(f"numeric readback: {skipped} of {len(pend)} checked outputs skipped, deallocated unread ("
             + ("checks off after an error" if nc.off else f"--numeric-budget-s {nc.budget_s:g}s spent") + ")")
    return spent, skipped


def check_due(n, last, s, burst_s, queued, every):
    """Queue a checked matmul now? Op 1 and the last always; else (every > 0) when s seconds into the burst reach the
    next of NUMERIC_CHECKS slots of burst_s / NUMERIC_CHECKS (queued: the checks already queued, op 1 the first).
    Spaced by time, not op count: a 16k heavy op takes 30-110 ms, so every 500th op gave 2-3 checks (finding 3)."""
    if n == 1 or last:
        return True
    return every > 0 and queued <= NUMERIC_CHECKS and s >= queued * burst_s / NUMERIC_CHECKS


def _fused_act(ttnn, name, ran):
    """(UnaryWithParam, label) for --sfpu-act <name>[:<param>] (e.g. gelu:0, the accurate gelu), or None: no name, or
    this ttnn lacks the op type (ran["sfpu"] then says so and the matmul runs without it)."""
    if not name:
        return None
    op, _, p = name.partition(":")
    try:
        t = getattr(ttnn.UnaryOpType, op.upper())
        return (ttnn.UnaryWithParam(t, float(p)) if p else ttnn.UnaryWithParam(t)), name
    except Exception as e:  # noqa: BLE001
        ran["sfpu"] = f"fused {name} off ({type(e).__name__}: {str(e)[:120]})"
        return None


def _int_eltwise_op(a, ttnn, torch, mesh):
    """--heavy 8k_int32_eltwise. int32 A*B+C, 8192x8192 per chip, in DRAM, on the SFPU integer
    path; inputs within +-2^15 so A*B+C stays inside int32 (bit-exact, no overflow). Then the int8 path: A through
    int32 -> uint8 -> int32 typecasts, added to the result, so a wrong uint8 pack or unpack also changes the output. A ttnn
    without the uint8 typecast leaves it out and op.ran["int8"] says why. FLOPs: 3 integer ops per element."""
    S = HEAVY_SIZE[a.heavy]
    rep, DRAM = ttnn.ReplicateTensorToMesh(mesh), ttnn.DRAM_MEMORY_CONFIG

    def up(seed):
        torch.manual_seed(seed)
        return ttnn.from_torch(torch.randint(-(1 << 15), 1 << 15, (1, 1, S, S), dtype=torch.int32), dtype=ttnn.int32,
                               layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=rep)
    A, B, C = up(4), up(5), up(6)
    ran = dict(op="ttnn.multiply + ttnn.add (int32)", fallback=None)
    st = {"mm": None, "int8": getattr(ttnn, "uint8", None) is not None}
    if not st["int8"]:
        ran["int8"] = "off: ttnn has no uint8"

    def op():
        p = ttnn.multiply(A, B, memory_config=DRAM)
        o = ttnn.add(p, C, memory_config=DRAM); ttnn.deallocate(p)
        if st["int8"]:
            try:
                u = ttnn.typecast(A, ttnn.uint8, memory_config=DRAM)
                v = ttnn.typecast(u, ttnn.int32, memory_config=DRAM); ttnn.deallocate(u)
                o2 = ttnn.add(o, v, memory_config=DRAM); ttnn.deallocate(v); ttnn.deallocate(o)
                o = o2
            except Exception as e:   # refused on the host: the int32 part goes on alone
                st["int8"] = False
                ran["int8"] = f"off ({(str(e).splitlines() or [''])[0][:160] or type(e).__name__})"
                print(f"heavy op: int8 round trip {ran['int8']}", flush=True)
        return o
    op.tensors, op.state, op.ran = (A, B, C), st, ran
    return op, 3 * S * S, f"int32 multiply+add {S}x{S} + int32->uint8->int32 typecast"


def heavy_op(a, ttnn, torch, mesh, x, w, cq1=None, fabric=True, cq_why=None):
    """(op, FLOPs per op, description) for --heavy; op.tensors are the inputs it uploaded (freed at the end); op.ran the
    op that runs, {"op": name, "fallback": None or why minimal_matmul is not used}. A load with
    HEAVY_EXTRAS (lofi_all) adds op.ran["load"] and op.ran["extras_fallback"] (each extra in line or off, and why):
    cq1 the PCIe write's queue (None: queue 0, cq_why says why), fabric False: no all_gather."""
    if HEAVY[a.heavy] is None:
        op = lambda: ttnn.matmul(x, w)
        op.tensors, op.ran = (), dict(op="ttnn.matmul", fallback=None)
        return op, 2 * a.m * a.k * a.n, f"bf16 matmul {a.m}x{a.k} @ {a.k}x{a.n}"
    if a.heavy in HEAVY_ELTWISE:
        return _int_eltwise_op(a, ttnn, torch, mesh)
    fid, fp32, sh, sw, dt = HEAVY[a.heavy]
    S = HEAVY_SIZE.get(a.heavy, 16384)
    rep, dtype = ttnn.ReplicateTensorToMesh(mesh), getattr(ttnn, dt)
    tdt = torch.float32 if dt == "float32" else torch.bfloat16

    def up(seed, scale):
        torch.manual_seed(seed)
        return ttnn.from_torch(torch.randn(1, 1, S, S, dtype=tdt) * scale, dtype=dtype,
                               layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=rep)
    A, B = up(2, 1.0), up(3, 0.02)
    if dt == "bfloat4_b":   # a bf16 output: bfloat4_b inputs, the check compares bf16 bits
        okw = {"dtype": ttnn.bfloat16}
    else:
        okw = {}
    ck = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True)
    g = mesh.compute_with_storage_grid_size()
    cfg = ttnn.MinimalMatmulConfig(M_block_size=8, K_block_size=16, N_block_size=8, subblock_h=sh, subblock_w=sw,
                                   compute_with_storage_grid_size=g)
    st = {"mm": getattr(getattr(ttnn, "experimental", None), "minimal_matmul", None)}
    ran = (dict(op="ttnn.experimental.minimal_matmul", fallback=None) if st["mm"] is not None else
           dict(op="ttnn.matmul", fallback="ttnn.experimental.minimal_matmul missing"))
    # a.heavy_act (chunk_load.py --sfpu-par 1): an SFPU activation fused into the matmul's
    # epilogue, applied to every output tile in DST just before it is packed, while the readers fetch the next blocks;
    # FPU and SFPU are both driven by the math thread, so this is the SFPU load that rides along the matmul with no op of
    # its own. Refused on the host: the matmul goes on without it and ran["sfpu"] says why.
    act = _fused_act(ttnn, getattr(a, "heavy_act", None), ran)
    kw = {"fused_activation": act[0]} if act else {}

    def op():
        if st["mm"] is not None:
            try:
                return st["mm"](A, B, config=cfg, compute_kernel_config=ck, **kw, **okw)
            except Exception as e:   # refused on the host before anything is queued
                why = f"minimal_matmul refused: {(str(e).splitlines() or [''])[0][:160] or type(e).__name__}"
                if okw:   # no output dtype argument: the output keeps the input format
                    ran["out_dtype"] = f"{dt} output ({why})"
                    okw.clear()
                    return op()
                if kw:   # first without the epilogue: the matmul is the load, the SFPU part of it is extra
                    kw.clear(); ran["sfpu"] = f"fused {act[1]} off ({why})"
                    print(f"heavy op: {ran['sfpu']}; minimal_matmul without it", flush=True)
                    return op()
                print(f"heavy op: {why}; ttnn.matmul instead", flush=True)
                st["mm"] = None
                ran.update(op="ttnn.matmul", fallback=why)
                if act:
                    ran["sfpu"] = f"fused {act[1]} off (ttnn.matmul fallback)"
        try:
            return ttnn.matmul(A, B, compute_kernel_config=ck, **okw)
        except TypeError:
            okw.clear()
            return ttnn.matmul(A, B, compute_kernel_config=ck)
    desc = (f"{dt} {fid}{' fp32-acc' if fp32 else ''} matmul {S}x{S} @ {S}x{S}"
            f"{f' + fused {act[1]} SFPU epilogue' if act else ''}")
    ex = HEAVY_EXTRAS.get(a.heavy, ())
    if not ex:
        op.tensors, op.state, op.ran = (A, B), st, ran   # op.state["mm"] None: minimal_matmul was refused (chunk_load.py's LOAD line)
        return op, 2 * S ** 3, desc
    W, DRAM = 2048, ttnn.DRAM_MEMORY_CONFIG
    ph = max(32, ((getattr(a, "heavy_pcie_mb", None) or HEAVY_PCIE_MB) << 20) // (2 * W) // 32 * 32)   # host -> device MB per chip

    def up16(shape, seed, dev=True):
        torch.manual_seed(seed)
        t = torch.randn(*shape, dtype=torch.bfloat16)
        if not dev:
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                               mesh_mapper=rep)
    hsrc, hdst = up16([1, 1, ph, W], 8, dev=False), up16([1, 1, ph, W], 9)
    G, D1, D2 = up16([1, 1, 1024, 4096], 10), up16([1, 1, 8192, W], 11), up16([1, 1, 8192, W], 12)
    shape = tuple(getattr(mesh, "shape", ()) or ())
    ax = 1 if len(shape) == 2 and shape[1] <= shape[0] else 0   # the 4-way axis (axis 1 of 8x4), in-box links only
    off = {}   # extra -> why it is off
    if not fabric:
        off["all_gather"] = "all_gather off: no fabric (--topo none)"
    lin = getattr(getattr(ttnn, "Topology", None), "Linear", None)

    def ag():
        try:
            return ttnn.all_gather(G, dim=3, cluster_axis=ax, memory_config=DRAM, topology=lin)
        except TypeError:
            return ttnn.all_gather(G, dim=3, cluster_axis=ax, memory_config=DRAM)

    def fb():
        f = ([f"pcie on cq 0 ({cq_why or 'one command queue'})"] if cq1 is None and "pcie" not in off else [])
        return "; ".join(f + [off[e] for e in ex if e in off]) or None

    def all_op():
        y = op()
        for e in ex:
            if e in off:
                continue
            try:
                if e == "pcie":
                    ttnn.copy_host_to_device_tensor(hsrc, hdst, **({"cq_id": cq1} if cq1 is not None else {}))
                elif e == "all_gather":
                    ttnn.deallocate(ag())
                else:
                    ttnn.deallocate(ttnn.add(D1, D2, memory_config=DRAM))
            except Exception as err:   # refused on the host: that extra goes off, the matmul and the rest go on
                off[e] = f"{e} refused: {(str(err).splitlines() or [''])[0][:160] or type(err).__name__}"
                print(f"{a.heavy}: {off[e]}; the load goes on without it", flush=True)
                ran["extras_fallback"] = fb()
        return y
    ran.update(load=a.heavy, extras_fallback=fb())
    all_op.tensors, all_op.state, all_op.ran = (A, B, hdst, G, D1, D2), st, ran
    return all_op, 2 * S ** 3, (f"{a.heavy} ({desc} + {ph * W * 2 >> 20} MB PCIe write"
                                f"{' on cq 1' if cq1 is not None else ''} + all_gather [1, 1, 1024, 4096] cluster_axis {ax} + DRAM add per step)")


def fabric_name(a):
    return None if a.topo == "none" else "FABRIC_1D" if a.topo == "linear" else "FABRIC_1D_RING"


def ag_desc(name, axis, dim, shape):
    out = list(shape)
    out[dim] *= AG_WAYS[axis]
    call = (f"all_gather(dim={dim}, cluster_axis={axis})" if axis is not None else
            f"all_gather(dim={dim}, cluster_axis=1) then all_gather(dim={dim}, cluster_axis=0)")
    return (f"{name}: {call} {list(shape)} {nbytes(shape) / 2**20:.0f} MiB -> "
            f"{out} {nbytes(out) / 2**30:.2f} GiB per chip")


def ag_scope(name, axis, dim, mesh):
    """The variant and the mesh axis it gathers along, in words, for the CCL hang line."""
    r, c = (int(x) for x in mesh.lower().split("x"))
    where = {0: f"{r}-way down each of the {c} columns", 1: f"{c}-way along each of the {r} rows",
             None: f"{r * c}-way over all chips, axis 1 then axis 0,"}[axis]
    return f"{name} (cluster_axis {axis}, dim {dim}: {where} of the {mesh} mesh)"


def nbytes(shape):
    n = 2                                 # bf16
    for v in shape:
        n *= v
    return n


def traffic_plan(a):
    """What the run puts on the device, in words (logged, kept in the result)."""
    fab = (f"fabric {fabric_name(a)} set before the open (routers start on the ethernet cores, up all run)"
           if fabric_name(a) else "NO fabric (no fabric config, no routers started, no CCL possible)")
    if a.load == "allgather":
        return (f"{fab}; burst {a.burst_s}s on all chips of the {a.mesh} mesh: bf16 DRAM ttnn.all_gather back to "
                "back, in turn: " + "; ".join(ag_desc(*v) for v in AG_VARIANTS) + "; NO matmul")
    gap = (f"in bursts of {a.burst_steps} with {a.burst_gap_ms:g} ms pause after each" if a.burst_gap_ms > 0 and
           a.burst_steps > 0 else "back to back, one synchronize at the end")
    ccl = ("all_gather on in-box links in each step" if "all_gather" in HEAVY_EXTRAS.get(a.heavy, ()) and fabric_name(a)
           else "NO CCL op")
    return f"{fab}; burst {a.burst_s}s on all chips of the {a.mesh} mesh: --heavy {a.heavy} {gap}, {ccl}"


def body(run):
    a, ids = run.a, run.ids
    mark(f"start host={os.uname().nodename} chips={len(ids)} load={a.load} m={a.m} k={a.k} n={a.n} "
         f"idle={a.idle_s}s burst={a.burst_s}s topo={a.topo} mesh={a.mesh} "
         f"fmax={a.fmax}{' DRY-RUN' if a.dry_run else ''}")
    mm = run.result["tt_mm_env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_MM_")}
    mark(f"mitigation env seen by this (the ttnn) process: {' '.join(f'{k}={v}' for k, v in mm.items()) or 'none'}")
    tr = run.result["traffic"] = dict(load=a.load, plan=traffic_plan(a), fabric_config=fabric_name(a),
                                      fabric_init=False, matmuls=0)
    mark(f"traffic plan: {tr['plan']}")
    if a.hold_on_hang or a.telem_full or a.link_poll:
        hd = run.result["hang_debug"] = dict(hold_on_hang=a.hold_on_hang, snapshot_cmd=a.snapshot_cmd,
                                             triage=a.triage or None, telem_full=a.telem_full,
                                             link_poll=a.link_poll or None, suspect_pci=a.suspect_pci or None)
        mark("hang debug: " + " ".join(f"{k}={v}" for k, v in hd.items()))
    mark("trays: " + " ".join(f"{t}:{[c for c, p in ids.items() if tray_of(p) == t]}" for t in (1, 2, 3, 4)))
    v = run.result["versions"] = versions(ids)
    mark("versions: " + " ".join(f"{k}={'/'.join(x) if isinstance(x, list) else x}" for k, x in v.items()))

    # preflight: refuse before any load if a chip or tray is missing
    run.set_phase("preflight")
    problems = preflight(ids)
    watched = None
    if not problems and a.wait_quiet_min > 0 and not a.dry_run:
        run.set_phase("wait_quiet")
        watched = wait_quiet(run, ids, a.wait_quiet_min)
        problems = preflight(ids)
        mark(f"AICLK after the quiet: {clocks_line(ids)}")
    holding, hidden = holders()
    st = load_state(a.state_file) if a.state_file else {}
    last_end = st.get("last_end_epoch") or st.get("last_start_epoch")
    if st.get("last_start_epoch") and (not last_end or st["last_start_epoch"] > last_end):
        last_end = st["last_start_epoch"]
    up = uptime_s()
    clocks = {c: aiclk(c) for c in ids}
    cold = cold_state(clocks, holding, up, time.time() - last_end if last_end else None, watched)
    cold.update(uptime_s=up, last_use_by_this_tool_utc=utc(last_end) if last_end else None,
                holders_unreadable_procs=hidden,
                aiclk_now=[min((x for x in clocks.values() if x is not None), default=None),
                           max((x for x in clocks.values() if x is not None and x < LOST_AICLK), default=None)])
    run.result["cold_state"] = cold
    if cold["status"] == "cold":
        mark(f"cold state: CONFIRMED ({'; '.join(cold['reasons'])})")
    elif cold["status"] == "likely_cold":
        mark(f"cold state: LIKELY ({'; '.join(cold['reasons'])})")
    else:
        bar = "!" * 100
        mark(bar)
        mark(f"WARNING cold state {cold['status'].upper()}: a trip needs a cold start (first open after a chassis "
             f"power cycle, or >= {COLD_S // 60} min with no process on the device). A NO TRIP here says little.")
        for r in cold["reasons"]:
            mark(f"WARNING   {r}")
        if hidden:
            mark(f"WARNING   {hidden} processes could not be checked for open device handles (run as root to check)")
        mark(bar)
    if problems:
        for p in problems:
            mark(f"PREFLIGHT REFUSED: {p}")
        mark("nothing was loaded. Recover the box first; a reset with a tray missing is allowed")
        run.finish("PREFLIGHT_REFUSED", EXIT_REFUSED, preflight_problems=problems)
    mark(f"preflight ok: {len(ids)}/{TRAYS * CHIPS_PER_TRAY} chips, {TRAYS} trays, all answering")
    if a.dry_run:
        mark(f"DRY-RUN plan: {'fmax ' + str(a.fmax) + ' MHz, ' if a.fmax is not None else ''}open {a.mesh} mesh "
             f"(fabric {a.topo}), upload, idle {a.idle_s}s, burst {a.burst_s}s of {a.load} "
             f"(--heavy {a.heavy}, back to back); limits open {a.open_timeout_s}s, burst {a.burst_s + a.burst_grace_s}s")
        run.finish("DRY_RUN_OK", EXIT_NO_TRIP)

    if a.sel_baseline:                            # run.sh's once-a-run SEL read: no per-try BMC wait
        base = load_sel_baseline(a.sel_baseline)
        run.result["sel_pre"] = base
        mark(f"SEL baseline from {a.sel_baseline}: {len(base) if base else 'unread'} line(s); no CPLD baseline read")
    elif a.ipmitool != "none" or a.sel_ipmitool:  # before the device is touched: the baseline a loss is read against
        run.result["cpld_pre"] = cpld_read(a.ipmitool)
        run.result["sel_pre"] = sel_tail(a.sel_ipmitool or a.ipmitool)
        mark(f"CPLD before the run (reg 0x1E/0x19 per tray): "
             f"{ {t: '/'.join(str(v[r]) for r in CPLD_REGS) for t, v in run.result['cpld_pre'].items()} }; "
             f"SEL {'read' if run.result['sel_pre'] is not None else 'unread'}")
    run.result["aer_pre"] = {}                    # root port AER counters now, so a loss reads what rose in the run
    for c, pci in ids.items():
        d = pci_dir(c, pci)
        up = os.path.dirname(d) if d else ""
        if os.path.basename(up).count(":") == 2:
            run.result["aer_pre"][os.path.basename(up)] = aer_counts(up)
            run.root_ports[c], run.port_dirs[os.path.basename(up)] = os.path.basename(up), up
    run.result["device_used"] = True
    save_state(a.state_file, last_start_utc=utc(), last_start_epoch=time.time())
    run.start_watchdog()
    if a.fmax is not None:
        run.set_phase("fmax_set", a.device_timeout_s)
        run.fmax_set = bool(a.fmax)
        ok, arb, per = set_fmax(a.fmax)
        mark(f"fmax {a.fmax or 'stock'} sent to {ok}/{len(ids)} chips; arb_max readback {arb}")
        mark(f"arb_max per chip ({sum(v is not None for v in per.values())}/{len(ids)} read): "
             + " ".join(f"{c}={'-' if v is None else v}" for c, v in sorted(per.items())))

    # the load: unchanged from the validated ub0 sequence
    if a.load == "allgather":   # tt-metal's own op timeout, a backup behind the watchdog's per-op limit
        os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", str(a.ccl_first_op_timeout_s + 60))
    run.set_phase("import", a.import_timeout_s)
    import torch
    import ttnn
    run.ttnn = ttnn
    mark("imports done")
    run.set_phase("mesh_open", a.open_timeout_s)
    if a.topo != "none":
        run.step_mark(f"fabric: set_fabric_config({fabric_name(a)}) called")
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric_name(a)))
        run.step_mark(f"fabric: set_fabric_config({fabric_name(a)}) returned")
    r, c = (int(v) for v in a.mesh.lower().split("x"))
    run.step_mark("UMD start: open_mesh_device called (UMD cluster start, then fabric routers come up)")
    cq_why, mesh = None, None
    if "pcie" in HEAVY_EXTRAS.get(a.heavy, ()):   # lofi_all's PCIe write on cq 1, beside cq 0's matmul
        try:
            mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c), num_command_queues=2)
        except Exception as e:
            cq_why = f"2 command queues refused ({type(e).__name__}: {str(e)[:120]})"
            mark(f"mesh open: {cq_why}; one queue")
    if mesh is None:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c))
    run.mesh = mesh
    cq1 = (ttnn.QueueId(1) if hasattr(ttnn, "QueueId") else 1) if "pcie" in HEAVY_EXTRAS.get(a.heavy, ()) and \
        cq_why is None else None
    run.step_mark("UMD start: open_mesh_device returned (fabric up)")
    mark(f"mesh open {tuple(mesh.shape)}")
    mark(f"AICLK after mesh open: {clocks_line(ids)}")   # measured: the open takes the chips to stock 1350 MHz
    tr["fabric_init"] = a.topo != "none"
    mark(f"traffic: fabric init {'done, ' + fabric_name(a) if tr['fabric_init'] else 'NOT done (no fabric)'}")

    def upload(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    if a.load == "allgather":
        allgather_burst(run, ttnn, torch, mesh, upload, tr)   # does not return
    run.set_phase("upload", a.upload_timeout_s)
    torch.manual_seed(0)
    w_cpu = torch.randn(1, 1, a.k, a.n, dtype=torch.bfloat16) * 0.02
    x_cpu = torch.randn(1, 1, a.m, a.k, dtype=torch.bfloat16)
    w, x = upload(w_cpu), upload(x_cpu)
    nc = xb = None
    if a.numeric_every >= 0:   # numeric check: a 2nd activation, so a stale output cannot pass
        from numeric_check import NumericCheck
        nc = NumericCheck(torch, ttnn, mesh, a.numeric_every, a.numeric_budget_s, a.numeric_cpu_rows, mark)
        torch.manual_seed(1)
        xb_cpu = torch.randn(1, 1, a.m, a.k, dtype=torch.bfloat16)
        xb = upload(xb_cpu)
        nc.set_inputs(0, x_cpu, w_cpu)
        nc.set_inputs(1, xb_cpu, w_cpu)
    else:
        del w_cpu, x_cpu
    ttnn.synchronize_device(mesh)
    mark(f"upload end ({a.k * a.n * 2 / 2**20:.0f} MiB weight per chip), no compute yet")
    if nc:
        mark(f"numeric check on: matmul 1 (cold), then " + (f"{NUMERIC_CHECKS} spaced by time over the burst" if
             a.numeric_every > 0 else "none") + " and the last, up to "
             f"{a.numeric_budget_s:g}s of checks; bit-exact per chip vs the majority output, CPU slice "
             f"{a.numeric_cpu_rows} rows; a mismatch never stops the load")

    def mm(xx=None):
        return ttnn.matmul(x if xx is None else xx, w)

    hv, hflops, hdesc = heavy_op(a, ttnn, torch, mesh, x, w, cq1=cq1, fabric=a.topo != "none",
                                 cq_why=cq_why)   # the full-load op (lofi_all by default)
    ttnn.synchronize_device(mesh)
    mark(f"heavy op: {hdesc}; inputs uploaded, no compute yet")

    y, tm = None, 0
    for cyc in range(1, a.cycles + 1):   # --cycles: each cycle its own quiet and burst, one mesh open
        open_quiet(run, a.idle_s, cyc)
        run.set_phase(cyc_phase("burst", cyc), a.burst_s + a.burst_grace_s)
        mark(f"burst{cyc_tag(a, cyc)} begin (first matmul; trips usually come within ~1-20 s of here)")
        # 100% load. No readback or compare inside the burst (each used to be a ~1.6 s quiet
        # gap). The checked matmuls (alternating activations) go into the same stream and keep their outputs; they are read
        # back and compared after the burst's last synchronize. The heavy ops run in bursts of --burst-steps
        # with a --burst-gap-ms pause after each (BurstGap: its synchronize and sleep are the only ones inside the loop).
        hs = burst_host_stress(a, mark)   # a dev variant's host workers; None on the standard load
        t = run.burst_t0 = time.time()
        run.result["burst_t0"], run.result["cycle"] = round(t, 3), cyc   # the burst a loss falls in
        pend = []   # (op n, activation k, output, s into the burst): read back after the burst

        def checked(n, last=False):
            if nc and len(pend) < MAX_PENDING_CHECKS and not nc.off and \
                    check_due(n, last, time.time() - t, a.burst_s, len(pend), nc.every) and nc.want(n, last=True):
                k = (len(nc.checks) + len(pend)) % 2
                pend.append((n, k, mm(x if k == 0 else xb), time.time() - t))
                return True
            return False

        ra = RunAhead(ttnn, mesh, RUN_AHEAD_OPS, op_cap(a.burst_s, a.heavy, hflops, hv.ran["fallback"]))
        gap = BurstGap(ttnn, mesh, a.burst_steps, a.burst_gap_ms, ra)
        n, nh = 1, 0   # n: the op count for the check schedule; nh: heavy ops
        if not checked(1):   # op 1, cold: the numeric check's start-up case
            ttnn.deallocate(mm())
        more = True
        while more and time.time() - t < a.burst_s:
            y2 = hv(); n += 1; nh += 1
            if hv.ran["fallback"]:   # minimal_matmul refused: the cap at the fallback's rate (finding 1)
                ra.cap = min(ra.cap, op_cap(a.burst_s, a.heavy, hflops, hv.ran["fallback"]))
            more = ra.queued(nh)
            if y is not None:
                ttnn.deallocate(y)   # freed after the next op is queued: the queue never runs dry
            y = y2
            checked(n)
            gap.after(nh)
            if n in (2, 10) or n % 50 == 0: mark(f"op {n} queued {time.time() - t:.1f}s")
        checked(n + 1, last=True)
        ttnn.synchronize_device(mesh)   # the burst's only synchronize
        el = time.time() - t
        hs_n = hs.stop() if hs else 0
        ns = 1 + sum(1 for p in pend if p[0] != 1)   # the small matmuls: op 1 and the checked ones
        tflops = (nh * hflops + ns * 2 * a.m * a.k * a.n) / el / 1e12
        mark(f"burst{cyc_tag(a, cyc)} end {nh + ns} ops in {el:.1f}s, {gap.n} gaps ({gap.s:.1f}s paused) + one synchronize; run-ahead: {ra.desc()}")
        spent, skipped = read_back(nc, ttnn, pend, mark) if nc else (0.0, 0)
        ran = hv.ran["op"] + (f", fallback: {hv.ran['fallback']}" if hv.ran["fallback"] else "") + \
            (f", load fallback: {hv.ran['extras_fallback']}" if hv.ran.get("extras_fallback") else "")
        mark(f"LOAD burst{cyc_tag(a, cyc)}: {nh} x {hdesc} ({ran}) back to back + {ns} checked {a.m}x{a.k} @ {a.k}x{a.n} matmuls; "
             f"{el:.1f}s, {tflops:.0f} TFLOP/s per chip, {gap.desc()}"
             + (f"; numeric readback after it {spent:.1f}s, {skipped} skipped" if nc else "")
             + (f"; variant {a.variant}: host stress {hs_n} workers (started in {hs.start_s}s)" if hs else ""))
        run.result["burst"] = dict(matmuls=nh + ns, heavy_ops=nh, heavy=a.heavy, op=hv.ran["op"],
                                   fallback=hv.ran["fallback"], seconds=round(el, 2), t0=round(t, 3),
                                   load=hv.ran.get("load", a.heavy), load_fallback=hv.ran.get("extras_fallback"),
                                   gap=gap.desc(), gap_ms=(a.burst_gap_ms if gap.on() else 0), burst_steps=a.burst_steps,
                                   gaps=gap.n, gap_s=round(gap.s, 2),
                                   tflops_per_chip=round(tflops, 1), numeric_s=round(spent, 2), numeric_skipped=skipped,
                                   numeric_pending=len(pend),
                                   run_ahead=ra.desc(),
                                   **({"variant": a.variant, "host_stress": hs_n, "host_stress_start_s": hs.start_s} if hs else {}))
        tm += nh + ns
        if a.cycles > 1:
            run.result.setdefault("cycles", []).append(dict(run.result["burst"], cycle=cyc))
    tr["matmuls"] = n = tm
    if nc:
        numeric_report(run, nc)
    end_clean(run, (w, x) + ((y,) if y is not None else ()) + hv.tensors + ((xb,) if xb is not None else ()))


def burst_host_stress(a, mark):
    """FLEET_LOAD_VARIANT in a qb/step burst: its WARNING into the log; host workers started when host-stress is
    in it. None on the standard load."""
    if not getattr(a, "variant", ""):
        return None
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet"))
    import load_variant as lv
    vs = lv.parse(a.variant)
    a.variant = ",".join(vs)
    mark(lv.warning(vs))
    hs = lv.HostStress("host-stress" in vs)
    hs.start()
    return hs if hs.on else None


def numeric_report(run, nc):
    """The try's numeric result into the result JSON and one line per mismatched chip; never fails the try."""
    try:
        r = run.result["numeric"] = nc.result(run.ids, tray_of)
    except Exception as e:
        run.result["numeric"] = dict(schema=1, enabled=True, error=f"report: {type(e).__name__}: {str(e)[:200]}")
        mark(f"NUMERIC report error: {run.result['numeric']['error']}")
        return
    bad = r["mismatched_chips"]
    mark(f"NUMERIC try: {r['checks']} checks at matmuls {r['check_iters'][:6]}{'...' if r['checks'] > 6 else ''}, "
         f"{r['check_s']}s, map {r['map_source']}; {len(bad)}/{len(r['chips'])} chips mismatched"
         + (f"; check error: {r['error']}" if r.get("error") else ""))
    for i in bad:
        c = r["chips"][str(i)]
        mark(f"NUMERIC MISMATCH {c['pci']} (tray {c['tray'] or '?'}) chip {c['dev']}: {c['mismatched']}/{c['checks']} checks, "
             f"first at matmul {c['first_iter']}, max abs err {c['max_abs_err']}, NaN {c['nan']}, Inf {c['inf']}")


def allgather_burst(run, ttnn, torch, mesh, upload, tr):
    """--load allgather: upload the AG_VARIANTS inputs, run each once (warm-up: compile, halving), idle, then
    all_gather in turn for --burst-s. No return."""
    a, ids = run.a, run.ids
    run.set_phase("upload", a.upload_timeout_s)
    torch.manual_seed(0)
    var = [dict(name=nm, axis=ax, dim=d, shape=list(sh), x=upload(torch.randn(*sh, dtype=torch.bfloat16)),
                ops=0, gib_in=0.0, first_s=None) for nm, ax, d, sh in AG_VARIANTS]
    ttnn.synchronize_device(mesh)
    mark(f"upload end ({sum(nbytes(v['shape']) for v in var) / 2**20:.0f} MiB of all_gather inputs per chip), "
         "no CCL yet")
    cut = []

    def gather(v):
        ttnn_kw = dict(dim=v["dim"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        first = v["first_s"] is None
        t0 = time.monotonic()
        with run.lock:
            run.op = (v["name"], t0 + (a.ccl_first_op_timeout_s if first else a.ccl_op_timeout_s), t0)
        if v["axis"] is None:                       # 32-way: axis 1 (8-way), then axis 0 (4-way) on its output
            h = ttnn.all_gather(v["x"], cluster_axis=1, **ttnn_kw)
            y = ttnn.all_gather(h, cluster_axis=0, **ttnn_kw)
        else:
            h, y = None, ttnn.all_gather(v["x"], cluster_axis=v["axis"], **ttnn_kw)
        ttnn.synchronize_device(mesh)
        if run.holding:     # the op ended late, after the watchdog called it a hang: a slow op, not a stuck one
            run.result["op_completed_during_hold"] = dict(op=v["name"], utc=utc(), s=round(time.monotonic() - t0, 1))
            mark(f"all_gather {v['name']} completed {time.monotonic() - t0:.1f}s after it began, during the hold")
            threading.Event().wait()            # no more device calls; the hold path ends the run
        with run.lock:
            run.op = None
        for t in (h, y):
            if t is not None:
                ttnn.deallocate(t)
        v["ops"] += 1
        v["gib_in"] += nbytes(v["shape"]) * (AG_WAYS[v["axis"]] - 1) / 2**30
        if first:
            v["first_s"] = round(time.monotonic() - t0, 2)
            mark(f"first {ag_desc(v['name'], v['axis'], v['dim'], v['shape'])} end {v['first_s']}s")

    def first_call(v):
        """The variant's first call; out of DRAM with every chip present: halve it (twice at most), then drop it.
        Any other error with every chip present (an API check, a bad shape) drops it at once."""
        for tries in range(3):
            try:
                gather(v)
                return True
            except Exception as e:
                if [c for c in ids if not present(c)] or ("TIMEOUT" in str(e) and "hang" in str(e).lower()):
                    raise                                   # a loss or a hang: guarded() reports it, run.op still set
                with run.lock:
                    run.op = None
                err = f"{type(e).__name__}: {(str(e).strip().splitlines() or [''])[0][:200]}"
                ttnn.deallocate(v["x"])
                if tries == 2 or not re.search(r"out of memory|not enough space|alloc", str(e), re.I):
                    cut.append(f"{v['name']} dropped: {err}")
                    mark(f"all_gather {v['name']} dropped after {tries + 1} failed first call(s): {err}")
                    return False
                v["shape"][2] //= 2
                cut.append(f"{v['name']} halved to {v['shape']}: {err}")
                mark(f"all_gather {v['name']} failed ({err}); retrying at {v['shape']}")
                v["x"] = upload(torch.randn(*v["shape"], dtype=torch.bfloat16))
                ttnn.synchronize_device(mesh)
        return False

    # warm-up before the idle: each variant's first call compiles its kernels (and is halved if it fails), so the
    # burst after the idle is a sudden full load with no compile in its window. Limit: up to 3 first calls per
    # variant, each within --ccl-first-op-timeout-s, plus a re-upload.
    run.set_phase("warmup", 3 * len(var) * (a.ccl_first_op_timeout_s + 60))
    mark(f"warm-up begin: each all_gather variant once ({len(var)}), before the idle")
    var = [v for v in var if first_call(v)]
    if not var:
        raise RuntimeError("no all_gather variant ran: " + "; ".join(cut))
    for v in var:                          # the burst's counts start after the warm-up
        v["ops"], v["gib_in"] = 0, 0.0
    mark("warm-up end: " + " ".join(f"{v['name']}={v['first_s']}s" for v in var))
    open_quiet(run, a.idle_s)
    run.set_phase("burst", a.burst_s + a.burst_grace_s)
    mark("burst begin (all_gather on all 32 chips at once, every variant compiled in the warm-up)")
    t = run.burst_t0 = time.time()
    n = 0
    while time.time() - t < a.burst_s:
        gather(var[n % len(var)])
        n += 1
        if n in (1, 10) or n % 25 == 0:
            mark(f"all_gather {n} done {time.time() - t:.1f}s")
    el = time.time() - t
    gib = sum(v["gib_in"] for v in var)
    per = {v["name"]: dict(ops=v["ops"], cluster_axis=v["axis"], dim=v["dim"], in_shape=v["shape"],
                           in_mib=round(nbytes(v["shape"]) / 2**20), out_gib=round(nbytes(v["shape"]) *
                           AG_WAYS[v["axis"]] / 2**30, 2), first_s=v["first_s"]) for v in var}
    mark(f"burst end {n} all_gathers in {el:.1f}s ({gib:.0f} GiB into each chip, {gib / el:.1f} GiB/s per chip); "
         + " ".join(f"{k}={x['ops']}" for k, x in per.items()) + (f"; cut: {'; '.join(cut)}" if cut else ""))
    run.result["burst"] = dict(all_gathers=n, seconds=round(el, 2), gib_into_each_chip=round(gib, 1),
                               variants=per, cut=cut)
    tr["all_gathers"] = n
    mark(f"traffic: {n} all_gathers, 0 matmuls")
    end_clean(run, [v["x"] for v in var])


def end_clean(run, tensors):
    """The load ended with every chip present: free, close, re-check, PASS. Does not return."""
    ids, ttnn = run.ids, run.ttnn
    run.set_phase("post_burst")
    for v in tensors:
        ttnn.deallocate(v)
    run.close_quietly("end of run")
    lost = [c for c in ids if not present(c)]
    if lost:  # the watchdog normally catches this first
        run.finish("TRIP", EXIT_TRIP, lost_chips=lost,
                   first_loss=dict(utc=utc(), chips=lost, pci=[ids[c] for c in lost], phase="post_burst",
                                   trays=sorted({tray_of(ids[c]) for c in lost}, key=str)))
    mark(f"RESULT PASS 0/{len(ids)} chips lost")
    run.printed = True
    run.finish("NO_TRIP", EXIT_NO_TRIP)


if __name__ == "__main__":
    main()
