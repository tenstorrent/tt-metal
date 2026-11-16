#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/gddr_telem.py: per-chip GDDR telemetry for run.sh (lib/gddr.sh; one chip's case
# showed memory faults leave no trace in the run log). Read-only: it reads each chip's ARC telemetry table, never writes.
#   gddr_telem.py read [<bdf> ...] [--nr <bdf> ...]   one line per chip in BDF order, each written as it is read
#                             (pyluwen; the same fields and packing as tt-smi's tt_smi/utils.py decoders, firmware 19.x).
#                             The <bdf>s hung or were lost in this run: never opened (an MMIO or ARC read of a hung chip
#                             can give a fatal MCE, lib/mce_guard.sh); nor the --nr chips (firmware not ready, FW_NR) nor
#                             a chip whose config space reads ffffffff. Each chip's BDF comes from sysfs before it is
#                             opened; with a skip or --nr list and no sysfs map, no chip is opened. Each chip is read in
#                             a child process cut at TRAY_GDDR_CHIP_S (10 s; 0: in this process, no limit):
#     <bdf> max_temp=<C> temp=<c0>,..,<c7> corr_rd=<n0>,..,<n7> corr_wr=<n0>,..,<n7> uncorr=0x<mask>
#       temp: per GDDR channel 0-7 the hotter of its two dies (GDDR_x_y_TEMP, a byte per die)
#       corr_rd/corr_wr: corrected read/write EDC counts per channel (GDDR_x_y_CORR_ERRS, a byte each, saturates at 255)
#       uncorr: GDDR_UNCORR_ERRS, bit 2n = channel n read, bit 2n+1 = channel n write uncorrected error seen
#     A field the firmware does not give is '-'; a chip that cannot be read or is skipped: "<bdf> unavailable <why>"; no pyluwen at
#     all: one line "unavailable no pyluwen: <why>". Exit 0 in every case: telemetry never fails a run.
#   gddr_telem.py diff <start file> <end file>    (plain python, no pyluwen) the summary lines run.sh prints:
#     GDDR <bdf> (tray N): corr +rd/+wr (by channel), new uncorr, max temp start -> end C   one per chip
#     GDDR WARN <bdf> (tray N): new uncorrected error(s) on GDDR 3 read ...                   any new uncorr flag
#     GDDR WARN <bdf> (tray N): corrected errors rose on GDDR 3 (read +5) ...                 any corrected-count rise
#     GDDR: <n> chips read at start and end; ...                                              one overall line, last
#   A counter lower at the end than at the start was cleared by a reset in the run: the end value counts as the rise.
import os
import select
import signal
import sys
import time

N = 8  # Blackhole GDDR channels
FIELDS = ["max_temp", "temp", "corr_rd", "corr_wr", "uncorr"]


def tray(bdf):  # tray 1-4 from the PCI bus byte, as lib/tdp.sh's tdp_tray and run.sh's rk_tray
    try:
        bus = int(bdf.split(":")[-2], 16)
    except (ValueError, IndexError):
        return "?"
    return {0x00: "1", 0x40: "2", 0xC0: "3", 0x80: "4"}.get(bus & 0xF0, "?")


def decode(t):
    """A pyluwen Telemetry (or any object with its attribute names) -> the read line's fields; '-' where missing."""

    def get(name):
        v = getattr(t, name, None)
        return v if isinstance(v, int) else None

    out = {}
    m = get("max_gddr_temp")
    out["max_temp"] = str(m) if m is not None else "-"
    temp, rd, wr = [], [], []
    for p in range(N // 2):
        tw = get(f"gddr{2 * p}{2 * p + 1}_temp")
        cw = get(f"gddr{2 * p}{2 * p + 1}_corr_errs")
        for ch in range(2):
            sh = 16 * ch
            temp.append("-" if tw is None else str(max((tw >> sh) & 0xFF, (tw >> (sh + 8)) & 0xFF)))
            rd.append("-" if cw is None else str((cw >> sh) & 0xFF))
            wr.append("-" if cw is None else str((cw >> (sh + 8)) & 0xFF))
    out["temp"], out["corr_rd"], out["corr_wr"] = ",".join(temp), ",".join(rd), ",".join(wr)
    u = get("gddr_uncorr_errs")
    out["uncorr"] = "-" if u is None else f"0x{u:04x}"
    return out


def line(bdf, f):
    return bdf + " " + " ".join(f"{k}={f[k]}" for k in FIELDS)


def short_bdf(bdf):
    return bdf[5:] if bdf.count(":") == 2 else bdf


def devices(sysfs=None):
    """[(index, bdf)] of the /dev/tenstorrent/<index> chips from sysfs, without opening any; None when sysfs has no map."""
    sysfs = sysfs or os.environ.get("TRAY_GDDR_SYS", "/sys")
    out, d = [], os.path.join(sysfs, "class", "tenstorrent")
    try:
        names = os.listdir(d)
    except OSError:
        return None
    for n in names:  # tt-kmd's class devices: tenstorrent!<index> -> device -> the PCI function
        i = n.rsplit("!", 1)[-1]
        if i.isdigit():
            out.append((int(i), short_bdf(os.path.basename(os.path.realpath(os.path.join(d, n, "device"))))))
    return sorted(out) or None


def config_dead(bdf, sysfs=None):  # vendor id ffff: the chip fell off the bus (lspci rev ff)
    sysfs = sysfs or os.environ.get("TRAY_GDDR_SYS", "/sys")
    try:
        with open(os.path.join(sysfs, "bus", "pci", "devices", "0000:" + bdf, "config"), "rb") as f:
            return f.read(4) == b"\xff\xff\xff\xff"
    except OSError:
        return False


def chip_line(pyluwen, i, bdf):
    """Open chip <i> and read its line; '<bdf> unavailable <why>' when it cannot be read."""
    try:
        c = pyluwen.PciChip(i)
        bdf = short_bdf(c.get_pci_bdf())
        return line(bdf, decode(c.get_telemetry()))
    except Exception as e:
        return f"{bdf} unavailable {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}"


def chip_line_timed(pyluwen, i, bdf, limit):
    """chip_line in a child process cut at <limit> s: a chip that hangs in the open or the read costs its own
    line, not the rest of the read. The child is killed and reaped without waiting (a child stuck in the kernel is left).
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: one line to the pipe, then out without the parent's exit handlers
        os.close(r)
        try:
            os.write(w, (chip_line(pyluwen, i, bdf) + "\n").encode())
        finally:
            os._exit(0)
    os.close(w)
    buf, end, eof = b"", time.monotonic() + limit, False
    with os.fdopen(r, "rb", buffering=0) as f:
        while not buf.endswith(b"\n"):
            left = end - time.monotonic()
            if left <= 0 or not select.select([f], [], [], left)[0]:
                break
            b = f.read(4096)
            if not b:
                eof = True
                break
            buf += b
    if buf.endswith(b"\n") or eof:
        os.waitpid(pid, 0)
        return buf.decode(errors="replace").rstrip("\n") or f"{bdf} unavailable the read process ended without a line"
    try:
        os.kill(pid, signal.SIGKILL)
        for _ in range(10):  # at most 1 s: a child in uninterruptible sleep is not waited for
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
            time.sleep(0.1)
    except OSError:
        pass
    return f"{bdf} unavailable no answer in {limit:g} s (TRAY_GDDR_CHIP_S)"


def read(skip=(), nr=()):
    skip = {short_bdf(b if b.count(":") == 2 else "0000:" + b) for b in skip if b}
    nr = {short_bdf(b if b.count(":") == 2 else "0000:" + b) for b in nr if b}
    try:
        import pyluwen
    except Exception as e:  # bundle boxes without PyPI: no pyluwen
        print(f"unavailable no pyluwen: {e}", flush=True)
        return
    devs = devices()
    if devs is None and (skip or nr):  # which index is a hung or not-ready chip is unknown: open none
        print(
            f"unavailable no sysfs map of /dev/tenstorrent to PCI addresses, and "
            f"{', '.join(sorted(skip | nr))} {'hung in this run' if skip else 'firmware not ready'}: no chip opened"
            f"{' before the end reset' if skip else ''}",
            flush=True,
        )
        return
    if devs is None:
        try:
            devs = [(i, None) for i in pyluwen.pci_scan()]
        except Exception as e:
            print(f"unavailable pci_scan failed: {e}", flush=True)
            return
    try:
        limit = float(os.environ.get("TRAY_GDDR_CHIP_S", "10"))
    except ValueError:
        limit = 10.0
    # One chip at a time, in BDF order, each line written as it is read: a read cut by gd_read's
    # TRAY_GDDR_S keeps every earlier chip's line, and a chip with no answer in TRAY_GDDR_CHIP_S loses only its own.
    for i, bdf in sorted(devs, key=lambda d: (d[1] is None, d[1] or "", d[0])):
        if bdf in skip:
            l = f"{bdf} unavailable hung in this run: not opened before the end reset"
        elif bdf in nr:  # FW_NR (lib/fw_check.sh): UMD waits 300 s on such a chip; never opened
            l = f"{bdf} unavailable firmware not ready: not opened"
        elif bdf and config_dead(bdf):
            l = f"{bdf} unavailable config space reads ffffffff: not opened"
        elif limit > 0:
            l = chip_line_timed(pyluwen, i, bdf or f"pci{i}", limit)
        else:  # TRAY_GDDR_CHIP_S=0: in this process, no per-chip limit
            l = chip_line(pyluwen, i, bdf or f"pci{i}")
        print(l, flush=True)


def parse(path):
    """A read file -> {bdf: fields}, with None for an unavailable chip; ({}, why) when nothing was read."""
    chips, why = {}, ""
    try:
        text = open(path).read()
    except OSError as e:
        return {}, f"no file ({e.strerror})"
    for l in text.splitlines():
        w = l.split()
        if not w:
            continue
        if w[0] == "unavailable":
            why = " ".join(w[1:]) or "unavailable"
            continue
        if len(w) > 1 and w[1] == "unavailable":
            chips[w[0]] = None
            continue
        f = dict(x.split("=", 1) for x in w[1:] if "=" in x)
        chips[w[0]] = f if all(k in f for k in FIELDS) else None
    return chips, (why if not chips else "")


def ints(s):
    return [None if x == "-" else int(x) for x in s.split(",")] if s else [None] * N


def rise(a, b):  # the counter's rise; a reset in the run clears it (b < a): then all of b is new
    if a is None or b is None:
        return None
    return b - a if b >= a else b


def diff(start_path, end_path):
    s, swhy = parse(start_path)
    e, ewhy = parse(end_path)
    if not s or not e:
        print(
            "GDDR: telemetry unavailable ("
            + "; ".join(x for x in (f"start: {swhy}" if not s else "", f"end: {ewhy}" if not e else "") if x)
            + "); no GDDR error check this run"
        )
        return
    per, warns, unread, sat, ncmp = [], [], [], [], 0
    hot = None
    for bdf in sorted(set(s) | set(e)):
        a, b = s.get(bdf), e.get(bdf)
        tag = f"{bdf} (tray {tray(bdf)})"
        if a is None or b is None:
            unread.append(tag + (" at start" if a is None else "") + (" at end" if b is None and a is not None else ""))
            continue
        ncmp += 1
        rd = [rise(x, y) for x, y in zip(ints(a["corr_rd"]), ints(b["corr_rd"]))]
        wr = [rise(x, y) for x, y in zip(ints(a["corr_wr"]), ints(b["corr_wr"]))]
        ua = None if a["uncorr"] == "-" else int(a["uncorr"], 16)
        ub = None if b["uncorr"] == "-" else int(b["uncorr"], 16)
        new_u = (ub & ~ua) if ua is not None and ub is not None else None
        if any(x == 255 for x in ints(b["corr_rd"]) + ints(b["corr_wr"])):
            sat.append(tag)
        mt = None if b["max_temp"] == "-" else int(b["max_temp"])
        if mt is not None and (hot is None or mt > hot[0]):
            hot = (mt, tag)
        cr = ",".join("-" if x is None else f"+{x}" for x in rd)
        cw = ",".join("-" if x is None else f"+{x}" for x in wr)
        nu = "-" if new_u is None else ("none" if new_u == 0 else f"0x{new_u:04x}")
        per.append(
            f"GDDR {tag}: corr rd {cr} wr {cw} (GDDR 0-7); new uncorr {nu}; uncorr at start "
            f"{a['uncorr']}; max temp {a['max_temp']} -> {b['max_temp']} C"
        )
        if new_u:
            ch = [
                f"GDDR {c} {'read' if bit == 0 else 'write'}"
                for c in range(N)
                for bit in (0, 1)
                if new_u >> (2 * c + bit) & 1
            ]
            warns.append(f"GDDR WARN {tag}: new uncorrected error(s) on {', '.join(ch)} during the run")
        up = [
            f"GDDR {c} ({', '.join(x for x in (f'read +{rd[c]}' if rd[c] else '', f'write +{wr[c]}' if wr[c] else '') if x)})"
            for c in range(N)
            if rd[c] or wr[c]
        ]
        if up:
            warns.append(f"GDDR WARN {tag}: corrected errors rose on {'; '.join(up)}")
    for l in per + warns:
        print(l)
    tail = [f"GDDR: {ncmp} chips read at start and end"]
    nu = sum(1 for w in warns if "new uncorrected" in w)
    nc = sum(1 for w in warns if "corrected errors rose" in w)
    tail.append(f"{nu} with new uncorrected errors" if nu else "no new uncorrected errors")
    tail.append(f"{nc} with a corrected-error rise" if nc else "no corrected-error rise")
    if hot:
        tail.append(f"max GDDR temp {hot[0]} C on {hot[1]}")
    if sat:
        tail.append(f"corrected counter at its 255 cap on {', '.join(sat)} (a rise there may be undercounted)")
    if unread:
        tail.append(f"not read: {', '.join(unread)}")
    print("; ".join(tail))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "read":
        a = sys.argv[2:]
        read(a[: a.index("--nr")] if "--nr" in a else a, a[a.index("--nr") + 1 :] if "--nr" in a else ())
    elif len(sys.argv) == 4 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: gddr_telem.py read [<skip bdf> ...] [--nr <bdf> ...] | diff <start file> <end file>")
