#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""cpld_5f.py: the tray CPLDs' reg 0x1E after a loss, decoded, so a whole-tray drop names its CPLD state before any
reset can change it. run.sh calls it after a loss, just before their reset and re-check (which it leaves as it is).
Reads only: the debug registers 0x5F and 0x58 need a CPLD index write first, so they are not read and
each line says 'dbg5f/dbg58 not read: needs a CPLD index write (not allowed)'. --parse still decodes them in old captures.

  cpld_5f.py [--out DIR] [--ipmitool CMD] [--dry-run]   read the 4 tray CPLDs; write DIR/cpld_reg1e_dbg5f.txt and
                                                        .json; print the per-tray lines and one SUMMARY line
  cpld_5f.py --parse FILE...                            decode earlier captures: this script's file or run_repro.sh's
                                                        cpld_dump.txt (reg0x1E=.. dbg0x5F=..); no reads

Reads (in-band, as run_repro.sh's cpld_dump and the trays project's cpld_verdict.py): tray CPLD at 0x80 on bus
tray 1 0x11, tray 2 0x13, tray 3 0x0D, tray 4 0x0F; reg 0x1E = raw 0x06 0x52 <bus> 0x80 0x01 0x1E (a 1-byte read).
No write of any kind: no CPLD index write, no power or reset action. Each value is read until two reads in a row
agree (another reader may race on the bus), at most 4 reads, 15 s per call, 90 s in all.

Live decode, reg 0x1E alone (ff = all chips alive): ff ALIVE, 00 TRAY_OFF (no chip alive: the tray is off; the CPLD
cannot say why without the debug registers), other CHIPS_DOWN, no stable read UNREAD.

--parse decode of old captures with reg1e/dbg5f (dbg 0x5F b3 = 12 V hot-swap A power good, b2 = B; dbg 0x58 b7 power enable, b6 brake_n,
b3 PSU-OC_n (0 = over-current warning), b1 3V3 PG; healthy reg1e/dbg5f ff/7c, dbg58 fa):
  ff/7c        HEALTHY      the tray's rails are good (a lost chip on it is a chip/board fault, not the 12 V feed)
  00/78        BUSBAR_SIG   12 V-B hot-swap PG low, tray off: the busbar signature (10/10 drops on one tray of
                            one box). With dbg58 power enable on and no
                            over-current it is the full busbar pattern
  other 12 V   TRAY_12V_LOST a hot-swap PG low (rail A: 00/74 was seen on another box), tray off
  any 12 V low with an over-current warning (dbg58 b3 = 0): HOTSWAP_OC, the tray's hot-swap tripped on its load
  not ff, 7c   CHIPS_DOWN_12V_GOOD  chips dead with the 12 V good: not the 12 V feed
  else         OTHER / UNREAD (raw values kept)
The CPLD cannot tell a hot-swap controller on the tray that trips without an over-current from a sagging 12 V feed
into it (trays analysis): BUSBAR_SIG says "matches the busbar fault"; a tray/slot swap proves it.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

TRAY_BUS = {1: 0x11, 2: 0x13, 3: 0x0D, 4: 0x0F}      # tray N = UBB N-1 (ub0_repro.py CPLD_BUS)
HEALTHY, BUSBAR = ("ff", "7c"), ("00", "78")
REF = "healthy ff/7c, busbar 00/78"
CALL_S, DEADLINE_S, MAX_READS = 15, 90, 4
NOT_READ = "dbg5f/dbg58 not read: needs a CPLD index write (not allowed)"
LIVE_REF = "reg1e ff = all chips alive; " + NOT_READ


def bits(v, *b):
    return [None if v is None else bool(int(v, 16) >> i & 1) for i in b]


def decode(reg1e, dbg5f, dbg58=None):
    """(state, text) for one tray's reg 0x1E / dbg 0x5F (/ dbg 0x58) as two-digit hex strings or None."""
    if reg1e is None:
        return "UNREAD", "no stable read"
    if dbg5f is None and dbg58 is not None:   # an old capture with dbg 0x58 but no dbg 0x5F
        return "UNREAD", "no stable read"
    if dbg5f is None:          # live read: reg 0x1E only
        if reg1e == "ff":
            return "ALIVE", "all chips alive; " + NOT_READ
        if reg1e == "00":
            return "TRAY_OFF", "no chip alive, the tray is off; " + NOT_READ
        return "CHIPS_DOWN", f"alive {reg1e}, not every chip alive; " + NOT_READ
    hs_a, hs_b = bits(dbg5f, 3, 2)
    en, brake_n, oc_n = bits(dbg58, 7, 6, 3)
    low = [r for r, ok in (("A", hs_a), ("B", hs_b)) if not ok]
    extra = []
    if dbg58 is not None:
        extra.append("power enable " + ("on" if en else "OFF"))
        extra.append("no over-current" if oc_n else "OVER-CURRENT warning")
        if not brake_n:
            extra.append("power brake on")
    tail = f" ({', '.join(extra)})" if extra else ""
    if (reg1e, dbg5f) == HEALTHY:
        return "HEALTHY", "rails good"
    if low and dbg58 is not None and not oc_n:
        return "HOTSWAP_OC", f"12 V-{'/'.join(low)} hot-swap PG low with an over-current warning: the tray's hot-swap tripped on its load{tail}"
    if (reg1e, dbg5f) == BUSBAR:
        full = dbg58 is not None and en and oc_n
        return "BUSBAR_SIG", ("12 V-B hot-swap PG low, tray off: matches the busbar signature"
                              + (" in full (power enable on, no over-current)" if full else tail))
    if low and reg1e != "ff":
        return "TRAY_12V_LOST", f"12 V-{'/'.join(low)} hot-swap PG low, tray off; not the 00/78 busbar reading{tail}"
    if dbg5f == HEALTHY[1] and reg1e != "ff":
        return "CHIPS_DOWN_12V_GOOD", f"alive {reg1e} with the 12 V good: chip/board, not the 12 V feed{tail}"
    return "OTHER", f"unexpected reading{tail}"


def parse_raw(out):
    """ipmitool raw output -> the one byte it read ('7c'), or None."""
    t = out.split()
    return t[0].lower() if len(t) == 1 and re.fullmatch(r"[0-9a-fA-F]{2}", t[0]) else None


class Reader:
    def __init__(self, ipmitool, dry):
        self.cmd, self.dry, self.t_end = shlex.split(ipmitool), dry, time.monotonic() + DEADLINE_S

    def call(self, *a):
        argv = self.cmd + ["raw", "0x06", "0x52", *a]
        if self.dry:
            print("+ " + " ".join(argv))
            return None
        left = self.t_end - time.monotonic()
        if left <= 0:
            return None
        try:
            return subprocess.run(argv, capture_output=True, text=True, timeout=min(CALL_S, left)).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    def stable(self, read):
        """read() until two in a row agree: (value or None, number of reads)."""
        prev, n = None, 0
        while n < (1 if self.dry else MAX_READS):
            v = read(); n += 1
            if v is not None and v == prev:
                return v, n
            prev = v
        return None, n

    def reg(self, bus, r):
        return self.stable(lambda: parse_raw(self.call(f"0x{bus:02X}", "0x80", "0x01", f"0x{r:02X}") or ""))


def row(tray, reg1e, dbg5f, dbg58, reads=None):
    state, text = decode(reg1e, dbg5f, dbg58)
    return dict(tray=tray, ubb=tray - 1, bus=f"0x{TRAY_BUS[tray]:02X}", reg1e=reg1e, dbg5f=dbg5f, dbg58=dbg58,
                state=state, text=text, reads=reads)


def line(r):
    return (f"tray {r['tray']} bus {r['bus']}: reg1e={r['reg1e'] or '??'} dbg5f={r['dbg5f'] or '??'} "
            f"dbg58={r['dbg58'] or '??'} -> {r['state']}: {r['text']}")


def summary(rows, label="cpld reg1e/dbg5f"):
    if all(r["dbg5f"] is None for r in rows):      # live read: reg 0x1E only
        parts = [f"tray {r['tray']} {r['reg1e'] or '??'} {r['state']}" for r in rows]
        return f"SUMMARY {label} ({LIVE_REF}): " + "; ".join(parts)
    parts = [f"tray {r['tray']} {r['reg1e'] or '??'}/{r['dbg5f'] or '??'} {r['state']}" for r in rows]
    return f"SUMMARY {label} ({REF}): " + "; ".join(parts)


def parse_file(path):
    """Rows from this script's file or run_repro.sh's cpld_dump.txt (the last reading of each tray)."""
    by_bus = {v: k for k, v in TRAY_BUS.items()}
    got = {}
    with open(path, errors="replace") as f:
        text = f.read().splitlines()
    for ln in text:
        m = re.search(r"bus (0x[0-9A-Fa-f]{2})", ln)
        if not m or ln.startswith("#") or ln.startswith("PDB"):
            continue
        tray = by_bus.get(int(m.group(1), 16))
        if tray is None:
            continue
        val = {}
        for key, pat in (("reg1e", r"reg(?:0x)?1[eE]=([0-9a-fA-F]{2})\b"), ("dbg5f", r"dbg(?:0x)?5[fF]=([0-9a-fA-F]{2})\b"),
                         ("dbg58", r"dbg(?:0x)?58=([0-9a-fA-F]{2})\b")):
            v = re.search(pat, ln)
            val[key] = v.group(1).lower() if v else None
        got[tray] = row(tray, val["reg1e"], val["dbg5f"], val["dbg58"])
    return [got[t] for t in sorted(got)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="write cpld_reg1e_dbg5f.txt and .json here")
    p.add_argument("--ipmitool", help="in-band ipmitool command (default: ipmitool as root, else sudo -n ipmitool)")
    p.add_argument("--dry-run", action="store_true", help="print the reads; touch nothing")
    p.add_argument("--parse", nargs="+", metavar="FILE", help="decode earlier captures instead of reading")
    p.add_argument("--trays", help="read only these trays, comma-separated (e.g. 2,4; default: all 4)")
    a = p.parse_args()
    if a.parse:
        for f in a.parse:
            rows = parse_file(f)
            print(f"# {f}")
            for r in rows:
                print(line(r))
            print(summary(rows) if rows else "SUMMARY cpld reg1e/dbg5f: no tray rows in the file")
        return 0
    ipmi = a.ipmitool or ("ipmitool" if os.geteuid() == 0 else "sudo -n ipmitool")
    rd = Reader(ipmi, a.dry_run)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.dry_run:
        print(f"dry run: the reads cpld_5f.py makes after a loss, before the reset (writes {a.out or '.'}/cpld_reg1e_dbg5f.txt)")
    want = TRAY_BUS
    if a.trays:
        try:
            want = {t: TRAY_BUS[t] for t in sorted({int(x) for x in a.trays.split(",") if x.strip()})}
        except (ValueError, KeyError):
            p.error(f"--trays {a.trays!r}: trays are 1-4, comma-separated")
    # each tray's line is printed and flushed as soon as it is read: a caller's timeout (a slow BMC) keeps what was read
    text = [f"# {ts} tray CPLD reg 0x1E after a loss, before the reset, reads only ({LIVE_REF}); decode: cpld_5f.py --help"]
    if not a.dry_run:
        print(text[0], flush=True)
    rows = []
    for tray, bus in want.items():
        r1e, n1 = rd.reg(bus, 0x1E)
        rows.append(row(tray, r1e, None, None, reads=[n1]))
        if not a.dry_run:
            print(line(rows[-1]), flush=True)
    if a.dry_run:
        print(f"then one line per tray and: SUMMARY cpld reg1e/dbg5f ({LIVE_REF}): tray 1 .. <state>; ...")
        return 0
    text += [line(r) for r in rows] + [summary(rows)]
    print(text[-1], flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "cpld_reg1e_dbg5f.txt"), "w") as f:
            f.write("\n".join(text) + "\n")
        with open(os.path.join(a.out, "cpld_reg1e_dbg5f.json"), "w") as f:
            json.dump(dict(ts=ts, ipmitool=ipmi, rows=rows), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
