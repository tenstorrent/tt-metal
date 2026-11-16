#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The table under repro.sh busbar-confirm: one run dir's bus-bar HSC reads, per tray and feed.

usage: busbar_confirm.py <run dir> <run output file> [<try start epoch>]

Reads <run dir>/fleet/hsc/{busbar.peak, oc_limits, busbar.before, busbar.after}, <run dir>/fleet/busbar/parts.txt,
<run dir>/fleet/busbar/drop_s.txt (or any drop_s.txt under the run dir) and the run's own output (CPLD summary, box kW
lines). Prints, for information only:
  one row per tray and feed: peak-hold A (A6h, 1/16 A) against the warn (4Ah) and trip (46h) limits, STATUS_WORD
  before -> after, the power latches newly set (OFF+IOUT_OC, VIN_UV_WARN), the tray's CPLD state;
  the time to the drop from the start of the bus-bar load and into the heavy chunk;
  the box kW (PSU sum) peak and mean of each load chunk (mix, heavy; 'not read' for a heavy chunk that ended in a drop).
A peak over warn or trip is never a verdict here: healthy trays exceed warn under this load. The verdict is the drop
and the two latches, judged by repro.sh from run.sh's own findings. Missing files print as 'not read'."""
import glob
import os
import re
import sys


def rows(path):
    try:
        return [ln.rstrip("\n").split("\t") for ln in open(path, errors="replace") if ln.strip()]
    except OSError:
        return []


def num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def hexw(v):
    n = num(v)
    return "not read" if n is None else f"0x{n:04X}"


def newbits(sw0, si0, sw1, si1):
    out = []
    for name, a, b, mask in (("VIN_UV_WARN", si0, si1, 0x20), ("OFF", sw0, sw1, 0x40), ("IOUT_OC", sw0, sw1, 0x10)):
        if a is not None and b is not None and not a & mask and b & mask:
            out.append(name)
    return "+".join(out) or "none"


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    rd, outf = argv[1], argv[2]
    hs = os.path.join(rd, "fleet", "hsc")
    peaks = {(r[0], r[1]): num(r[2]) for r in rows(os.path.join(hs, "busbar.peak")) if len(r) >= 3}
    lims = {(r[0], r[1]): (num(r[2]), num(r[3])) for r in rows(os.path.join(hs, "oc_limits")) if len(r) >= 4}
    bef = {(r[0], r[1]): (num(r[2]), num(r[3])) for r in rows(os.path.join(hs, "busbar.before")) if len(r) >= 4}
    aft = {(r[0], r[1]): (num(r[2]), num(r[3])) for r in rows(os.path.join(hs, "busbar.after")) if len(r) >= 4}
    try:
        text = open(outf, errors="replace").read()
    except OSError:
        text = ""
    cpld = {}
    for m in re.finditer(
        r"SUMMARY cpld reg1e/dbg5f[^\n]*?: (tray \d [0-9a-f]{2} \w+(?:; tray \d [0-9a-f]{2} \w+)*)", text
    ):
        for t, _, st in re.findall(r"tray (\d) ([0-9a-f]{2}) (\w+)", m.group(1)):
            cpld[t] = st  # the last summary of the run
    print("confirm table (information only: a peak over warn or trip is never the verdict)")
    print(
        "  naming: only a drop or an OFF+IOUT_OC latch names a tray; any other latch alone names the box's bus bar set"
    )
    print("  tray feed  peak-hold A  warn A  trip A  STATUS_WORD before -> after   latch newly set      CPLD")
    keys = sorted(set(peaks) | set(lims) | set(bef) | set(aft))
    if not keys:
        print("  not read: no fleet/hsc busbar files in the run dir")
    for t, a in keys:
        p = peaks.get((t, a))
        trip, warn = lims.get((t, a), (None, None))
        sw0, si0 = bef.get((t, a), (None, None))
        sw1, si1 = aft.get((t, a), (None, None))
        print(
            "  %-4s %-5s %-12s %-7s %-7s %-6s -> %-6s             %-20s %s"
            % (
                t,
                a,
                "not read" if p is None else f"{p / 16:.1f}",
                warn if warn is not None else "not read",
                trip if trip is not None else "not read",
                hexw(sw0),
                hexw(sw1),
                newbits(sw0, si0, sw1, si1),
                cpld.get(t, "not read"),
            )
        )
    parts = []
    try:
        for ln in open(os.path.join(rd, "fleet", "busbar", "parts.txt"), errors="replace"):
            w = ln.split()
            if len(w) >= 2 and num(w[1]) is not None:
                parts.append((w[0], num(w[1])))
    except OSError:
        pass
    drop = None
    for f in glob.glob(os.path.join(rd, "**", "drop_s.txt"), recursive=True):
        w = open(f, errors="replace").read().split()
        try:
            drop = float(w[2]) if len(w) >= 3 and w[2] else float(w[0]) + float(w[1])
        except (ValueError, IndexError):
            continue
        break
    start = parts[0][1] if parts else (num(argv[3]) if len(argv) > 3 else None)
    heavy = next((e for n, e in parts if n == "heavy"), None)
    if drop is None:
        print("time to drop: no drop in this run")
    else:
        s = "not read" if start is None else f"{(drop - start) / 60:.1f} min from the start of the bus-bar load"
        if heavy is None:
            h = "the heavy chunk did not start"
        elif drop < heavy:
            h = "the drop came in the mix chunk, before the heavy chunk"
        else:
            h = f"{(drop - heavy) / 60:.1f} min into the heavy chunk"
        print(f"time to drop: {s}; {h}")
    kwre = re.compile(r"(box \(PSU0-3 sum\)|trays' sum \(capped \d+ W a tray\)) peak ([0-9.]+) kW mean ([0-9.]+) kW")
    # every PEAK chunk line, also one that says "box kW not read" (chunk_sched.peaks_line)
    kw = [(c, kwre.search(t)) for c, t in re.findall(r"PEAK chunk (\S+): ([^\n]*)", text)]
    # each chunk_load.py launch prints its own "PEAK chunk 1/1": the mix, then the heavy chunk. A chunk that ended in a drop
    # prints none, so the parts file names which chunk each line belongs to: by position when every chunk has a line, or
    # when the drop fell at or after the start of the first chunk without one; otherwise the lines keep their own numbers.
    lpp = [(n, e) for n, e in parts if n in ("mix", "heavy")]
    lp = [n for n, _ in lpp]
    named = bool(lp) and (len(kw) == len(lp) or (len(kw) < len(lp) and drop is not None and drop >= lpp[len(kw)][1]))

    def one(i, c, m):
        what = f"{lp[i]} chunk" if named else f"chunk {c}"
        if m is None:
            return f"{what} kW not read"
        return f"{what} ({'PSU sum' if m[1][0] == 'b' else 'trays sum, capped'}) peak {m[2]} mean {m[3]}"

    seen = [one(i, c, m) for i, (c, m) in enumerate(kw)]
    if named and heavy is not None and "heavy" not in lp[: len(kw)]:
        seen.append("heavy chunk kW not read (it ended before its PEAK chunk line)")
    if any(m is not None for _, m in kw):
        print("box kW per load chunk: " + "; ".join(seen))
    elif kw:
        print("box kW: not read (" + "; ".join(seen) + ")")
    elif named and heavy is not None:
        print("box kW: not read (no PEAK chunk line in the run output); heavy chunk kW not read")
    else:
        print("box kW: not read (no PEAK chunk line in the run output)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
