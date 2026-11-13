#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tray_12v.py: judges the per-second 12 V log of one run.sh run.

The fleet launcher polls the host's BMC SDR from another machine (tools/tray-power-poll, read-only, no BMC write) for the
length of the run and merges its power.csv into the run's fleet/ dir. This flags, by tray 1-4 with its BDFs:
  - a tray whose CPLD vr_pg is not 0xff inside a load window (load_times.tsv);
  - a tray whose Power_UBBn is <= COLLAPSE_W while every other tray reads >= HOLD_W (a 12 V collapse), even if it
    recovers.
Samples within MASK_S of a reset (C2, C9 or a recovery reset; reset_times.tsv) are never flagged: resets give partial
vr_pg episodes on healthy trays. err/na rows are ignored. The poller's
lspci count is never used for chip loss (132 vr_pg 00 episodes kept lspci at 8).
Standard library only (tests/test_tray_12v.py).

  tray_12v.py --csv power.csv [--resets reset_times.tsv] [--loads load_times.tsv] [--map map.txt]
reset_times.tsv / load_times.tsv rows: <epoch>\\tbegin|end\\t<label>. map.txt rows: <bdf> <chip>.
Prints one '12V FAIL tray N (...)' line per flag, then one '12V log: ...' summary line. Exit 0 always.
"""
import argparse
import csv
import time

MASK_S = 40.0  # +-40 s around every reset
COLLAPSE_W = 50.0  # a tray at or below this ...
HOLD_W = 600.0  # ... while every other tray holds at least this
NEAR_S = 3.0  # other trays' samples within this of the collapse sample


def tray_of(bdf):
    """run.sh rk_tray: bus & 0xF0 -> 00 tray 1, 40 tray 2, c0 tray 3, 80 tray 4."""
    try:
        bus = int(bdf.split(":")[-2], 16) & 0xF0
    except (ValueError, IndexError):
        return 0
    return {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}.get(bus, 0)


def windows(path, open_end):
    """[(start, end)] from begin/end rows; an unclosed begin runs to open_end."""
    out, start = [], None
    try:
        with open(path) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 2:
                    continue
                try:
                    t = float(p[0])
                except ValueError:
                    continue
                if p[1] == "begin":
                    start = t if start is None else start
                elif p[1] == "end" and start is not None:
                    out.append((start, t))
                    start = None
    except OSError:
        return out
    if start is not None:
        out.append((start, open_end))
    return out


def inside(t, wins, pad=0.0):
    return any(a - pad <= t <= b + pad for a, b in wins)


def hms(t):
    return time.strftime("%H:%M:%S", time.gmtime(t)) + " UTC"


def read_csv(path):
    vr, ubb = [], []  # (epoch, tray, state), (epoch, tray, watts)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if (r.get("flags") or "").strip() in ("err", "na") or not (r.get("value") or "").strip():
                continue
            try:
                t, tray = float(r["epoch"]), int(r["tray"])
            except (KeyError, TypeError, ValueError):
                continue
            s = r.get("sensor") or ""
            if s.endswith("_vr_pg"):
                st = (r.get("state") or "").strip().lower()
                if st:
                    vr.append((t, tray, st))
            elif s.startswith("Power_UBB"):
                try:
                    ubb.append((t, int(s[len("Power_UBB") :]) + 1, float(r["value"])))
                except ValueError:
                    continue
    return vr, ubb


def judge(csv_path, resets_path=None, loads_path=None, map_path=None):
    vr, ubb = read_csv(csv_path)
    end = max([x[0] for x in vr + ubb] or [0.0])
    resets = windows(resets_path, end) if resets_path else []
    loads = windows(loads_path, end) if loads_path else []
    bdfs = {}
    if map_path:
        try:
            with open(map_path) as fh:
                for line in fh:
                    p = line.split()
                    if p and tray_of(p[0]):
                        bdfs.setdefault(tray_of(p[0]), []).append(p[0])
        except OSError:
            pass

    def name(u):
        b = sorted(bdfs.get(u, []))
        return "tray %d (%s)" % (u, ", ".join(b) if b else "BDFs not in map.txt")

    lines = []
    glitch = {}
    for t, u, st in vr:
        if st != "0xff" and inside(t, loads) and not inside(t, resets, MASK_S):
            glitch.setdefault(u, []).append((t, st))
    for u in sorted(glitch):
        ev = glitch[u]
        states = ",".join(sorted({s for _, s in ev}))
        lines.append(
            "12V FAIL %s: CPLD vr_pg %s under load (%d sample(s), first %s, last %s; not within %d s of a reset)"
            % (name(u), states, len(ev), hms(ev[0][0]), hms(ev[-1][0]), MASK_S)
        )
    by_tray = {}
    for t, u, w in ubb:
        by_tray.setdefault(u, []).append((t, w))
    coll = {}
    for t, u, w in ubb:
        if w > COLLAPSE_W or inside(t, resets, MASK_S):
            continue
        others = [o for o in by_tray if o != u]
        if not others:
            continue
        held = []
        for o in others:
            near = [x for x in by_tray[o] if abs(x[0] - t) <= NEAR_S]
            if not near:
                break
            ow = min(near, key=lambda x: abs(x[0] - t))[1]
            if ow < HOLD_W:
                break
            held.append(ow)
        else:
            coll.setdefault(u, []).append((t, w, min(held)))
    for u in sorted(coll):
        ev = coll[u]
        lines.append(
            "12V FAIL %s: Power_UBB%d %d W while the other trays held >= %d W (min %d W; %d sample(s), first %s, "
            "last %s; a 12 V collapse, flagged even if it recovered)"
            % (
                name(u),
                u - 1,
                min(w for _, w, _ in ev),
                HOLD_W,
                min(h for _, _, h in ev),
                len(ev),
                hms(ev[0][0]),
                hms(ev[-1][0]),
            )
        )
    n = len({(t, u) for t, u, _ in ubb})
    lines.append(
        "12V log: %d Power_UBB and %d vr_pg samples, %d load window(s), %d reset(s) masked +-%d s; %s"
        % (
            n,
            len(vr),
            len(loads),
            len(resets),
            MASK_S,
            "%d tray(s) flagged" % len(set(glitch) | set(coll)) if (glitch or coll) else "no tray flagged",
        )
    )
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--resets")
    ap.add_argument("--loads")
    ap.add_argument("--map")
    a = ap.parse_args()
    try:
        out = judge(a.csv, a.resets, a.loads, a.map)
    except OSError as e:
        out = ["12V log: power.csv unreadable (%s)" % e]
    print("\n".join(out))


if __name__ == "__main__":
    main()
