#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""bb_load.py: the bus-bar stage's load reached against its plan.

Usage: bb_load.py <busbar dir> <planned s> [<power.txt>]
Prints two lines:
  bus-bar load: X of Y min, mean A kW, peak B kW
  low | ok        (low: under FLEET_BB_LOAD_MIN_PCT, default 80, % of the planned minutes)

Load windows, from the files the parts leave under <busbar dir> (standard library only, no device):
  rows.jsonl (fl_c7: mix, heavy, long): each "chunk" row's start-end; a chunk with a "start" row and no "chunk" row (a loss
    or hang ended it) counts from its start to the file's last write.
  result.json (fl_bb_step: qb<i>, step; and the fallback's per-tray t<N>/result.json): burst t0 for burst seconds. A run.log
    with "burst begin" and no burst in result.json (the burst was ended) counts from that line to the log's last write.
Windows are merged, so the fallback's trays (all at once) count once. Box kW: the sum of power.txt columns 11-14 (PSU0-3 W)
on rows inside a window, where all four read; no row inside: every row; none read: "kW unread".
"""
import datetime
import json
import os
import re
import sys

BEGIN = re.compile(r"^(?:tray \d+: )?STRESS\|([0-9T:.\-]+)Z\|burst begin")


def utc_epoch(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()


def rows_windows(path):
    out, open_k, last = [], {}, os.path.getmtime(path)
    for ln in open(path, errors="replace"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        ev = r.get("event")
        if ev == "start" and "start" in r:
            open_k[r.get("k")] = float(r["start"])
        elif ev == "chunk" and "start" in r and "end" in r:
            open_k.pop(r.get("k"), None)
            out.append((float(r["start"]), float(r["end"])))
    out += [(s, max(s, last)) for s in open_k.values()]
    return out


def burst_window(d):
    try:
        b = json.load(open(os.path.join(d, "result.json"))).get("burst") or {}
        if "t0" in b and "seconds" in b:
            return (float(b["t0"]), float(b["t0"]) + float(b["seconds"]))
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ):  # TypeError: maxutil.py writes seconds null when t0 is unset
        pass
    lg = os.path.join(d, "run.log")
    try:
        for ln in open(lg, errors="replace"):
            m = BEGIN.match(ln)
            if m:
                t = utc_epoch(m.group(1))
                return (t, max(t, os.path.getmtime(lg)))
    except (OSError, ValueError):
        pass
    return None


def windows(top):
    w = []
    for d, subs, files in os.walk(top):
        subs[:] = [s for s in subs if s != "sync"]
        if "rows.jsonl" in files:
            w += rows_windows(os.path.join(d, "rows.jsonl"))
        elif "result.json" in files or "run.log" in files:
            b = burst_window(d)
            if b:
                w.append(b)
    w.sort()
    m = []
    for s, e in w:
        if m and s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m


def box_kw(path, w):
    allr, inw = [], []
    try:
        for ln in open(path, errors="replace"):
            f = ln.split()
            if len(f) < 14:
                continue
            try:
                t, kw = float(f[0]), sum(float(x) for x in f[10:14]) / 1000
            except ValueError:
                continue
            allr.append(kw)
            if any(s <= t <= e for s, e in w):
                inw.append(kw)
    except OSError:
        pass
    v = inw or allr
    return (sum(v) / len(v), max(v)) if v else None


def main(a):
    if len(a) < 2:
        sys.stderr.write(__doc__)
        return 2
    top, plan = a[0], float(a[1])
    w = windows(top)
    got = sum(e - s for s, e in w)
    kw = box_kw(a[2], w) if len(a) > 2 else None
    k = "mean %.1f kW, peak %.1f kW" % kw if kw else "kW unread"
    print("bus-bar load: %.1f of %.1f min, %s" % (got / 60, plan / 60, k))
    pct = float(os.environ.get("FLEET_BB_LOAD_MIN_PCT", "80"))
    print("low" if plan > 0 and got < plan * pct / 100 else "ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
