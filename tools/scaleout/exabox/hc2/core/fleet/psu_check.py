#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The box's PSUs from one in-band read (lib/psu_check.sh reads, this judges).

  psu_check.py <when> <sdr.txt> <watts.txt> [<power.txt>]

sdr.txt: `ipmitool sdr type 'Power Supply'` (PSU<n>, PSU<n>_Status, PSU<n>_Input, PSU_Redundancy rows).
watts.txt: `ipmitool sensor reading Power_PSU<n> ...` ("Power_PSU0 | 900"). power.txt: the stage's power poll
(lib/fleet.sh fl_power_start; columns 11-14 are Power_PSU0-3 W), for the stage's peak per PSU and box.
Prints, in this order:
  PSU <when>: PSU0 900 W, PSU2 0 W (absent), ...; box 2.76 kW (PSU sum)[; stage peak PSU0 3600 W, ..., box 12.80 kW]
  PSU FINDING PSU<n>: <sensor> <state>, ... (<when>)        a PSU Device Absent, Critical or failed
  PSU FINDING redundancy lost: PSU_Redundancy <state> (PSU<n>) (<when>)
  PSU WARN PSU<n> at <W> W, <pct>% of its <rated> W rating (<when>)   above PSU_WARN_PCT (90) % of PSU_RATED_W (4000)
  PSU WARN box at <kW> kW, above 14 kW (<when>)                       above PSU_BOX_WARN_KW (14): a flag only
  PSU WARN redundancy degraded: ... / PSU WARN PSU check not run: <why>
  STOP <why>       last, only when no more load may start: "PSU redundancy lost (PSU2)" or "PSU lost (PSU2)"
Exit 0; 2 on bad usage.
"""
import os
import re
import sys

BAD = re.compile(r"absent|(?<!non-)critical|fail|ac lost|non-?recoverable|not present", re.I)  # never "Non-Critical"
RED_LOST = re.compile(r"redundancy lost|non-?redundant|insufficient", re.I)
RED_DEGR = re.compile(r"degraded", re.I)


def rows(text):
    """'name | id | status | entity | reading' rows -> [(name, reading)]."""
    out = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 5 and p[0]:
            out.append((p[0], p[4]))
    return out


def watts(text):
    """'Power_PSU0 | 900' rows -> {0: 900}."""
    w = {}
    for line in text.splitlines():
        m = re.match(r"\s*Power_PSU(\d+)\s*\|\s*([0-9.]+)\s*$", line)
        if m:
            w[int(m.group(1))] = float(m.group(2))
    return w


def peaks(path):
    """The stage poll's peak per PSU (columns 11-14) and peak box sum over rows where all four read: ({n: W}, W) or None."""
    pk, box = {}, 0.0
    try:
        with open(path) as f:
            for line in f:
                v = line.split()
                if len(v) < 14 or not all(x.isdigit() for x in v[10:14]):
                    continue
                p = [int(x) for x in v[10:14]]
                for i, x in enumerate(p):
                    pk[i] = max(pk.get(i, 0), x)
                box = max(box, sum(p))
    except OSError:
        return None
    return (pk, box) if pk else None


def judge(when, sdr, wtxt, power=None, rated=4000.0, pct=90.0, box_kw=14.0):
    r = rows(sdr)
    psus = sorted({int(m.group(1)) for n, _ in r for m in [re.match(r"PSU(\d+)(_|$)", n)] if m})
    if not psus:
        return [f"PSU WARN PSU check not run: no PSU<n> rows in ipmitool sdr type 'Power Supply' ({when})"]
    w = watts(wtxt)
    bad = {}
    for n, v in r:
        m = re.match(r"PSU(\d+)(_\w+)?$", n)
        if m and BAD.search(v):
            bad.setdefault(int(m.group(1)), []).append(f"{n} {v}")
    red = [v for n, v in r if n.lower() == "psu_redundancy"]
    out = []
    parts = [f"PSU{i} " + (f"{w[i]:.0f} W" if i in w else "no reading") + (" (lost)" if i in bad else "") for i in psus]
    line = f"PSU {when}: " + ", ".join(parts)
    if w:
        line += f"; box {sum(w.values()) / 1000:.2f} kW (PSU sum)"
    pk = peaks(power) if power else None
    if pk:
        line += (
            "; stage peak "
            + ", ".join(f"PSU{i} {x} W" for i, x in sorted(pk[0].items()))
            + f", box {pk[1] / 1000:.2f} kW"
        )
    out.append(line)
    for i in sorted(bad):
        out.append(f"PSU FINDING PSU{i}: {', '.join(bad[i])} ({when})")
    lost = [v for v in red if RED_LOST.search(v)]
    names = ",".join(f"PSU{i}" for i in sorted(bad)) or "no PSU named"
    if lost:
        out.append(f"PSU FINDING redundancy lost: PSU_Redundancy {lost[0]} ({names}) ({when})")
    elif any(RED_DEGR.search(v) for v in red):
        out.append(f"PSU WARN redundancy degraded: PSU_Redundancy {red[0]} ({names}) ({when})")
    top = dict(w)
    for i, x in pk[0].items() if pk else []:
        top[i] = max(top.get(i, 0), x)
    for i in sorted(top):
        if top[i] > rated * pct / 100:
            out.append(
                f"PSU WARN PSU{i} at {top[i]:.0f} W, {100 * top[i] / rated:.0f}% of its {rated:.0f} W rating ({when})"
            )
    bx = max(sum(w.values()), pk[1] if pk else 0)
    if bx > box_kw * 1000:
        out.append(f"PSU WARN box at {bx / 1000:.2f} kW, above {box_kw:g} kW ({when})")
    if lost:
        out.append(f"STOP PSU redundancy lost ({names})")
    elif bad:
        out.append(f"STOP PSU lost ({names})")
    return out


def main(argv):
    if len(argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    rd = lambda p: open(p, errors="replace").read() if os.path.exists(p) else ""
    e = os.environ
    for line in judge(
        argv[1],
        rd(argv[2]),
        rd(argv[3]),
        argv[4] if len(argv) > 4 else None,
        float(e.get("PSU_RATED_W") or 4000),
        float(e.get("PSU_WARN_PCT") or 90),
        float(e.get("PSU_BOX_WARN_KW") or 14),
    ):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
