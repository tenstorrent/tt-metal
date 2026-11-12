#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-chip core voltage under load (the fleet Vmin work). Report only: never a finding, never a grade.

From the heavy chunk's VCORE samples (lib/vcore.sh, "<epoch> <bdf> <mV>") and C8's telemetry (c8_telemetry.txt,
"<epoch> <bdf> <temp> <power> <aiclk MHz> <curr> <vcore> ..."): one line per tray with each chip's mean [min-max] mV and
its median AICLK over that chip's own sample window, one box line, and a VCORE HIGH line for each chip whose mean is at
least <high_mv> over the box median of chip means (one chip seen at 758 mV mean against about 720, at the
lowest clock). No flag below 3 chips (no median to compare with).

  vcore_chips.py <vcore.txt> <c8_telemetry.txt or -> <high_mv> [<bdf>=<tray> ...]
"""
import statistics
import sys


def short(b):
    return b[5:] if b.startswith("0000:") else b


def load_vcore(path):
    s = {}
    with open(path, errors="replace") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 3 or not p[0].isdigit() or not p[2].isdigit():
                continue
            s.setdefault(short(p[1]), []).append((int(p[0]), int(p[2])))
    return s


def load_clocks(path, win):
    """Each chip's AICLK samples (MHz) inside its own VCORE window [first, last] epoch; '-' and junk are skipped."""
    c = {}
    if path == "-":
        return c
    try:
        f = open(path, errors="replace")
    except OSError:
        return c
    with f:
        for ln in f:
            p = ln.split()
            if len(p) < 5 or not p[0].isdigit() or not p[4].isdigit():
                continue
            b, e, a = short(p[1]), int(p[0]), int(p[4])
            if b in win and win[b][0] <= e <= win[b][1] and 0 < a < 5000:
                c.setdefault(b, []).append(a)
    return c


def report(samples, clocks, high_mv, tray):
    rows = {}
    for b, v in samples.items():
        mv = [x for _, x in v]
        ck = clocks.get(b)
        rows[b] = dict(n=len(mv), mean=round(statistics.fmean(mv)), lo=min(mv), hi=max(mv),
                       clk=round(statistics.median(ck)) if ck else None)
    if not rows:
        return []
    out = []
    by_tray = {}
    for b in sorted(rows):
        by_tray.setdefault(tray.get(b, "?"), []).append(b)
    for t in sorted(by_tray):
        cells = ", ".join(f"{b} {rows[b]['mean']} [{rows[b]['lo']}-{rows[b]['hi']}] @{rows[b]['clk'] or '-'}" for b in by_tray[t])
        out.append(f"VCORE tray {t} under load (heavy chunk; mean [min-max] mV @ AICLK MHz): {cells}")
    means = [r["mean"] for r in rows.values()]
    med = statistics.median(means)
    cks = [r["clk"] for r in rows.values() if r["clk"]]
    mck = statistics.median(cks) if cks else None
    out.append(f"VCORE box under load: {len(rows)} chips, median of chip means {med:.0f} mV (range {min(means)}-{max(means)}),"
               f" median AICLK {f'{mck:.0f} MHz' if mck else 'not read'}")
    if len(rows) < 3:
        return out
    for b in sorted(rows, key=lambda x: (-rows[x]["mean"], x)):
        r = rows[b]
        if r["mean"] - med < high_mv:
            continue
        ck = "AICLK not read"
        if r["clk"]:
            ck = f"AICLK {r['clk']} MHz (box median {mck:.0f}{', lowest on the box' if r['clk'] == min(cks) and len(set(cks)) > 1 else ''})"
        out.append(f"VCORE HIGH {b} (tray {tray.get(b, '?')}): mean {r['mean']} mV under load, {r['mean'] - med:.0f} mV over the"
                   f" box median {med:.0f} mV (flag at +{high_mv} mV, FLEET_VCORE_HIGH_MV), {ck}; report only")
    return out


def main(a):
    if len(a) < 3:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    try:
        high = int(a[2])
    except ValueError:
        high = 30
    tray = {}
    for kv in a[3:]:
        b, _, t = kv.partition("=")
        tray[short(b)] = t or "?"
    s = load_vcore(a[0])
    win = {b: (min(e for e, _ in v), max(e for e, _ in v)) for b, v in s.items()}
    for ln in report(s, load_clocks(a[1], win), high, tray):
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
