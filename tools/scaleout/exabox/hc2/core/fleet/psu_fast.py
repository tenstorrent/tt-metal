#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The fast per-PSU input power sampler's summary (lib/psu_fast.sh samples, this reports).

  psu_fast.py <psu_fast.txt> <what> [<power.txt>]

psu_fast.txt: "# names <sensor>..." (Power_PSU<n>..., Power_Total), "# mark <epoch> <label>" (a part starts), "# off <why>",
and one "<epoch start> <epoch end> <W>..." row per sweep (NA: not read). power.txt (lib/fleet.sh fl_power_start): columns
2-5 are the trays' UBB W (they cap at 2550 W), the tray-side model is their sum / 0.94 + 1.5 kW.

Lines (report only, never a FAIL or WARN):
  PSU FAST <what>: sweeps, rate reached and how often the BMC's values change; per PSU peak / mean; the PSU sum's peak / mean
  PSU FAST <what> share: the largest single-PSU share of the PSU sum over sweeps at half its peak or more (a quarter each
    when balanced: one PSU above that carries more than its outlet's quarter)
  PSU FAST <what> BMC: Power_Total peak / mean beside the PSU sum
  PSU FAST <what> onset <i>: each load onset (PSU sum up >= ONSET_W within ONSET_S s): the level before, the peak in the
    PEAK_S s after, the settled mean after that, the overshoot, and the tray model at the peak
"""
import sys
import time

ONSET_W = 1000.0   # a load onset: the PSU sum rises this much ...
ONSET_S = 3.0      # ... within this many seconds of the level before it
PRE_S = 5.0        # the level before: the mean over this window
PEAK_S = 15.0      # the onset's peak: the max over this window after it
SETTLE_S = (15.0, 45.0)   # the settled level: the mean over this window after it
EFF, BASE_W, UBB_CAP = 0.94, 1500.0, 2550.0


def load(path):
    names, marks, rows, off = [], [], [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "#":
                if len(p) > 1 and p[1] == "names":
                    names = p[2:]
                elif len(p) > 3 and p[1] == "mark":
                    try:
                        marks.append((float(p[2]), " ".join(p[3:])))
                    except ValueError:
                        pass
                elif len(p) > 1 and p[1] == "off":
                    off = " ".join(p[2:]) or "no route"
                continue
            if not names or len(p) != 2 + len(names):
                continue
            try:
                t = (float(p[0]) + float(p[1])) / 2
                rows.append((t, float(p[1]) - float(p[0]), [None if x == "NA" else float(x) for x in p[2:]]))
            except ValueError:
                continue
    return names, marks, rows, off


def trays(path):
    """power.txt -> [(epoch, tray W sum, capped)] for rows where all four trays read."""
    out = []
    try:
        fh = open(path, errors="replace")
    except OSError:
        return out
    with fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            try:
                t, w = float(p[0]), [float(x) for x in p[1:5]]
            except ValueError:
                continue
            out.append((t, sum(w), any(x >= UBB_CAP for x in w)))
    return out


def near(series, t, within=5.0):
    best = None
    for s in series:
        if abs(s[0] - t) <= within and (best is None or abs(s[0] - t) < abs(best[0] - t)):
            best = s
    return best


def model(tr, t):
    s = near(tr, t)
    if s is None:
        return ""
    return f"; trays {s[1]:.0f} W /{EFF} + {BASE_W / 1000:g} kW = {s[1] / EFF + BASE_W:.0f} W{' (a tray at its 2550 W cap: a floor)' if s[2] else ''}"


def hms(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def onsets(pts):
    """[(i of the onset sample)] where the PSU sum rose >= ONSET_W over the PRE_S mean within ONSET_S s; one per rise."""
    out, j, last = [], 0, -1e18
    for i, (t, s) in enumerate(pts):
        while pts[j][0] < t - ONSET_S - PRE_S:
            j += 1
        pre = [v for (u, v) in pts[j:i] if u < t - ONSET_S]
        if not pre or t - last < SETTLE_S[1]:
            continue
        if s - sum(pre) / len(pre) >= ONSET_W:
            out.append(i)
            last = t
    return out


def report(path, what, power=None):
    names, marks, rows, off = load(path)
    if off is not None and not rows:
        return [f"PSU FAST {what}: off: {off}"]
    if not rows:
        return [f"PSU FAST {what}: off: no sweep read"]
    psu = [k for k, n in enumerate(names) if n.startswith("Power_PSU")]
    tot = names.index("Power_Total") if "Power_Total" in names else None
    full = [(t, [v[k] for k in psu]) for (t, _, v) in rows if all(v[k] is not None for k in psu) and sum(v[k] for k in psu) > 0]   # every PSU read, not all 0 W
    t0, t1 = rows[0][0], rows[-1][0]
    span = t1 - t0
    rate = span / (len(rows) - 1) if len(rows) > 1 else 0.0
    ch, prev, pt = [], None, None   # how often any value changes: the BMC's own refresh
    for (t, _, v) in rows:
        if prev is not None and v != prev:
            if pt is not None:
                ch.append(t - pt)
            pt = t
        prev = v
    ch.sort()
    chs = f", the BMC's values change every {ch[len(ch) // 2]:.1f} s (median)" if ch else ""
    per = []
    for k in psu:
        vs = [v[k] for (_, _, v) in rows if v[k] is not None]
        per.append(f"{names[k].replace('Power_', '')} peak {max(vs):.0f} W mean {sum(vs) / len(vs):.0f} W" if vs else f"{names[k].replace('Power_', '')} not read")
    out = [f"PSU FAST {what}: {len(rows)} sweeps over {span:.0f} s, one every {rate:.2f} s{chs}; " + ", ".join(per)]
    tr = trays(power) if power else []
    if full:
        sums = [(t, sum(w)) for (t, w) in full]
        pk = max(sums, key=lambda x: x[1])
        out[0] += f"; PSU sum peak {pk[1]:.0f} W at {hms(pk[0])} (+{pk[0] - t0:.0f} s), mean {sum(s for _, s in sums) / len(sums):.0f} W{model(tr, pk[0])}"
        sh = max(((t, k, w[k] / sum(w), w[k], sum(w)) for (t, w) in full if sum(w) > 0 and sum(w) >= pk[1] / 2 for k in range(len(psu))), key=lambda x: x[2])   # under load: sweeps at half the peak or more
        hi = max(full, key=lambda x: sum(x[1]))
        hs = max(range(len(psu)), key=lambda k: hi[1][k])
        out.append(f"PSU FAST {what} share: largest single-PSU share {100 * sh[2]:.1f}% ({names[psu[sh[1]]].replace('Power_', '')} {sh[3]:.0f} of {sh[4]:.0f} W at {hms(sh[0])}); at the PSU sum's peak "
                   f"{names[psu[hs]].replace('Power_', '')} carries {100 * hi[1][hs] / sum(hi[1]):.1f}% ({hi[1][hs]:.0f} W); balanced is {100 / len(psu):.0f}% each")
        for n, i in enumerate(onsets(sums), 1):
            t = sums[i][0]
            pre = [s for (u, s) in sums if t - ONSET_S - PRE_S <= u < t - ONSET_S]
            win = [(u, s) for (u, s) in sums if t <= u <= t + PEAK_S]
            st = [s for (u, s) in sums if t + SETTLE_S[0] <= u <= t + SETTLE_S[1]]
            p = max(win, key=lambda x: x[1])
            lab = [m for (u, m) in marks if u <= t]
            stl = f", settled {sum(st) / len(st):.0f} W, overshoot {p[1] - sum(st) / len(st):+.0f} W" if st else ", not settled (the stage ended)"
            out.append(f"PSU FAST {what} onset {n}{f' ({lab[-1]})' if lab else ''} at {hms(t)}: PSU sum {sum(pre) / len(pre):.0f} W before, peak {p[1]:.0f} W "
                       f"{p[0] - t:.1f} s after{stl}{model(tr, p[0])}")
    else:
        out[0] += "; no sweep read every PSU"
    if tot is not None:
        vs = [(t, v[tot]) for (t, _, v) in rows if v[tot] is not None]
        if vs:
            pk = max(vs, key=lambda x: x[1])
            out.append(f"PSU FAST {what} BMC: Power_Total peak {pk[1]:.0f} W at {hms(pk[0])}, mean {sum(x for _, x in vs) / len(vs):.0f} W")
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    for line in report(argv[1], argv[2], argv[3] if len(argv) > 3 else None):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
