#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""run.sh --mode=long's end summary: lib/long.sh's EXIT trap runs it.

  long_summary.py <legs.tsv>

legs.tsv: one row per leg that ran: i, mode, rc, start epoch, end epoch, results dir, log, why it stopped ('' if not).
Prints per leg its mode, verdict (COMPLETE PASS / FAIL / INCOMPLETE), duration and dir; every fault per tray 1-4 and chip
BDF with k/5 (from each leg's fleet/trays.txt root-cause lines); each leg's HSC latch lines (where its
log has them); then one 'VERDICT: ...' line for the whole long run (FAIL if any leg failed, else INCOMPLETE if any leg
did not pass or did not run, else PASS) and last one combined 'Repair recommendation:' line built from each
leg's own last line. Exit 0 for PASS, 8 for FAIL, 9 for INCOMPLETE (lib/long.sh exits with it after a normal end).
"""
import os
import re
import sys

N = 5
BDF = r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
HEAD = re.compile(r"^(FAIL|INCOMPLETE|PASS|KNOWN-DROP) tray ([1-4])\b")
WHY = re.compile(r"^    (.*?) \| .*\| cause: (.*?) -> next: (.*)$")
RR = "Repair recommendation: "
VRC = {"PASS": 0, "FAIL": 8}   # the long run's VERDICT word -> exit code; anything else (INCOMPLETE) 9
HSC = re.compile(r"(?:^|: )((?:WARN )?HSC\b(?! feed ).*)$")  # not the per-feed sag lines


def rd(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def verdict(rc, log):
    v = ""
    for l in log.splitlines():
        if l.startswith("VERDICT: "):
            v = l[9:].strip()
    if v == "PASS" and rc == 0:
        return "COMPLETE PASS", v
    if v.startswith(("INCOMPLETE", "NOT RUN", "KNOWN-DROP")) or rc in (6, 9, 10, 12, 129, 130, 143) or not v:
        return "INCOMPLETE", v or f"no VERDICT line, exit {rc}"
    return "FAIL", v


def faults(trays):
    """{(tray, bdf, cause)} of the FAIL trays' root-cause lines."""
    got, u, fail = set(), None, False
    for l in trays.splitlines():
        h = HEAD.match(l)
        if h:
            u, fail = h.group(2), h.group(1) == "FAIL"
            continue
        w = WHY.match(l)
        if not (w and u and fail):
            continue
        cause = re.sub(r"\s+\(.*$|:.*$", "", w.group(2)).strip() or w.group(2)
        bs = re.findall(rf"({BDF}) \(tray {u}\)", w.group(1)) or re.findall(BDF, w.group(1)) or ["-"]
        for b in bs:
            got.add((u, b, cause))
    return got


def rr_items(log):
    last = ""
    for l in log.splitlines():
        if l.startswith(RR):
            last = l[len(RR):].strip()
    return last


def summary(rows):
    out = ["", f"=== run.sh --mode=long summary: {len(rows)} of {N} legs ran (plain, plain, plain, full, full)"]
    counts, legs_rr, stop = {}, [], ""
    for r in rows:
        i, mode, rc, t0, t1, d, logp, why = (r + [""] * 8)[:8]
        rc = int(rc) if rc.lstrip("-").isdigit() else 1
        log = rd(logp)
        cls, v = verdict(rc, log)
        dur = (int(t1) - int(t0)) if t0.isdigit() and t1.isdigit() else -1
        name = "plain" if mode == "quick" else mode
        out.append(f"leg {i}/{N} {name:5s} {cls} ({v}), {dur // 60} min {dur % 60:02d} s, {d}")
        for k in faults(rd(os.path.join(d, "fleet", "trays.txt"))):
            counts.setdefault(k, set()).add(i)
        legs_rr.append((i, cls, rr_items(log)))
        if why:
            stop = why
    for i in range(len(rows) + 1, N + 1):
        out.append(f"leg {i}/{N} {'plain' if i <= 3 else 'full':5s} NOT RUN ({stop or 'the long run stopped'})")
    if stop:
        out.append(f"stopped early: {stop}")
    out.append("faults per tray and chip (k/5 legs):")
    if not counts:
        out.append("  none")
    for (u, b, c), ls in sorted(counts.items()):
        out.append(f"  tray {u} {b}: {c}: {len(ls)}/{N} (legs {','.join(sorted(ls))})")
    out.append("HSC latch lines per leg:")
    for r in rows:
        hs = [m.group(1) for m in (HSC.search(l) for l in rd(r[6] if len(r) > 6 else "").splitlines()) if m]
        out.append(f"  leg {r[0]}: " + ("; ".join(hs) if hs else "not reached"))
    out.append("VERDICT: " + overall([c for _, c, _ in legs_rr]))
    out.append(RR + combine(legs_rr, len(rows), stop))
    return out


def overall(classes):
    """The whole long run's verdict from each leg's class; a leg that did not run counts as INCOMPLETE."""
    fails = classes.count("FAIL")
    if fails:
        return f"FAIL ({fails}/{N} legs)"
    passes = classes.count("COMPLETE PASS")
    if passes == N:
        return f"PASS ({N}/{N} legs)"
    return f"INCOMPLETE ({N - passes}/{N} legs)"


def combine(legs_rr, ran, stop):
    rep = {}
    for i, _, r in legs_rr:
        if r.startswith("none"):   # 'none (...)' and 'none determined (...); rerun ...': the leg names no repair
            continue
        for it in (x.strip() for x in r.split("; ")):
            if it and not it.startswith("none"):   # 'none ...' and 'none determined ...; rerun' name no repair
                rep.setdefault(it, []).append(i)
    if rep:
        items = [f"{it} ({len(ls)}/{N} legs)" for it, ls in rep.items()]
        return "; ".join(items + ([f"stopped early ({stop}); rerun run.sh --mode=long"] if stop else []))
    inc = [i for i, c, _ in legs_rr if c != "COMPLETE PASS"]
    if stop or ran < N or inc:
        what = f"stopped early: {stop}" if stop else f"legs {', '.join(inc)} incomplete" if inc else f"{ran} of {N} legs ran"
        return f"none determined ({what}); rerun run.sh --mode=long"
    return f"none (all 4 trays pass in {N}/{N} legs)"


def main(argv):
    rows = [l.rstrip("\n").split("\t") for l in rd(argv[1] if len(argv) > 1 else "").splitlines() if l.strip()]
    try:
        lines = summary(rows)
    except Exception as e:   # the last line is always printed
        lines = [f"long summary failed: {e}", "VERDICT: INCOMPLETE (long summary failed)",
                 RR + "unknown: see each leg's FAIL lines above"]
    print("\n".join(lines))
    v = [l[9:].split()[0] for l in lines if l.startswith("VERDICT: ")]
    return VRC.get(v[-1] if v else "", 9)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
