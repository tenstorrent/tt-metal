#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The run's last line, 'Repair recommendation: ...'.

It maps each failing tray's root-cause line (tray_why.py's "    part | stage | evidence | cause: C -> next: N" lines
under fl_report's FAIL tray lines) to the part to repair, naming tray 1-4 and the chip BDF.

  repair_rec.py --rc N --verdict TEXT [--trays FILE] [--findings FILE] [--fix FILE] [--mode M] [--commit C]
                [--rootcause FILE]

Prints one line. --rootcause appends one tab-separated row per numeric fault:
kind, part, cause, confidence, evidence, ruled_out, fix, next_step. Any read error falls back to 'unknown: see the FAIL lines above'.
"""
import argparse
import re
import sys

BDF = r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
HEAD = re.compile(r"^(FAIL|INCOMPLETE|PASS|KNOWN-DROP) tray ([1-4])\b(.*)$")
WHY = re.compile(r"^    (.*?) \| .*\| cause: (.*?) -> next: (.*)$")
UNKNOWN = "unknown: see the FAIL lines above"

# (cause pattern, repair template); first match wins. {u} tray, {b} chip BDF(s) of the line, {p} its part text.
# None: the line asks for no repair (a warning or not a fault). Order: power, software/firmware, links, PCIe, chip.
RULES = [
    (r"not opened after the reset because tray ([1-4]) lost power", None),   # another tray's power broke the chain
    (r"whole-tray power loss|lost power and its re-power failed|bus-bar signature|^tray 12 V input lost or under-voltage",
     "bus bar / tray 12 V feed of tray {u}, then tray {u} board if it still drops"),
    (r"^POWER: tray [1-4] 12 V feed", "POWERFEED"),   # an HSC power latch newly set during this run
    (r"^test infrastructure", "no hardware repair (software: test infrastructure on tray {u})"),
    (r"^firmware version differs", "no hardware repair (firmware: reflash tray {u})"),
    (r"warning only|not a fault|not checked|leave the box|read no link map|ETH ports not up \(warning\)|down before the run", None),
    (r"which fails all its C5 pairs: that chip", "the tray {t} board carrying {c} (all its links fail)"),
    (r"^in-box link down or bad", "the tray-to-tray connection of link {p}, then the tray boards"),
    (r"^link pair hung|^link down or bad", "the cable or port of link {p}"),
    (r"^weak PCIe link|^correctable AER rate outlier|^PCIe path", "the PCIe path (slot, retimer) of {b} on tray {u}, then the tray {u} board"),
    (r"^CCL group fault: one bad result from the likely source", "NUMERIC:{s}"),
    (r"^chip compute fault|^chip DRAM fault|^chip read fault", "NUMERIC:{b}"),
    (r"^these chips stopped while the others kept going|^chip fell off PCIe|^chip lost after its warm reset"
     r"|^chip did not take the warm reset|^hang on the named chip\(s\)|^chip slow to finish its warm reset"
     r"|^chip stopped answering the kernel driver under load|^chip hung for MMIO",
     "the tray {u} board carrying {b}"),
]


def bdfs(text, u=None):
    """Chip BDFs named in text, of tray u where the text ties them to a tray."""
    tied = re.findall(rf"({BDF}) \(tray ([1-4])\)", text)
    if tied:
        got = [b for b, t in tied if u is None or t == u] or [b for b, _ in tied]
    else:
        got = re.findall(BDF, text)
    return list(dict.fromkeys(got))


def item(u, part, cause, nxt, commit):
    """The repair for one root-cause line, '' for none, None for unknown."""
    for pat, tpl in RULES:
        if not re.search(pat, cause):
            continue
        if tpl is None:
            return ""
        bs = bdfs(part, u) or bdfs(nxt, u) or bdfs(cause, u)
        b = " ".join(bs) if bs else f"a chip of tray {u}"
        if tpl == "POWERFEED":
            m = re.search(r"12 V feed (.+?) / bus bar: HSC (0x[0-9a-f]+) ([A-Z_+]+)", cause)
            src = f"feed {m.group(1)} of tray {u} (HSC {m.group(2)} {m.group(3)} newly set during this run)" if m else f"feed of tray {u}"
            return f"bus bar / tray 12 V {src}, then tray {u} board if it persists"
        if tpl.startswith("NUMERIC:"):
            if tpl == "NUMERIC:{s}":
                m = re.search(rf"likely source ({BDF})", cause)
                b = m.group(1) if m else b
            cc = re.search(r"compute core \((\d+),(\d+)\)", cause)
            if cc:   # the owning compute core names the board to replace
                wc = re.search(r"\(translated worker core (\d+),(\d+)\)", cause)   # not the die location
                return (f"the tray {u} board (UBB-{u}): ASIC {b} compute core logical ({cc.group(1)},{cc.group(2)})"
                        + (f", translated worker core ({wc.group(1)},{wc.group(2)})" if wc else "") + " writes wrong values under matmul load")
            # a numeric FAIL is the box only once clean hosts on the same commit stay clean
            return (f"the tray {u} board carrying {b} (numeric fault), only if the same check with commit {commit} "
                    f"on clean hosts shows no similar counts (similar counts there: test or CCL issue, not the box)")
        m = re.search(rf"of ({BDF}) \(tray ([1-4])\)", cause)
        return tpl.format(u=u, b=b, p=part, c=m.group(1) if m else b, t=m.group(2) if m else u)
    return None


RC_COLS = ("kind", "part", "cause", "confidence", "evidence", "ruled_out", "fix", "next_step")
NUMERIC = re.compile(r"^chip compute fault|^chip DRAM fault|^chip read fault|^CCL group fault")
POWER = re.compile(r"whole-tray power loss|lost power|bus-bar signature|^POWER: tray [1-4] 12 V feed")


def rootcause_rows(trays_text, commit="?"):
    """One dict per numeric root-cause line under a FAIL tray: tray_why.py's cause and next, repair_rec's
    item() as the fix. A compute core named means the DRAM readback split did not fail for that chip (tray_why only
    says DRAM when it did: a pinpoint 'DRAM bank' is a tile index mod 8, not a bank)."""
    rows, u, heads, lines = [], None, {}, []
    for l in trays_text.splitlines():
        h = HEAD.match(l)
        if h:
            u = h.group(2)
            heads[u] = h.group(1)
            continue
        w = re.match(r"^    (.*?) \| (.*?) \| (.*) \| cause: (.*?) -> next: (.*)$", l)
        if w and u and heads.get(u) == "FAIL":
            lines.append((u,) + w.groups())
    power = {t for t, *r in lines if POWER.search(r[3])}
    for t, part, st, ev, c, nxt in lines:
        if not NUMERIC.search(c):
            continue
        core = re.search(r"compute core \((\d+),(\d+)\)", c)
        if c.startswith("chip DRAM fault"):
            kind, conf = "numeric-dram", "high" if re.search(r"\bch(annel)? \d+|bank \d+", c) else "medium"
            ro = "compute (the DRAM readback split failed)"
        elif c.startswith("chip read fault"):   # one bad read across a core column, memory or NoC not told apart
            kind, conf, ro = "numeric-read", "medium", "one compute core (every core of the column got the same bad read)"
        elif c.startswith("CCL group fault"):
            kind, conf, ro = "numeric-ccl", "low", "none (one bad result; the likely source is not confirmed)"
        else:
            kind = "numeric-compute"
            conf = "high" if core else "medium"
            ro = "DRAM (no DRAM readback failure for this chip)" if core else "none located (no core named)"
        if t not in power:
            ro += "; tray power loss (none this run)"
        r = item(t, part, c, nxt, commit)
        pt = re.sub(r"^tray [1-4]$|\s*\(tray [1-4]\)", "", part).strip()   # the tray once
        p = f"tray {t}" + (f" {pt}" if pt else "") + (f" compute core logical ({core.group(1)},{core.group(2)})" if core else "")
        rows.append(dict(kind=kind, part=p, cause=c, confidence=conf, evidence=f"{st}: {ev}", ruled_out=ro,
                         fix=("Replace " + r) if r else "none determined", next_step=nxt))
    return rows


def write_rootcause(path, rows):
    """Append rows to path as TSV, with a '# kind<TAB>...' header when the file is new or empty."""
    import os
    clean = lambda v: re.sub(r"[\t\r\n]+", " ", str(v)).strip()   # noqa: E731
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if new:
            f.write("# " + "\t".join(RC_COLS) + "\n")
        for r in rows:
            f.write("\t".join(clean(r[k]) for k in RC_COLS) + "\n")


def recommend(rc, verdict, trays_text="", findings_text="", mode="", commit="?", fix_text=""):
    rerun = "run.sh" + (f" --mode={mode}" if mode and mode != "quick" else "")
    items, unknown, heads, named = [], [], {}, set()
    u = None
    for l in trays_text.splitlines():
        h = HEAD.match(l)
        if h:
            u = h.group(2)
            heads[u] = (h.group(1), h.group(3).strip())
            continue
        w = WHY.match(l)
        if not (w and u and heads.get(u, ("",))[0] == "FAIL"):
            continue
        r = item(u, w.group(1), w.group(2), w.group(3), commit)
        if r is None:
            unknown.append(u)
        elif r:
            named.add(u)
            if r not in items:
                items.append(r)
    fx = fix_text.splitlines()   # lib/link_check.sh's lk_hint: "FIX tray N: **...**", then "  why: ..." (host link check)
    for i, l in enumerate(fx):
        m = re.match(r"FIX tray ([1-4]): \*\*(.*?)\*\*", l)
        if not m:
            continue
        t, why = m.group(1), fx[i + 1] if i + 1 < len(fx) and fx[i + 1].startswith("  why:") else ""
        if "bus bar" in m.group(2):
            r = f"bus bar / tray 12 V feed of tray {t}, then tray {t} board if it still drops"
        else:
            bs = bdfs(why, t)
            r = (f"the tray {t} board carrying {' '.join(bs) if bs else 'its missing chips'} or its PCIe path, after the "
                 f"remote recovery steps ({m.group(2)})")
        named.add(t)
        if r not in items:
            items.append(r)
    for l in findings_text.splitlines():   # PSU findings name no tray (lib/psu_check.sh)
        if l.startswith("PSU FINDING") and "the PSU named in the PSU FINDING line" not in items:
            items.append("the PSU named in the PSU FINDING line")
    for t in "1234":   # the HSC-named feed of a tray replaces that tray's generic bus-bar item
        if any(i.startswith("bus bar / tray 12 V feed ") and f" of tray {t} (HSC " in i for i in items):
            items = [i for i in items if i != f"bus bar / tray 12 V feed of tray {t}, then tray {t} board if it still drops"]
    fails = sorted(t for t, (k, _) in heads.items() if k == "FAIL")
    unknown += [t for t in fails if t not in named]   # a FAIL tray with no repairable cause line under it
    if unknown:
        items.append(f"{UNKNOWN} (tray {', '.join(sorted(set(unknown)))})")
    kd = [t for t, (k, _) in heads.items() if k == "KNOWN-DROP"]
    if items:
        return "; ".join(items + [f"tray {t} (known dropped, not tested)" for t in kd])
    v = verdict or ""
    if v.startswith(("NOT RUN", "INCOMPLETE")) or rc in (9, 10, 129, 130, 143) or \
            any(k == "INCOMPLETE" for k, _ in heads.values()):
        inc = [f"tray {t}" for t, (k, _) in sorted(heads.items()) if k == "INCOMPLETE"]
        return f"none determined ({v or f'exit {rc}'}); rerun {rerun}" + (f" for {', '.join(inc)}" if inc else "")
    if kd:
        return f"tray {kd[0]} (known dropped, not tested); none for the other trays"
    if rc == 0 or v == "PASS":
        return "none (all 4 trays pass)"
    if not heads:
        return f"none determined ({v or f'exit {rc}'}); fix the cause above and rerun {rerun}"
    return UNKNOWN


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--rc", type=int, default=0)
    a.add_argument("--verdict", default="")
    a.add_argument("--trays", default="")
    a.add_argument("--findings", default="")
    a.add_argument("--mode", default="")
    a.add_argument("--commit", default="?")
    a.add_argument("--fix", default="")
    a.add_argument("--rootcause", default="")
    o = a.parse_args(argv)

    def rd(p):
        try:
            return open(p, errors="replace").read() if p else ""
        except OSError:
            return ""
    try:
        r = recommend(o.rc, o.verdict, rd(o.trays), rd(o.findings), o.mode, o.commit, rd(o.fix))
    except Exception:   # the last line is always printed
        r = UNKNOWN
    if o.rootcause:   # never stops the last line
        try:
            rows = rootcause_rows(rd(o.trays), o.commit)
            if rows:
                write_rootcause(o.rootcause, rows)
        except Exception:  # noqa: BLE001
            pass
    print(f"Repair recommendation: {r}")


if __name__ == "__main__":
    sys.exit(main())
