# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The run's report, read from its finding lines: one line per tray with, under it, each finding's part, stage,
evidence and likely cause; the final verdict (fault families first, then each fault: what, where, what to do); and the
one-line repair recommendation.

The run directory holds findings.txt (one finding per line: 'ETH FAIL ...', 'NUMERIC FAIL ...') and finding_stage.tsv
('<stage code><TAB><line>'). A chip is named by tray 1-4, ASIC 1-8 and BDF; a link end by tray / QSFP-DD port / ASIC /
BDF with its ETH channels in brackets. An INCOMPLETE run is never a fault.

  python3 -m diag.report RUN_DIR [--rc N] [--verdict TEXT] [--result TEXT] [--mode M] [--commit C] [--fix FILE]
      [--known-tray N] [--stopped TEXT] [--rerun CMD] [--repair-history FILE] [--only trays|result|final]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from .galaxy import UBB_ETH_PORT_OF, UBB_ETH_PORTS, parse_bdf_physical

BDF = r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
XBDF = rf"(?:0000:)?({BDF})"
CHIP = re.compile(XBDF + r" \(tray ([1-4])\)")
ONE = re.compile(XBDF + r" \(tray ([1-4])\)(?: chip \d+| tt-smi (\d+))?")
_ONE = rf"(?:0000:)?{BDF} \(tray [1-4]\)(?: (?:chip|tt-smi) \d+)?"
LIST = re.compile(r"^((?:[A-Z0-9-]+:? )+)(" + _ONE + r"(?:, " + _ONE + r")*)(?=[ :])")
LINK = re.compile(XBDF + r" \(tray ([1-4])\) e([0-9,]+) <-> " + XBDF + r" \(tray ([1-4])\) e([0-9,]+)")
ONECHIP = re.compile(
    r"LINK FAIL one chip " + XBDF + r" \(tray ([1-4])\)[^:]*: in all (\d+) failing (.*?) pairs \(([^)]*)\): (.*)$"
)
ONELINK = re.compile(XBDF + r" e([0-9,]+) <-> " + XBDF + r" e([0-9,]+) ([^;(]*?) \((p\d+)\)")
AFTER = re.compile(r" \[(after fault \(.*, no reset\))\]$")  # a finding made after a fault no reset cleared
PEER = re.compile(r" \(to (?!(?:0000:)?[0-9a-f]{2}:)[^()]*\)")  # a QSFP line's far end on another host
DIED = re.compile(r"CHUNK FAIL chunk \d+/\d+ .*no chunk-end readback, the load died or hung")
INC = re.compile(r"^(ETH|HANG) INCOMPLETE ")
STAGE = {
    "P0": "stage 1 gate",
    "P1": "stage 2 pre-run checks",
    "C2": "stage 3 start reset",
    "busbar": "stage 4 bus-bar stage",
    "numeric": "stage 5 numeric verify",
    "C5": "stage 6 pair link test",
    "C9": "stage 7 end reset",
    "sweep": "stage 8 per-link sweep",
    "end": "end checks",
    "linka": "link-A repro",
    "abortfab": "abort-fabric repro",
}
STAGE_DIR = {
    STAGE["busbar"]: "busbar",
    STAGE["busbar"] + " heavy": "busbar/heavy",
    STAGE["numeric"]: "busbar",
    STAGE["numeric"] + " heavy": "busbar/heavy",
    STAGE["numeric"] + " g3": "busbar/rw",
    STAGE["sweep"]: "sweep",
    STAGE["C5"]: "c5",
}
POWER_NEXT = "1. Replace the bus bar -> 2. if the tray still drops, replace tray {u}"
PCIE_NEXT = "swap this tray's host PCIe cable with a neighbour tray's; if the errors follow the tray, replace the board"
# HSC FAIL lines naming bits newly set during the run; HSC_ADD: the OFF/IOUT_OC bits added to an under-voltage leg line
HSC_NEW = re.compile(
    r"^HSC FAIL tray ([1-4]) \([^)]*\) HSC (0x[0-9a-fA-F]{2}): ([A-Z_]+(?:\+[A-Z_]+)*) (?:latched|set) during "
)
HSC_ADD = re.compile(r"; ([A-Z_]+(?:\+[A-Z_]+)*) set during ")
HSC_POWER_BITS = ("VIN_UV_WARN", "VIN_UV_FAULT", "OFF", "IOUT_OC", "IOUT_OC_FAULT")  # CML: a comm fault, never power
HSC_FEED = {"0x40": "A", "0x42": "B"}
HSC_FAIL = re.compile(r"HSC FAIL tray ([1-4]) \([^)]*\) HSC (0x[0-9a-f]{2}): (.*)")
# a tray's own power loss this run: its POWER DROP, its re-power failing at the C9 reset, or its HSC latched OFF
POWER_LOST = re.compile(
    r"^POWER DROP tray ([1-4])\b|^POWER FAIL tray ([1-4]) .*reset chain broken by its power loss|^HSC FAIL tray ([1-4]) .*: \S*\bOFF\b"
)
HSC_TRIP = re.compile(
    r"^\s*(?:POWER DROP tray ([1-4])\b|HSC FAIL tray ([1-4])\b.*\b(?:OFF|IOUT_OC(?:_FAULT)?|over-current|VIN_UV_FAULT)\b)"
)
IPMI_DOWN = re.compile(r"\(C9 end reset\).*(in-band IPMI not answering|in-band re-power refused)")
PDN_CHIP = re.compile(
    rf"^RESET FAIL 0000:({BDF}) \(tray ([1-4])\) .*\(C9 end reset\).*on PCIe with its tt-kmd node, but tt-umd cannot open it"
)
CONFIG_RE = re.compile(r"; CONFIG: retimer/ETH speed (x\d+G(?: and x\d+G)?), expected (x\d+G)")
RETIMER_RE = re.compile(r"; RETIMER: chip side (x\d+G), retimer speed not readable")
RERUN = "the health check"  # the command a rerun step names (--rerun)
LANE, COL_SKEW, CONC = 0.9, 2.0, 0.8  # one lane parity's share, top column group vs its even share, one bank's share


def tray_asic(b):
    return parse_bdf_physical("0000:" + b[-7:]) or (None, int(b[-6], 16))


def chip(b, idx=None):
    t, a = tray_asic(b)
    return (f"tray {t} ASIC {a} / {b}" if t else b) + (f" (tt-smi chip {idx})" if idx is not None else "")


def trays(bs):  # a BDF outside the tray map names no tray
    return {str(t) for t in (tray_asic(b)[0] for b in bs) if t is not None}


def board(b):
    t = tray_asic(b)[0]
    return f"the tray {t} board" if t else "the board"


def holder(b):
    t = tray_asic(b)[0]
    return f"tray {t}" if t else f"the tray holding {b}"


def chans(s):
    return [int(x) for x in re.findall(r"\d+", str(s))]


def ports(b, chs):
    t, a = tray_asic(b)
    if t is None:  # a BDF outside the tray map: no ASIC, so no port
        return []
    return sorted({p[1] for c in chs for p in [UBB_ETH_PORT_OF.get((a, c))] if p and p[0] == "QSFP_DD"})


def ids(xs):
    return ",".join(map(str, xs))


def hit(u, l):
    return re.search(rf"tray {u}([^0-9]|$)", l)


def thru(lines, rx):
    return {next(g for g in m.groups() if g) for m in map(rx.search, lines) if m}


# ── cause: one line per finding under its tray ──────────────────────────────────────────────────────────────────────


def tray_lines(lines, u, drop_trays=None, stages=None):
    stages = stages or {}
    dstages = {stages.get(l) for l in lines if re.match(r"(POWER|CHIP) DROP tray ", l)}
    fails, warns = [], []
    for l in lines:
        if not l.strip():
            continue
        if drop_trays and str(u) not in drop_trays and DIED.search(l) and stages.get(l) in dstages | {None}:
            continue
        if " WARN " in l or l.startswith(("ETH INCOMPLETE ", "HANG INCOMPLETE ")):
            if hit(u, re.sub(r" by tray [1-4](?: and tray [1-4])* power loss", "", PEER.sub("", l))):
                warns.append(l)
        elif hit(u, PEER.sub("", re.sub(r", BMC after: .*", "", l))):
            fails.append(l)
    return fails, warns


def stage_of(l, stages):
    s = l.lower()
    m = re.search(r"(?<!before )chunk (\d+)(?:/(\d+))?(?![\d/]| or before)", l)
    if stages.get(l) == "sweep":
        return STAGE["sweep"]
    if l.startswith("NUMERIC ") or stages.get(l) == "numeric" or "stage 5 numeric verify" in s or "/busbar/rw" in s:
        tail = " heavy chunk" if "heavy chunk" in s or "/busbar/heavy/" in s else ""
        tail = tail or (" g3" if "g3" in s or "/busbar/rw" in s else f" chunk {m.group(1)}" if m else "")
        return STAGE["numeric"] + tail
    if "bus-bar stage heavy chunk" in s or "/busbar/heavy/" in s:
        return STAGE["busbar"] + " heavy chunk"
    if "bus-bar stage" in s or "/busbar/" in s or stages.get(l) == "busbar" or l.startswith(("CHUNK ", "QUIET ")):
        return STAGE["busbar"] + (f" chunk {m.group(1)}" if m else "")
    if "/c5/" in s:
        return STAGE["C5"]
    if "/sweep/" in s or l.startswith("PINPOINT"):
        return STAGE["sweep"]
    if l.startswith(("C4 ", "C4v ")) or "matmul try" in s:
        return l.split()[0] if l.startswith("C4v ") else "C4"
    if re.search(r"\(c9 end reset\)|after the c9 reset| c9: |^links warn after c9", s):
        return "C9"
    if re.search(r"\(c2 start reset\)|at the start \(c2\)| c2: |^links warn after c2", s):
        return "C2"
    if l.startswith("VAE "):
        return "C4v"
    if l.startswith("PREFLIGHT"):
        return "preflight"
    if re.search(r"\) pre: | pre: ", l):
        return "C3 before the run"
    if l in stages:
        return STAGE.get(stages[l], stages[l])
    return "stage not recorded"


def tend(b, t, chs):
    cs = chans(chs)
    q = ports(b, cs)
    return (f"tray {t} QSFP-DD port {' and '.join(map(str, q))} / " if q else f"tray {t} ") + (
        f"ASIC {int(b[1], 16)} / {b} [ETH ch {ids(cs)}]"
    )


def has_port(part):  # a link part with a faceplate QSFP-DD port at both ends: a cable to reseat
    return part.count("QSFP-DD port") >= 2


def part_of(l, u):
    k = LINK.search(l)
    if k and u in (k.group(2), k.group(5)):
        return "link", f"{tend(k.group(1), k.group(2), k.group(3))} <-> {tend(k.group(4), k.group(5), k.group(6))}"
    m = re.match(
        rf"ETH (?:FAIL|WARN) {XBDF} \(tray ([1-4])\) \w+: local ETH ports \[([0-9, ]+)\] (?:not up|down at pre) \(to ([^;]*?)\)(?:;.*)?$",
        l,
    )
    if m:
        far = {}
        for b, t, pt in re.findall(XBDF + r" \(tray ([1-4])\) port (\d+)", m.group(4)):
            far.setdefault((b, t), []).append(pt)
        return "link", f"{tend(m.group(1), m.group(2), m.group(3))} <-> " + "; ".join(
            tend(b, t, ",".join(v)) for (b, t), v in far.items()
        )
    if re.match(rf"(POWER (DROP|FAIL)|SEL (FAIL|WARN)) tray {u}\b", l) or "tray drop" in l:
        return "tray", f"whole tray {u}"
    m = re.match(r"L1 FAIL " + XBDF + rf" tray {u} core (\S+)", l)
    if m:
        return "chips", f"{m.group(1)} (tray {u}) L1 core {m.group(2)}"
    if re.match(rf"CHIP DROP tray {u}\b", l):
        bs = re.findall(r"[0-9a-f]{2}:00\.[0-7]", l.split(", BMC")[0])
        return "chips", f"{' '.join(bs) or 'chips'} (tray {u})"
    cs = [c for c, t in CHIP.findall(PEER.sub("", l)) if t == u]
    if cs and ("UNATTRIBUTED" in l or "candidates:" in l):
        return "unnamed", f"no single chip named ({len(cs)} candidate chip{'s' * (len(cs) > 1)} on tray {u})"
    if l.startswith("CHUNK FAIL all "):
        return "unnamed", f"all chips (tray {u} among them)"
    m = re.search(rf"{XBDF} \(tray {u}\)[^\[]*?ETH ports? (\[[0-9, ]+\])", l)
    if m:
        return "chips", f"{m.group(1)} (tray {u}) ETH ports {m.group(2)[1:-1].replace(' ', '')}"
    if cs:
        k = {b: n for b, t, n in ONE.findall(l) if t == u and n}
        if k:
            return "chips", ", ".join(
                f"{b} (tray {u}" + (f", tt-smi {k[b]})" if b in k else ")") for b in dict.fromkeys(cs)
            )
        return "chips", " ".join(dict.fromkeys(cs)) + f" (tray {u})"
    return "tray", f"tray {u}"


def path_of(l, findings, stage):
    m = re.search(r"(?:snapshot|triage saved|saved|probe:?|triage) (/[^\s);,]+)", l) or re.search(r"\((/[^\s);,]+)", l)
    if m:
        return m.group(1)
    fl = os.path.dirname(os.path.abspath(findings))
    if stage == "C4":
        return os.path.join(os.path.dirname(fl), "heavy", "hold_try*")
    return os.path.join(fl, STAGE_DIR.get(stage.split(" chunk")[0].split(" at ")[0], ""))


def evidence(l, u=None):
    e = re.sub(r" ?\(?(?<![\w.])/[A-Za-z_~][^\s);,]*\)?", "", l.split(", BMC after:")[0])
    e = re.sub(r"; candidates: .*|, candidates: .*", "", e)
    e = re.sub(r"(\d+ chips lost): [0-9a-f:. ]+", r"\1", e)
    e = LINK.sub("", e, count=1)
    e = re.sub(r"(local ETH ports \[[0-9, ]+\] (?:not up|down at pre)) \(to [^;]*\)", r"\1", e)
    m = LIST.match(e)  # the part: its first chip goes, and of a longer list tray u's chips; another tray's peer stays
    if m:
        cs = [x.group(0) for x in ONE.finditer(m.group(2))]
        keep = [x for x in cs if ONE.match(x).group(2) != u] if u and len(cs) > 1 else cs[1:]
        e = m.group(1) + ", ".join(keep) + e[m.end() :]
    e = re.sub(r"\((?:snapshot|triage saved|probe:?) (?=\(its error)", "", e)
    e = re.sub(r"[(;,]?\s*(?:\b(?:snapshot|triage saved|probe:?)|(?<![\w-])triage)\s*(?=[);,]|$)\)?", "", e)
    e = re.sub(r"\s+", " ", e).replace(" :", ":").replace(" ,", "").strip(" ,;:")
    if "vr_pg 00" in l:
        e += " (tray VR power-good low)"
    return e if len(e) <= 140 else e[:137] + "..."


def ccl_source(l, lines=()):
    m = re.search(r"[;,] likely sources? (.*)$", l)
    if m:
        return list(dict.fromkeys(CHIP.findall(m.group(1))))
    mem, src = set(CHIP.findall(l)), []
    for o in lines:
        f = re.match(r"NUMERIC FAIL " + XBDF + r" \(tray ([1-4])\)(?: chip \S+)?(?: \[([^\]]*)\])?:", o)
        if (
            f
            and "CCL group fault" not in o
            and (f.group(1), f.group(2)) in mem
            and not (f.group(3) and set(f.group(3).split(",")) <= {"h3", "h3-ref"})
        ):
            src.append((f.group(1), f.group(2)))
    return list(dict.fromkeys(src))


def power_lost(lines, u):
    if not any(IPMI_DOWN.search(l) for l in lines):
        return []
    return sorted(t for t in thru(lines, POWER_LOST) if t != str(u))


def cause(
    l, kind, u, findings, stage="", lines=()
):  # (cause, next): each rule below gives them as "cause -> next: next"
    r = rule(l, kind, u, findings, stage, lines)
    return tuple(r.split(" -> next: ", 1)) if isinstance(r, str) else r


def rule(l, kind, u, findings, stage, lines):
    s, warn = l.lower(), " WARN " in l
    rerun = f"rerun {RERUN}"
    cf = CONFIG_RE.search(l) if l.startswith("ETH ") else None
    if cf:
        return f"CONFIG: retimer/ETH speed {cf.group(1)}, expected {cf.group(2)}: the in-box links' retimers are set to the wrong speed (how most BH Galaxy come from the factory), so the link cannot come up; not a cable or tray-board fault -> next: set the retimers to {cf.group(2)} and rerun, before any cabling or board step"
    if l.startswith("ETH INCOMPLETE ") and "start reset skipped; ETH not-active-at-open not judged" in l:
        return "ETH core not active again at mesh open with no start reset before it: start reset skipped; ETH not-active-at-open not judged, not checked, not a fault -> next: none on this box; judged only by a run whose C2 start reset (tt-smi -glx_reset) ran"
    if l.startswith("HANG PINPOINT") and "whole box" in l:
        return "every chip stalled together (one mesh: launch counts move in lockstep), so no chip or tray stopped first: not one tray's board; hardware or software not determined -> next: repeat the stage on this host with tt-triage/exalens there to read the hung state (hold the chips)"
    if l.startswith("HANG PINPOINT"):
        return "these chips stopped while the others kept going: chip/board or its links -> next: read the hang capture of these chips; rerun the stage on this host"
    if kind == "tray" and l.startswith("POWER FAIL") and "reset chain broken by its power loss" in l:
        return (
            f"tray {u} lost power and its re-power failed, so the reset chain broke there: the other trays' chips stay on PCIe but do not open (ENODEV); their boards are not at fault",
            POWER_NEXT.format(u=u),
        )
    if warn and "not opened: reset chain broken by tray" in l:
        t = re.search(r"broken by (tray [1-4](?: and tray [1-4])*) power loss", l).group(1)
        return f"not opened after the reset because {t} lost power and broke the reset chain; this chip stays on PCIe (ENODEV): not this tray's board -> next: fix {t} first (see its FAIL line), then reset and recheck this chip"
    hf = HSC_NEW.match(l)
    if hf and not warn:  # a power latch newly set during this run: a POWER cause naming its feed and HSC
        bs = hf.group(3).split("+") + [b for m in HSC_ADD.findall(l[hf.end() :]) for b in m.split("+")]
        pw = [b for b in dict.fromkeys(bs) if b in HSC_POWER_BITS]
        if pw:
            a, t = hf.group(2).lower(), hf.group(1)
            f = HSC_FEED.get(a, f"at HSC {a}")
            uv = "an under-voltage latch" if any(b.startswith("VIN_UV") for b in pw) else "a power latch"
            return f"POWER: tray {t} 12 V feed {f} / bus bar: HSC {a} {'+'.join(pw)} newly set during this run (clear at its start; {uv} from this run, not a pre-existing one) -> next: 1. Check the bus bar and 12 V feed {f} of tray {t} (HSC {a}) -> 2. replace the bus bar if the latch sets again -> 3. replace tray {t} only if it persists"
    if (
        kind == "tray"
        and l.startswith("HSC FAIL")
        and re.search(r"\b(VIN_UV_WARN|VIN_UV_FAULT|OFF)\b", l)
        and re.search(r"latched during|set during", l)
    ):
        ha = re.search(r"\bHSC (0x[0-9a-fA-F]{2})\b", l)
        return (
            f"POWER: 12 V feed of tray {u}{f' (HSC {ha.group(1)})' if ha else ''}: tray 12 V input lost or under-voltage: HSC OFF/under-voltage latch newly set during the run (12 V-lost evidence)",
            POWER_NEXT.format(u=u),
        )
    if kind == "tray" and l.startswith("HSC WARN ") and "marginal 12 V feed" in l:
        return f"marginal 12 V feed of tray {u}, no trip or latch this run: the feed sagged or came near its over-current trip under load, as on boxes whose trays tripped later: a warning only, not a power fault yet -> next: check tray {u}'s feed: a low idle VIN points at the PSU output, a big drop from idle to loaded at the bus bar or contacts; rerun the bus-bar stage"
    if kind == "tray" and l.startswith("SEL WARN ") and "frb3" in s and "during the run" in s:
        return f"SEL FRB3 processor init failure during the run with no 12 V-lost evidence (no chip lost, no newly set HSC OFF/under-voltage latch, no CPLD 12 V loss): a warning only, not a power fault -> next: none from this line; a lost chip or a newly set HSC latch on tray {u} would have its own line"
    if kind == "tray" and (l.startswith(("POWER DROP", "POWER FAIL")) or "frb3" in s or "tray drop" in s):
        if warn:
            return (
                "earlier whole-tray power loss in the BMC log (bus-bar signature), none in this run",
                POWER_NEXT.format(u=u) + " if it drops in a run",
            )
        return "whole-tray power loss", POWER_NEXT.format(u=u)
    if l.startswith("FAIL MCE GUARD") and "kmd burst" in l:
        return f"chip stopped answering the kernel driver under load (a burst of kmd 'Failed to set initial power state'; the MCE guard cut it): chip/board or its power, not determined -> next: read mce_guard/ and the capture; {rerun} on this host; same chip again: replace the board of tray {u}"
    if not l.startswith("FAIL MCE GUARD") and re.search(
        r"Setting power state failed on device \d+: Input/output error|Failed to set initial power state", l
    ):
        return (
            "PCIe fault: the chip did not take a power-state write over PCIe (UMD/kmd I/O error), not a compute fault: its PCIe path (board, retimer or cable, or the host slot)",
            PCIE_NEXT,
        )
    if re.match(r"AER FAIL .*: correctable \+\d+ ", l) or re.match(
        r"AER WARN .*: correctable \+\d+ .*every other root port \+2 or less", l
    ):
        return (
            "the chip's PCIe path: board, retimer or cable, or the host slot"
            + (" (likely the stall's cause: the only port with errors)" if "likely " in l else ""),
            PCIE_NEXT,
        )
    if l.startswith("FAIL MCE GUARD") and "hung for MMIO" in l:
        return f"chip hung for MMIO under load (config reads ffffffff or an ARC timeout; the MCE guard cut it): CHIP/BOARD -> next: read mce_guard/ and the capture; {rerun} on this host; same chip again: replace the board of tray {u}"
    if l.startswith("FAIL MCE GUARD") and "AER storm" in l:
        return f"PCIe path: an AER storm on its root port under load (the MCE guard cut the chip): slot, retimer or board -> next: read the AER/lspci capture; reseat tray {u}; if it repeats, replace the board of tray {u}"
    hang = not warn and re.search(r"\b(hang|hung)\b(?![- ]in)", s)
    o = ONECHIP.search(l)
    if o:  # every failing pair has this chip at one end, the far-end chips pass their other links
        e = re.search(r"active ethernet core (\d+)-25", l)
        c = (
            f"the chip itself, not the cables: its own ETH core {e.group(1)}-25 (channel {int(e.group(1)) - 20}) does not come up, so "
            if e
            else "the chip or its ETH side, not the cables: "
        ) + "every failing link has it at one end and the far-end chips pass their other links"
        return (
            c,
            f"reset and rerun C5; if it fails again, replace the board of tray {u} (chip {o.group(1)}); logs {path_of(l, findings, stage)}",
        )
    if kind == "link" and not hang:  # a pair line of a chip that a one-chip line names: that chip, not this link
        k = LINK.search(l)
        for x in lines:
            o = ONECHIP.search(x)
            if o and k and o.group(1) in (k.group(1), k.group(4)):
                return f"one of the links of {o.group(1)} (tray {o.group(2)}), which fails all its C5 pairs: that chip, not this link -> next: see the one-chip FAIL of {o.group(1)} (tray {o.group(2)})"
    if "incomplete (test-infra)" in s:
        return "test infrastructure: the pair could not start (out of processes/threads: the job's ulimit -u or cgroup pids.max); not a chip or link fault -> next: rerun with a higher process limit; this link was not tested"
    if warn and "still hidden" in s and "port_status mask" in s:
        return "in-box ETH link hidden by the link-A repro's port_status mask, no reset after it: not a fault -> next: undo: tt-smi -glx_reset (a reset of this host's chips, then the root-port rescan and recount)"
    if warn and "reset marker slow to clear" in s:
        return (
            "chip slow to finish its warm reset (marker still set after 5 s)"
            + (
                "; it then timed out, see its RESET FAIL line"
                if "timed out" in s
                else ", cleared in time: no fault proven"
            ),
            f"watch it; a marker timeout (RESET FAIL) that repeats: tt-smi reset it again, then the board of tray {u}",
        )
    if kind == "link" and hang:
        return f"link pair hung: cable, connector or an end chip -> next: reseat or replace the cable; triage {path_of(l, findings, stage)}"
    if l.startswith("ETH INCOMPLETE "):
        w = re.search(r"; not checked: (.*)$", l)
        return (
            "external ETH port, not an in-box link on this platform; not checked" + (f": {w.group(1)}" if w else ""),
            "none on this box; it is judged once C2's reset can run",
        )
    if "not checked" in s:
        return f"numeric not checked (the readback failed): cause not determined -> next: rerun the {stage or 'stage'}; read {path_of(l, findings, stage)}"
    if hang and kind != "link" or kind == "unnamed":
        p = path_of(l, findings, stage)
        if kind == "unnamed":
            return f"hang or failure with no chip named -> next: read the hold capture {p}; rerun the stage alone to name the chip"
        return f"hang on the named chip(s): chip/board or its links, not determined -> next: read the hold capture {p}; rerun the stage"
    if not warn and "down at pre" in s and "only that reset retrained it" in s:
        cab = has_port(part_of(l, u)[1])
        t4 = re.search(r"SMC uptime ([^)]*)\)", s)
        return (
            "in-box ETH link down when the box was found (before our first reset) and retrained only by the start reset (C2): the link was not up as left by the last user"
            + (f"; the trays were reset at different times (SMC uptime {t4.group(1)})" if t4 else ""),
            ("reseat or replace that QSFP-DD cable" if cab else f"the board of tray {u}")
            + "; reset every tray together (tt-smi -glx_reset) so both ends retrain, then rerun",
        )
    if warn and "down at pre" in s and "recovered by" in s:
        cab = has_port(part_of(l, u)[1])
        return (
            "in-box ETH link down before the run, back after the start reset (C2): left down by an earlier run or a reset",
            "none now; "
            + ("reseat or replace that QSFP-DD cable" if cab else f"this link has no cable: the board of tray {u}")
            + " if it goes down again later in a run",
        )
    if kind == "link" or re.match(r"(ETH|LINK|PINPOINT LINK) FAIL|C[34] FAIL mesh placement", l):
        if warn:  # rc 143/137 is the pair job killed at its time limit, not a link fault
            rc = re.search(r"pair run failed, rc (\d+)", l)
            if rc and rc.group(1) in ("143", "137"):
                return f"not a link fault: the link pair's test job was killed at its time limit (rc {rc.group(1)}), so the link was not judged -> next: none for the hardware; rerun the link pair (C5) with a longer time limit"
            return "link run failed without a hang or mismatch: cause not determined", "rerun the link pair (C5); " + (
                "if it fails again with a hang or data mismatch, reseat or replace that QSFP-DD cable"
                if kind == "link" and has_port(part_of(l, u)[1])
                else "this link has no cable (in-box or on-board): if it fails again with a hang or data mismatch, the tray board"
            )
        q = bool(re.search(r"QSFP (ETH )?ports", l.split(";")[0]))
        nv = re.search(r"; no C2 verdict \((.*)\)$", l)
        to = re.search(r"local ETH ports \[[0-9, ]*\] not up \(to (" + XBDF + r" \(tray [1-4]\))", l)
        if nv and not q:
            return f"in-box ETH link{' to ' + to.group(1) if to else ''} down before the run, not rechecked: {nv.group(1)} -> next: rerun once C2's reset can run; if it stays down after a reset, reseat or replace the cable, then the board of tray {u}"
        inbox = not q and kind == "link" and not has_port(part_of(l, u)[1])
        rt = RETIMER_RE.search(l)
        if inbox and rt:
            return f"in-box link down or bad (no QSFP-DD port, no cable), chip side {rt.group(1)} with a training timeout, retimer speed not readable: the retimer speed setting, the tray-to-tray connection or a tray board -> next: check the retimer speed setting (expected {rt.group(1)}); if it is right, no cable to swap: reseat the trays at both ends; if it stays down after a reset, the board of tray {u}"
        if inbox:
            return f"in-box link down or bad (no QSFP-DD port, no cable): the tray-to-tray connection or a tray board -> next: no cable to swap: reseat the trays at both ends; if it stays down after a reset, the board of tray {u}"
        return (
            "link down or bad: QSFP-DD cable or connector" + (" (QSFP to another host)" if q else ""),
            "reseat or replace that QSFP-DD cable"
            + ("" if q else f"; if it stays down after a reset, the board of tray {u}"),
        )
    for pat, c, n in ETH_WARNS:
        if warn and l.startswith("ETH WARN") and pat in l:
            return c, n
    if warn and l.startswith("ETH WARN"):
        return (
            "ETH ports not up (warning) -> next: recheck after a reset; reseat or replace the cable if they stay down"
        )
    unopened = "(C9 end reset)" in l and "on PCIe with its tt-kmd node, but tt-umd cannot open it" in l
    dn = power_lost(lines, u) if l.startswith("RESET FAIL") and unopened else []
    if dn:
        t = " and ".join(f"tray {x}" for x in dn)
        return f"POWER-DOWNSTREAM: of the power drop of {t}: the C9 glx_reset then failed box-wide (in-band IPMI not answering), so this chip stays on PCIe with its tt-kmd node but does not open: not this tray's board -> next: fix {t} first (see its POWER line), then reset and recheck this chip"
    if l.startswith("RESET FAIL") and "not recovered by tray re-power/glx_reset" in l and "cannot open it" in l:
        return f"chip lost after its warm reset and not recovered by its tray re-power or glx_reset: chip/board (hardware), not a load or CCL fault -> next: power-cycle the box; if it stays unopenable, replace the board of tray {u}"
    if (
        l.startswith("RESET FAIL")
        and "not recovered by tray re-power/glx_reset" in l
        and "tt-umd discovery fails on it" in l
    ):
        return f"chip lost after its warm reset and not recovered by its tray re-power or glx_reset: it is on PCIe and opens, but tt-umd discovery fails on it: chip/board (hardware), not a load or CCL fault -> next: power-cycle the box; if tt-umd discovery still fails on it, replace the board of tray {u}"
    if re.search(r"no such device|missing after|not answering|unknownx|lost from pcie", s) or l.startswith("CHIP DROP"):
        return f"chip fell off PCIe: chip/board or its PCIe path (slot, retimer) -> next: read the AER/lspci capture; same chip again: replace the board of tray {u}; all of tray {u} at once: bus bar first"
    if "failed on it" in s and l.startswith("RESET"):
        return f"chip did not take the warm reset: chip/board -> next: tt-smi reset it again; if it repeats, replace the board of tray {u}"
    w = re.search(r"corrected AER peak (\d+)/min \(over ([0-9.]+)/min", l) if warn and "weak PCIe link" in l else None
    if w:
        return f"weak PCIe link: {w.group(1)} corrected AER errors a minute under load on its root port (warn over {w.group(2)}/min): slot, retimer or board, an early sign of a chip that falls off PCIe -> next: read the stage's aer_rate.txt and the lspci capture; reseat tray {u}; if it repeats, replace the board of tray {u}"
    w = re.match(
        r"AER FAIL .*? correctable kernel AER events (during the run \(.*?\)|over 7 days|since repair .*?)(, last [^=]*?)? = ",
        l,
    )
    if w:
        return f"correctable AER rate outlier {w.group(1)}{w.group(2) or ''} in the kernel journal{'' if w.group(1).startswith('during the run') else ', not only this run'}: PCIe path (chip, slot, retimer or link) -> next: read the AER/lspci capture; reseat tray {u}; if it repeats, replace the board of tray {u}"
    if "SETUP/SOFTWARE: fabric setup failed" in l:
        return f"test infrastructure: fabric setup failed (setup or software: fabric descriptors or the tt-metal tree), not this tray's hardware -> next: fix the setup error the run's log names, then run {RERUN} again"
    if re.match(r"(PCIE|AER) FAIL|PREFLIGHT FAIL .*PCIe", l):
        return f"PCIe path (link speed/width or uncorrectable errors): slot, retimer or board -> next: reseat tray {u}; if it repeats, replace the board of tray {u}"
    if l.startswith("NUMERIC FAIL") and "CCL group fault" in l:
        src = ccl_source(l, lines)
        if not src:
            return "CCL group fault: one bad result spread over the group's all-gather, source not isolated (no chip of the group failed a check outside the CCL) -> next: rerun the numeric pinpoint on every chip of the group; replace only the board of a chip that fails it"
        n = ", ".join(f"{b} (tray {t})" for b, t in src)
        ts = sorted({t for _, t in src})
        return (
            f"CCL group fault: one bad result from the likely source {n} spread over the group's all-gather"
            + ("" if u in ts else f" (tray {u}'s chips only received it)")
            + f": chip/board of {n}",
            f"rerun the numeric pinpoint on {n}; if it repeats, replace the board of tray {'/'.join(ts)}",
        )
    if re.match(r"(NUMERIC|CHUNK|VAE) FAIL|C4v? FAIL", l):
        return numeric_cause(l, u, s)
    if l.startswith("QUIET FAIL"):
        return "chip stayed above idle clock in the quiet: cause not determined -> next: read tt-smi telemetry of that chip; rerun"
    if l.startswith("PREFLIGHT FAIL") and re.search(r"_ver 255\.255\.255\.255\b", l):
        return "chip not answering over PCIe (all-Fs reads), not a firmware mismatch -> next: glx_reset and recount; if it repeats after power cycles, the tray's PCIe path"
    if l.startswith("PREFLIGHT FAIL") and "_ver" in s:
        return f"firmware version differs from the other chips -> next: reflash tray {u} to the box's firmware bundle"
    if l.startswith("LOAD LOW WARN "):
        return f"the stage's load stayed under 90% of tray {u}'s chips' own power limits for most of it: a clean result there is not a clean pass, a weak tray may not have been loaded hard enough to drop -> next: no fault proven; the load mix toward each chip's own limit is the tuning step"
    if l.startswith("LOAD UNREAD WARN "):
        return f"tray {u}'s chip power or power limit was not read for enough of the load: whether it was loaded to its limit is not known (neither a pass nor a failed check) -> next: read tray {u}'s hwmon power1_input and power1_max; rerun"
    if warn:
        return "warning only, no fault proven -> next: watch it in the next run"
    return f"cause not determined -> next: {rerun} with its full log captures on this host; read {findings}"


ETH_WARNS = (  # (text in an ETH WARN line, cause, next)
    (
        "leave the box",
        "ETH ports leave the box, not cabled on a standalone galaxy (warning)",
        "none: the port leaves the box (no cable on a standalone galaxy)",
    ),
    (
        "not judged: no link map",
        "inter-host QSFP ETH ports down, not judged: the run read no link map (warning)",
        "none until a link discovery reads; then judged from the link map",
    ),
    (
        "dropped by our own reset on",
        "inter-host QSFP port up after the C2 reset and dropped by our own later reset: test-side, not this host's fault (warning)",
        "none on this host",
    ),
    (
        "off-box: far end not in this run",
        "inter-host QSFP ETH port down before and after the reset, far end not in this run: not judged (warning)",
        "check the QSFP-DD cable or the far host; judge it in a run that includes the far host",
    ),
    (
        "ETH WARN off-box ports",
        "off-box QSFP ETH ports down, no cabling map links them to a host in this run: not judged (warning)",
        "none on this host; check them in a run that includes the peer host",
    ),
    (
        "far end down or resetting",
        "inter-host QSFP port down on chips of several trays at once: far end down or resetting, not this host's fault (warning)",
        "none on this host; recheck once the far host is up",
    ),
    (
        "MAXUTIL not run: device open failed",
        "ETH core of this chip not active at device open, so the bus-bar stage max-util did not run on any chip (warning only; run INCOMPLETE): its ETH link, not a power or load fault",
        "rerun; if the same ETH core times out at device open again, check that chip's link as an ETH fault",
    ),
)


def numeric_cause(l, u, s):
    side = compute_side(l)
    if side:
        return f"chip compute fault (or its power under load): chip/board; {side}, not a DRAM bank -> next: rerun the numeric pinpoint on this chip; if it repeats, replace the board of tray {u}"
    bd, cr = re.search(r"FAIL " + XBDF, l), column_read(l)
    if bd and cr:
        return (
            f"chip read fault: one bad read of tile column {cr[0]} reached every core of core column {cr[1]} (compute cores "
            + ", ".join(f"({cr[1]},{y})" for y in cr[2])
            + "), not one compute core: chip/board",
            f"rerun the numeric pinpoint on ASIC {bd.group(1)} with the DRAM readback split and the decoupled arm to tell its memory from its NoC; if it repeats, replace the tray {u} board",
        )
    cc = re.search(r"compute core \((\d+),(\d+)\)", l)
    if cc:
        bd = re.search(rf"({BDF})", l)
        a_ = f"ASIC {bd.group(1)} " if bd else ""
        el = re.search(r"element \((\d+),(\d+)\)", l)
        wc = re.search(r"compute core \(\d+,\d+\) \(translated worker core (\d+),(\d+)\)", l)
        return (
            f"chip compute fault in {a_}compute core ({cc.group(1)},{cc.group(2)})"
            + (f" (translated worker core {wc.group(1)},{wc.group(2)})" if wc else "")
            + (f", element ({el.group(1)},{el.group(2)}) of its output tiles" if el else "")
            + ": tray board",
            f"replace the tray {u} board ({a_}compute core ({cc.group(1)},{cc.group(2)}))",
        )
    d = "dram" in s.replace("in-dram matmul", "") or ("banks [" in s and " rw)" in s)
    return f"chip {'DRAM' if d else 'compute'} fault{decoupled_bank(l) if d else ''}: chip/board -> next: rerun the numeric pinpoint on this chip; if it repeats, replace the board of tray {u}"


def compute_side(l):
    m = re.search(r"lanes even/odd (\d+)/(\d+)", l)
    if "columns decoupled from banks" in l:
        b = re.search(r"banks \[([\d, ]+)\]", l)
        v = chans(b.group(1)) if b else []
        if v and sum(v) and max(v) >= CONC * sum(v):
            return None
    if m:
        e, o = int(m.group(1)), int(m.group(2))
        if e + o and max(e, o) >= LANE * (e + o):
            return f"bad words on {'odd' if o > e else 'even'} lanes {max(e, o)}/{e + o}, they follow output columns"
    c = re.search(r"output column groups \[([\d, ]+)\]", l)
    if c:
        v = chans(c.group(1))
        if len(v) > 1 and sum(v) and max(v) >= COL_SKEW * sum(v) / len(v):
            return (
                f"bad words in output column group {v.index(max(v))} {round(100 * max(v) / sum(v))}% of {len(v)} groups"
            )
    return None


def decoupled_bank(l):
    b = re.search(r"banks \[([\d, ]+)\]", l)
    v = chans(b.group(1)) if b and "columns decoupled from banks" in l else []
    if not v or not sum(v) or max(v) < CONC * sum(v):
        return ""
    return f" in GDDR bank {v.index(max(v))} ({max(v)}/{sum(v)} bad words, columns decoupled from banks)"


def tile_cols(l):
    p = re.search(r"element row,element col\) (\[\[[\d, \[\]]*\]\])", l)
    if p:
        return [int(t[3]) for t in json.loads(p.group(1))]
    t = re.search(r"of tile \d+,(\d+)|\(tile \d+,(\d+),", l)
    return [int(t.group(1) or t.group(2))] if t else []


def column_read(l):
    c = re.search(r"compute cores x,y,n (\[\[[\d, \[\]]*\]\])", l)
    g = re.search(r"of the (\d+)x(\d+) grid", l)
    tc = set(tile_cols(l))
    if not c or len(tc) != 1:
        return None
    cores = json.loads(c.group(1))
    xs, ys = {k[0] for k in cores}, {k[1] for k in cores}
    if len(xs) == 1 and len(ys) >= max(2, min(int(g.group(2)), 8) if g else 2):
        return tc.pop(), xs.pop(), sorted(ys)
    return None


def result(lines, stages=None, drop_trays=None, known=None):
    stages, seen, out, trays = stages or {}, set(), [], []
    for u in "1234":
        if known and u == str(known):
            continue
        for l in tray_lines(lines, u, drop_trays, stages)[0]:
            if known and set(re.findall(r"tray ([1-4])(?![0-9])", re.sub(r", BMC after: .*", "", l))) == {str(known)}:
                continue
            if not l.startswith("NUMERIC FAIL"):
                trays.append(f"tray {u}")
            if l in seen:
                continue
            seen.add(l)
            if not l.startswith("NUMERIC FAIL"):
                out.append(f"tray {u}")
                continue
            k = re.search(r"(\d+/\d+) (checks|launches)", l)
            me = re.search(r"max abs err (\S+), wrong elements (\d+), (all finite|non-finite values present)", l)
            ev = ", ".join(
                x
                for x in (
                    f"{k.group(1)} {k.group(2)}" if k else "",
                    stage_of(l, stages),
                    f"max abs err {me.group(1)}, {me.group(2)} wrong elements, {me.group(3)}" if me else "",
                )
                if x
            )
            chips = ", ".join(f"{b} tray {t}" for b, t in dict.fromkeys(CHIP.findall(l))) or f"tray {u}"
            out.append(f"numeric fault {chips}" + (f" ({ev})" if ev else ""))
    num = [o for o in out if o.startswith("numeric fault")]
    n = len(out) - len(num)
    return "; ".join(num + ([f"{n} failed check{'s' * (n > 1)} on {', '.join(dict.fromkeys(trays))}"] if n else []))


def untag(lines, stages):
    tags, out = {}, []
    for l in lines:
        m = AFTER.search(l)
        if m:
            l = l[: m.start()]
            tags.setdefault(l, m.group(1))
        out.append(l)
    st = {}
    for k, v in stages.items():
        st.setdefault(AFTER.sub("", k), v)
    return out, st, tags


def why(lines, u, stages=None, cap=10, findings="findings.txt", drop_trays=None, tags=None):
    u, stages, tags = str(u), stages or {}, tags or {}
    fails, warns = tray_lines(lines, u, drop_trays, stages)
    fails.sort(key=lambda l: (not l.startswith(("POWER ", "HSC FAIL ")), not l.startswith("HANG PINPOINT")))
    out = []
    for l in fails + warns:
        tg = tags.get(l)
        l = re.sub(r"UBB([0-3])", lambda m: f"tray {int(m.group(1)) + 1}", l)
        kind, part = part_of(l, u)
        st = stage_of(l, stages)
        c, n = cause(l, kind, u, findings, st, lines)
        st = f"{st}, {tg}" if tg else st
        o = ONECHIP.search(l)
        ev = f"all {o[3]} failing {o[4]} pairs have this chip at one end ({o[5]})" if o else evidence(l, u)
        out.append(f"    {part} | {st} | {ev} | cause: {c} -> next: {n}")
        if o:  # its links one per line under it
            out += [f"      link {a} e{ca} <-> {b} e{cb}: {w}" for a, ca, b, cb, w, _ in ONELINK.findall(o[6])]
            t = re.search(r"tt-metal errors on this chip in .*?(?= \(/[^)]*\)$|$)", o[6])
            out += [f"      {t.group(0).strip()}"] if t else []
    if len(out) > cap:
        gone = list(dict.fromkeys(b for l in out[cap:] for b in loss_bdfs(u, l)))
        out = out[:cap] + [
            f"    ... {len(out) - cap} more for tray {u} in {findings}"
            + (f"; lost chips: {' '.join(gone)}" if gone else "")
        ]
    return out


def tray_block(raw, raw_stages, findings="findings.txt", known=None, stopped=""):
    """One 'FAIL|INCOMPLETE|PASS|KNOWN-DROP tray N (...)' line per tray, each finding's cause line under it."""
    lines, stages, tags = untag(raw, raw_stages)
    dt = sorted(thru(lines, re.compile(r"^(?:POWER|CHIP) DROP tray ([1-4])")))
    dead = {stages.get(l) for l in lines if re.match(r"(POWER|CHIP) DROP tray ", l)} | {None}
    wb = next((l[16:156] for l in lines if l.startswith("HANG INCOMPLETE whole box")), "")
    out = []
    for u in "1234":
        if u == str(known):
            out.append(f"KNOWN-DROP tray {u} (known dropped, not tested)")
            continue
        fails, warns = tray_lines(lines, u, dt, stages)
        nf, nw, mine = len(fails), sum(" WARN " in l for l in warns), [l for l in lines if hit(u, l)]
        ni = sum("INCOMPLETE (test-infra)" in l for l in mine)
        nv = sum(bool(INC.match(l)) and "start reset skipped; ETH not-active-at-open not judged" in l for l in mine)
        nd = next(
            (
                m[1]
                for l in mine
                if u not in dt and DIED.search(l) and stages.get(l) in dead
                for m in [re.search(r"CHUNK FAIL (chunk \d+/\d+) \(started [0-9:]+\): no chunk-end readback", l)]
                if m
            ),
            "",
        )
        x = rf"^ETH INCOMPLETE (\S+) \(tray {u}\) [^:]*: external ETH ports \[([0-9, ]*)\] not up; not checked: ([^(]*)"
        xs = [
            f"{m[2]} of {m[1]} not checked: {m[3]}".replace(", ", ",").rstrip()
            for m in map(re.compile(x).match, lines)
            if m
        ]
        xs = ";".join(dict.fromkeys(xs)).replace(";", "; ")
        if not xs and any(INC.match(l) and "start reset skipped" in l and "not-active-at-open" not in l for l in mine):
            xs = "inter-host ports not judged: start reset skipped"
        w = f", {nw} warn" if nw else ""
        if nf:
            out.append(f"FAIL tray {u} ({nf} finding{'s' * (nf > 1)}{w})")
        elif ni:
            out.append(
                f"INCOMPLETE tray {u} ({ni} C5 pair(s) not run: test-infra{f', {nw - ni} warn' if nw > ni else ''})"
            )
        elif nd:
            out.append(
                f"INCOMPLETE tray {u} (lost nothing; numeric not checked: the load of {nd} died when tray {' and tray '.join(dt)} lost chips{w})"
            )
        elif xs:
            out.append(f"INCOMPLETE tray {u} ({'' if xs.startswith('inter-host') else 'external ETH ports '}{xs}{w})")
        elif nv:
            out.append(
                f"INCOMPLETE tray {u} ({nv} ETH core(s) not active again at mesh open: start reset skipped; ETH not-active-at-open not judged{w})"
            )
        else:
            out.append(
                f"INCOMPLETE tray {u} ({stopped}{w})"
                if stopped
                else f"INCOMPLETE tray {u} (whole-box hang, no chip named, no hardware verdict: {wb}{w})"
                if wb
                else f"PASS tray {u}" + (f" ({nw} warn)" if nw else "")
            )
        if nf + nw + nv or xs:
            out += why(lines, u, stages, 10, findings, dt, tags)
    return out


# ── final verdict: fault families, then each fault (what | where | do) and the notes ────────────────────────────────

RESET = {"c2": "start (C2)", "c9": "end (C9)"}
CAP = 12  # faults printed; the rest are counted
# family -> (what it means physically, the finding classes in it: the words before FAIL/WARN/HANG/DROP); others: other
FAMILIES = {
    "power/feed": (
        "a tray's 12 V supply sagged or tripped under load: the power bricks on the PDB, the bus bar, its contacts or the tray's hot-swap controller",
        ("POWER", "POWER VIRUS", "PV", "PV HOST CRASH", "HSC", "PSU", "TDP"),
    ),
    "math": (
        "a chip computed wrong results under load: the chip itself or its voltage margin",
        ("NUMERIC", "MAXUTIL", "L1", "CODE"),
    ),
    "link": (
        "data between chips over Ethernet was lost, wrong or stalled: the QSFP-DD cable or in-box connection, or a chip's Ethernet port",
        ("ETH", "ETH LINK", "ETH FEC", "LINK", "LINKS", "LINK-A", "LINK-DATA", "PINPOINT", "PINPOINT LINK", "CCL")
        + ("FABRIC", "ABORT-FABRIC", "REPRO ABORT-FABRIC", "REPRO LINK-A", "H3SHAPE", "C4"),
    ),
    "PCIe": (
        "a chip's PCIe connection to the host erred or dropped: the PCIe path from the MB module through the PDB retimers, or the chip's tray board",
        ("PCIE", "H3 PCIE", "AER", "CHAIN", "CHIP", "POST-BOOT", "DROP", "C3", "RESET", "RESET UNATTRIBUTED"),
    ),
    "thermal": (
        "a part ran over its temperature limit: fans and airflow first, then the part named",
        ("THERMAL", "SEL THERMAL"),
    ),
    "DRAM": (
        "a chip's GDDR memory returned wrong data or reported errors: that chip's GDDR on its tray board",
        ("GDDR",),
    ),
}
FAMILY = {c: f for f, (_, cs) in FAMILIES.items() for c in cs}


def family(l):
    m = re.match(r"^((?:[A-Z][A-Z0-9-]*\s)*?[A-Z][A-Z0-9-]*) (?:FAIL|WARN|HANG|DROP)\b", l)
    c = m.group(1) if m else ""
    if c == "NUMERIC" and re.search(r"\[[^\]]*\b(dram|dbi|rw)\b", l):
        return "DRAM"
    return FAMILY.get(c, "other")


def end(b, chs, lone=False):
    t, a = tray_asic(b)
    ch = f" [ETH ch {ids(chs)}]" if chs else ""
    if t is None:  # a BDF outside the tray map names no tray or ASIC
        return f"{b}{ch}"
    ps = ports(b, chs)
    if ps:
        return f"tray {t} QSFP-DD port {' and '.join(map(str, ps))} / ASIC {a} / {b}{ch}"
    return f"tray {t} ASIC {a} / {b}{ch}" + (" (no QSFP-DD port)" if lone else "")


def link_where(a, ca, b, cb):
    ta, tb = tray_asic(a)[0], tray_asic(b)[0]
    w = f"{end(a, ca)} <-> {end(b, cb)}"
    if ports(a, ca) and ports(b, cb):
        return "QSFP-DD cable " + w, "cable"
    if ta == tb and ta is not None:
        return f"on-board link inside tray {ta} (no QSFP-DD port): " + w, "onboard"
    return f"in-box link between {holder(a)} and {holder(b)} (no QSFP-DD port): " + w, "inbox"


def link_do(kind, a, b, ca=(), cb=()):
    ta, tb = tray_asic(a)[0], tray_asic(b)[0]
    if kind == "cable":
        return f"reseat, then replace, the QSFP-DD cable from tray {ta} port {'/'.join(map(str, ports(a, ca)))} to tray {tb} port {'/'.join(map(str, ports(b, cb)))}; if it stays down with a good cable, the port on one of the two trays"
    if kind == "onboard":
        return f"no cable to swap: {board(a)} (link between ASIC {tray_asic(a)[1]} and ASIC {tray_asic(b)[1]})"
    return f"no cable to swap: reseat {f'trays {ta} and {tb}' if ta and tb else f'{holder(a)} and {holder(b)}'}; if it stays down, the tray-to-tray connection or one of the two tray boards (swap a tray to see which end follows)"


def reset_words(labels):
    ws = [RESET.get(x, x.upper()) for x in sorted(set(labels), key=lambda x: (x not in RESET, x))]
    return " and ".join(ws) + (" resets" if len(ws) > 1 else " reset")


def resets(rest):
    r = rest.lower()
    if "no c2 verdict" in r or "reset skipped" in r:
        return None
    c2, c9 = "still down after the c2 reset" in r, "c9 reset: still down" in r or "still down after the c9 reset" in r
    if c2 or c9:
        return "still down after the " + reset_words(["c2"] * c2 + ["c9"] * c9)
    return "came back after a reset (flaky)" if "back after" in r or "came back" in r else "down"


class V:
    def __init__(self):
        self.faults, self.notes, self.seen, self.fw, self.down = [], [], set(), {}, {}
        self.tag, self.pdn, self.pdn_chips, self.tripped = "", [], [], set()
        self.fam, self.lfam, self.rerun = {}, "other", ""

    def fault(self, key, what, where, do):  # a later line on the same part (the C9 recheck) replaces the earlier one
        where += f" [{self.tag}]" if self.tag else ""
        if key not in self.seen:
            self.seen.add(key)
            self.fam[key] = self.lfam
            self.faults.append((key, None))
        self.faults = [(k, (what, where, do) if k == key else f) for k, f in self.faults]

    def note(self, key, text):
        if key not in self.seen:
            self.seen.add(key)
            self.notes.append(text + (f" [{self.tag}]" if self.tag else ""))


def marked(rest, tag):
    i = rest.find("; " + tag)
    if i < 0:
        return set()
    seg = rest[i + 2 :].split("; ", 1)[0]
    return {(b, int(c)) for b, cs in re.findall(rf"({BDF}) \[ETH ch ([0-9,]+)\]", seg) for c in cs.split(",") if c}


READERS = (
    []
)  # (pattern, reader): a finding line's reader is the first whose pattern matches; one returning False passes


def reads(pattern):
    return lambda f: READERS.append((re.compile(pattern), f)) or f


@reads(
    rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\) (\w+): local ETH ports \[([0-9, ]*)\] (?:enabled but )?not up \(to ((?:[^()]|\([^()]*\))*)\)(.*)$"
)
def eth(v, m):
    a, rest, by = m[2], m[6], {}
    for c, (b, pc) in zip(chans(m[4]), re.findall(rf"({BDF}) \(tray [1-4]\) port (\d+)", m[5])):
        by.setdefault(b, ([], []))[0].append(c)
        by[b][1].append(int(pc))
    st = resets(rest)
    cf = re.search(r"; CONFIG: retimer/ETH speed (x\d+G(?: and x\d+G)?), expected (x\d+G)", rest)
    rt = re.search(r"; RETIMER: chip side (x\d+G), retimer speed not readable", rest)
    for b, (ca, cb) in by.items():
        hit_ = lambda tag: any((a, c) in marked(rest, tag) for c in ca) or any((b, c) in marked(rest, tag) for c in cb)
        x, y = sorted([(a, tuple(ca)), (b, tuple(cb))])
        where, kind = link_where(x[0], list(x[1]), y[0], list(y[1]))
        do = link_do(kind, x[0], y[0], x[1], y[1])
        if st is None:
            w = re.search(r"\(C2 reset skipped: ([^:)]*)", rest)
            v.note(
                ("eth", x, y),
                f"Ethernet link down before the run, not judged: the start (C2) reset was skipped{' (' + w.group(1) + ')' if w else ''} | where: {where} | do: rerun {v.rerun} with the start reset",
            )
        elif m[1] != "FAIL":
            pass
        elif cf and hit_("CONFIG:"):
            v.fault(
                ("eth", x, y),
                f"CONFIG: retimer/ETH speed {cf.group(1)}, expected {cf.group(2)}: Ethernet link down, {st}",
                where,
                f"set the retimers to {cf.group(2)} and rerun, before any cabling or board step",
            )
        elif rt and hit_("RETIMER:"):
            v.fault(
                ("eth", x, y),
                f"Ethernet link down, {st}; chip side {rt.group(1)} with a training timeout, retimer speed not readable",
                where,
                f"check the retimer speed setting (expected {rt.group(1)}) first, then " + do,
            )
        else:
            v.fault(("eth", x, y), f"Ethernet link down, {st}", where, do)


@reads(
    rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\) (\w+): (\d+) (QSFP cables?|wrap cables?) missing or bad: (.*?)(?: \(layout ([\w-]+)\))?(;.*)?$"
)
def cable(v, m):
    wrap = m[5].startswith("wrap")

    def port(x):
        p = [k for (pt, k), ls in UBB_ETH_PORTS.items() if pt == "QSFP_DD" and k <= 6 and ls[0][0] == tray_asic(x)[1]]
        return ("2" if wrap else "2 or 3") if len(p) > 1 else (str(p[0]) if p else "?")

    for a, b, ta, tb in re.findall(rf"({BDF})<->({BDF}) \(tray ([1-4])<->([1-4])\)", m[6]):
        v.fault(
            ("cable", a, b),
            f"{'wrap ' if wrap else ''}QSFP-DD cable missing or bad (cabling judged against layout {m[7] or '?'})",
            f"tray {ta} QSFP-DD port {port(a)} / ASIC {tray_asic(a)[1]} / {a} <-> tray {tb} QSFP-DD port {port(b)} / ASIC {tray_asic(b)[1]} / {b}",
            "check that cable is plugged in at both ends, then reseat or replace it (cabling, for the DC)",
        )


@reads(
    rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: enabled ETH ports \[([0-9, ]*)\] down before the run and still down after the (\w+) reset"
)
def ethboth(v, m):  # tt-smi ETH port numbers: ports 0-7 = ch 0-7, ports 10-13 = ch 8-11, ports 8 and 9 no channel
    b, ps = m[2], tuple(chans(m[3]))
    t = tray_asic(b)[0]
    if m[1] != "FAIL":
        return
    k = ("ethboth", t, ps)
    bs, rs = v.down.setdefault(k, ([], set()))
    bs.append(b) if b not in bs else None
    rs.add(m[4].lower())
    smi = {p: p if p < 8 else p - 2 if 10 <= p <= 13 else None for p in ps}
    cs = [c for c in smi.values() if c is not None]
    nc = ",".join(str(p) for p, c in smi.items() if c is None)
    lab = "; ".join(
        x for x in (f"ETH ch {ids(cs)}" if cs else "", f"tt-smi ETH port {nc} (no channel)" if nc else "") if x
    )
    v.fault(
        k,
        f"Ethernet ports down before the run, still down after the {reset_words(rs)}",
        (f"tray {t} " + ", ".join(f"ASIC {tray_asic(x)[1]} / {x}" for x in bs) if t else ", ".join(bs))
        + f" [{lab}]"
        + ("" if any(ports(x, cs) for x in bs) or t is None else " (no QSFP-DD port)"),
        f"glx_reset and retest; if they stay down, {board(b)}",
    )


QSFP_DO = (
    "reseat, then replace, that QSFP-DD cable; if it stays down with a good cable, the port on {} or on the far host"
)


@reads(
    rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: inter-host QSFP ETH ports \[[0-9, ]*\] \(ch \[([0-9, ]*)\]\) down before the run and still down after the (\w+) reset"
)
def interhost(v, m):
    b, cs = m[2], chans(m[3])
    if m[1] == "FAIL":
        v.fault(
            ("qsfp", b, tuple(cs)),
            f"QSFP-DD link to another host down before the run, still down after the {reset_words([m[4].lower()])}",
            "QSFP-DD cable " + end(b, cs, True) + " <-> another host",
            QSFP_DO.format(board(b)),
        )


@reads(
    rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: QSFP ETH ports \[([0-9, ]*)\] not up(?: \(to ([\w-]+) tray [1-4] ({BDF}) ports \[([0-9, ]*)\]\))?(.*)$"
)
def qsfp(v, m):
    b, cs = m[2], chans(m[3])
    far = f" <-> {m[4]} {end(m[5], chans(m[6]), True)}" if m[4] else " <-> another host"
    if m[1] == "FAIL":
        v.fault(
            ("qsfp", b, tuple(cs)),
            "QSFP-DD link to another host down"
            + (" while the far end is up and idle" if "up and idle" in m[7] else ""),
            "QSFP-DD cable " + end(b, cs, True) + far,
            QSFP_DO.format(board(b)),
        )


@reads(
    rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: ETH core ([0-9-]+) \(channel (\d+)[^)]*\) not active again at mesh open(.*)$"
)
def ethcore(v, m):
    b, c = m[2], int(m[4])
    if m[1] == "FAIL":
        v.fault(
            ("ethcore", b, c),
            f"Ethernet core {m[3]} not active again when the mesh opened",
            end(b, [c], True),
            f"glx_reset and retest; if it repeats, {board(b)} carrying {b}",
        )


@reads(rf"^NUMERIC (FAIL|WARN) ({BDF}) \(tray [1-4]\) chip (\d+) \[([^\]]*)\]: (.*)$")
def numeric(v, m):
    b, idx, kind, t = m.group(2, 3, 4, 5)
    k = (
        re.search(r"(\d+/\d+) checked outputs mismatched", t)
        or re.search(r"(\d+ mismatching chunks of \d+)", t)
        or re.search(r"(\d+/\d+) (?:launches )?mismatched", t)
    )
    hist = re.search(r"DRAM bank histogram \[([0-9, ]+)\]", t)
    banks = [str(i) for i, n in enumerate(chans(hist.group(1))) if n] if hist else re.findall(r"DRAM bank (\d)", t)[:1]
    stage = re.search(r"\((stage \d+[^:)]*)", t)
    known = re.search(r"matches ([A-Z]+-\d+ \([^)]*\))", t)
    what = "wrong compute results (numeric check" + (f", {kind.split(',')[0]} load" if kind else "") + ")"
    if k:
        what += f": {k.group(1).replace('mismatching chunks', 'chunks wrong')}" + (
            "" if "chunks" in k.group(1) else " outputs wrong"
        )
    if m[1] == "FAIL":
        v.fault(
            ("num", b),
            what,
            chip(b, idx)
            + (f", DRAM bank {','.join(banks)}" if banks else "")
            + (f" ({stage.group(1)})" if stage else ""),
            f"hardware only if the same check on the same commit stays clean on clean hosts; then {board(b)} carrying {b}"
            + (f"; matches known issue {known.group(1)}" if known else ""),
        )


def fw_why(t, x, y, mj):
    if x == "unread":
        return (
            f"tray {t}'s firmware version could not be read while the other chips run {y}",
            f"check that tray {t}'s chips answer, reflash tray {t} to {y} if its firmware differs",
        )
    if mj:
        return (
            f"tray {t} runs firmware {x}; the chips run mixed firmware with no version on most of them ({y} on {mj} chips)",
            "reflash every chip to one firmware version",
        )
    return f"tray {t} runs firmware {x} while the other chips run {y}", f"reflash tray {t} to {y}"


@reads(
    rf"^PREFLIGHT (FAIL|WARN) ({BDF}) \(tray ([1-4])\) chip \d+: tt_fw_bundle_ver ([\w.]+), (\d+ of \d+) chips ([\w.]+)( \(no majority\))?"
)
def preflight(v, m):
    t = int(m[3])
    x, y, mj, bs = v.fw.get(t, (m[4], m[6], m[5] if m[7] else "", []))
    v.fw[t] = (x, y, mj, bs + [m[2]])
    w, do = fw_why(t, x, y, mj)
    v.fault(("fw", t), f"firmware mismatch: {w}", f"tray {t} ({len(v.fw[t][3])} chips)", f"{do}, then retest")


@reads(r"^CHUNK FAIL chunk (\d+/\d+)(?: \([^)]*\))?: no chunk-end readback, the load died or hung(.*)$")
def chunk(v, m):
    tr = sorted(set(re.findall(r"tray ([1-4]):", m[2])))
    v.fault(
        ("chunk", m[1]),
        f"the load died or hung (chunk {m[1]} gave no end readback; numeric not checked)",
        ("trays " + ", ".join(tr)) if tr else "the box",
        "see the cause line and the Repair recommendation",
    )


@reads(rf"^AER (FAIL|WARN) ({BDF}) \(tray [1-4]\):? (.*)$")
def aer(v, m):
    b, t = m[2], m[3]
    n = re.search(r"(\d+) (correctable|uncorrectable)", t)
    if re.search(r"history before the run|not only this run|over \d+ days|before the run", t):
        v.note(
            ("aer", b),
            f"old PCIe (AER) errors on {chip(b)} from the kernel log before this run{' (' + n.group(0) + ' events)' if n else ''}, none during this run: not a failure",
        )
    elif m[1] == "FAIL":
        v.fault(
            ("aer", b),
            f"PCIe errors during the run (AER{': ' + n.group(0) if n else ''})",
            chip(b),
            f"the PCIe path of {b} (slot, retimer, cable), then {board(b)}",
        )


@reads(rf"^PCIE (FAIL|WARN) ({BDF}) \(tray [1-4]\)(.*?)(?:, root port ([0-9a-f:.]+))?(?:, host [\w-]+)?: (.*)$")
def pcie(v, m):
    b, t = m[2], m[5]
    sp = re.search(r"root port link ([\w.]+ x\d+)", t)
    if m[1] != "FAIL":
        return
    if "off pcie" in t.lower():
        what = "chip off PCIe (config space reads ffff)" + (f", link at {sp.group(1)}" if sp else "")
        do = f"glx_reset and recount; if still missing, power cycle the box; then {board(b)} or its PCIe slot"
    else:
        what = "weak PCIe link" + (f" ({sp.group(1)})" if sp else f": {t[:80]}")
        do = f"the PCIe path of {b} (slot, retimer), then {board(b)}"
    v.fault(("pcie", b), what, chip(b) + (f" (root port {m[4]})" if m[4] else ""), do)


@reads(rf"^WORKAROUND APPLIED: ({BDF}) ")
def workaround(v, m):  # a deliberate, recorded setting (a PCIe speed cap): a note, never a fault
    v.note(("workaround", m[1]), m.string)


@reads(
    r"^(CHIP|POWER) DROP tray ([1-4]) \((\d+) chips lost(.*?)\) ((?:in|after|at) .*?)( \[still missing from an earlier drop\])?$"
)
def drop(v, m):
    t, bs = m[2], list(dict.fromkeys(re.findall(BDF, re.sub(r"\(root port [^)]*\)", "", m[4]))))
    ga = re.search(r"(\d+) chips of the tray missing before its glx_reset judged", m[4])
    v.fault(
        ("drop", m.string[: m.start(6) if m[6] else None][:120], tuple(bs), bool(m[6])),
        f"{'chip' if m[1] == 'CHIP' else 'tray power'} drop: {m[3]} chips lost on tray {t} {m[5][:120]}"
        + (f"; {ga.group(1)} chips of the tray were missing in all when it was judged" if ga else ""),
        "; ".join(chip(b) for b in bs) or f"tray {t}",
        f"see the Repair recommendation line (tray {t})",
    )


@reads(rf"^LINK (FAIL|HANG) ({BDF}) \(tray [1-4]\) e([0-9,]+) <-> ({BDF}) \(tray [1-4]\) e([0-9,]+): (.*)$")
def link(v, m):
    x, y = sorted([(m[2], tuple(chans(m[3]))), (m[4], tuple(chans(m[5])))])
    where, kind = link_where(x[0], list(x[1]), y[0], list(y[1]))
    w = re.search(r"(\d+) words wrong", m[6])
    what = (
        "Ethernet link hung (no progress)"
        if m[1] == "HANG"
        else "wrong data over an Ethernet link" + (f" ({w.group(1)} words wrong)" if w else "")
    )
    v.fault(("link", x, y), what, where, link_do(kind, x[0], y[0], x[1], y[1]))


@reads(rf"^LINK FAIL one chip ({BDF}) \(tray [1-4]\) tt-smi (\d+): (.*)$")
def onechip(v, m):
    v.fault(
        ("chip", m[1]),
        "every link test through this one chip failed",
        chip(m[1], m[2]),
        f"{board(m[1])} carrying {m[1]}",
    )


@reads(r"^THERMAL FAIL ")
def thermal(v, m):
    if not thermal_parse(m.string):
        return False
    scope, n, sensor, rd, pb, part, step = thermal_parse(m.string)
    ev = f"BMC sensor {sensor}" + (f" read {rd} C, above its upper critical limit" if rd else "")
    if pb:
        v.fault(
            ("brick", pb),
            f"power brick {pb} overheated during the run ({ev})",
            part + ("" if scope == "host" else f" ({scope})"),
            step,
        )
    else:
        v.fault(
            ("thermal", sensor),
            f"over-temperature trip during the run ({n} BMC event(s))",
            f"{scope}, {ev} ({part})",
            f"{step}; the chips' own checks are listed separately",
        )


@reads(PDN_CHIP.pattern)
def downstream(v, m):  # a chip of another tray left unopenable after the C9 reset a tray's power drop broke: one fault
    ts = [x for x in v.pdn if x != m[2]]
    if not ts:
        return False
    v.pdn_chips.append(m[1])
    t, n = " and ".join(f"tray {x}" for x in ts), len(v.pdn_chips)
    trs = ", ".join(sorted(trays(v.pdn_chips)))
    v.lfam = "power/feed"
    v.fault(
        ("pdn",),
        f"POWER-DOWNSTREAM of the power drop of {t}: {n} chip{'s' * (n > 1)}{f' of tray {trs}' if trs else ''} stay on PCIe but do not open after the end (C9) reset (in-band IPMI not answering); not their boards",
        "; ".join(chip(b) for b in v.pdn_chips[:4]) + (f" and {n - 4} more" if n > 4 else ""),
        f"fix {t} first (see the Repair recommendation line), then reset and recheck these chips",
    )


@reads(r"^((?:[A-Z][A-Z0-9-]*\s)+?)(FAIL|HANG)\b:?\s*(.*)$")
def generic(v, m):
    t = m[3]
    bs, tr = list(dict.fromkeys(re.findall(BDF, t))), re.findall(r"tray ([1-4])", t)
    v.fault(
        ("gen", m.string[:120]),
        f"{m[1].strip()} {m[2].lower()}: {t[:160]}",
        "; ".join(chip(b) for b in bs[:4]) or (f"tray {tr[0]}" if tr else "the box"),
        "see the Repair recommendation line" + (f" (tray {tray_asic(bs[0])[0]})" if bs and tray_asic(bs[0])[0] else ""),
    )


def plain_reason(res):
    r = re.sub(r"^INCOMPLETE:\s*", "", re.sub(r"^RESULT:\s*", "", res or "").strip()).split("; chips counted")[0]
    m = re.match(r"stopped by (SIG\w+) during (\w+) \((\d+) s in\)(?:, last step: (.*))?", r)
    if m:
        return f"stopped from outside ({m.group(1)}) {int(m.group(3)) // 60} min in, during {m.group(4) or m.group(2)}"
    return r[:200]


def families(v):
    def chips(w):
        return re.findall(BDF, re.sub(r"root port \S+", "", w))

    def fam(k, where):  # a PCIe loss only on trays whose power dropped is that drop's
        ts = trays(chips(where))
        return "power/feed" if v.fam.get(k) == "PCIe" and ts and ts <= set(v.pdn) | v.tripped else v.fam.get(k, "other")

    out = []
    for f in [*FAMILIES, "other"]:
        fs = [(k, where) for k, (_, where, _) in v.faults if fam(k, where) == f]
        if not fs:
            continue
        bs = {b for k, w in fs for b in chips(w) + (v.pdn_chips if k == ("pdn",) else [])}
        ts = sorted(trays(bs) | {t for _, w in fs for t in re.findall(r"\btray ([1-4])\b", w)})
        nl = sum(" <-> " in w for _, w in fs) if f == "link" else 0
        cnt = ", ".join(f"{k} {u}{'s' * (k > 1)}" for k, u in [(nl, "link"), (len(fs) - nl, "fault")] if k)
        cnt += (f" on {len(bs)} chip{'s' * (len(bs) > 1)}" if bs else "") + (
            f", tray{'s' * (len(ts) > 1)} {', '.join(ts)}" if ts else ", the box"
        )
        what = FAMILIES[f][0] if f in FAMILIES else "a finding that names no part family: see its fault line"
        out.append(f"  Family {f}: {cnt}: {what}")
    return out


def verdict(rc, vtext="", findings="", result="", commit="?", mode="", rerun="the health check"):
    """The 'Final verdict:' line, its Family lines, then one line per fault and note."""
    v, fl = V(), findings.splitlines()
    v.rerun = rerun
    if any(IPMI_DOWN.search(l) for l in fl):
        v.pdn = sorted(thru(fl, POWER_LOST))
    v.tripped = thru(fl, HSC_TRIP)
    for l in fl:
        l = l.strip()
        if not l:
            continue
        g = AFTER.search(l)
        l, v.tag = (l[: g.start()], g.group(1)) if g else (l, "")
        v.lfam = family(l)
        next((1 for p, f in READERS if (m := p.match(l)) and f(v, m) is not False), None)
    v.tag = ""
    fold(v)
    vt = (vtext or "").strip()
    rerun += f" --mode={mode}" if mode and mode not in ("quick", "repro") else ""
    out = []
    if vt.startswith("NOT RUN") or rc in (6, 10):
        out.append(
            f"Final verdict: NOT RUN: {vt[8:].strip(' ()') or f'exit {rc}'}: no hardware result; do: {'wait for the other user, then ' if rc == 10 else ''}rerun {rerun}"
        )
    elif vt.startswith("INCOMPLETE") or rc in (9, 129, 130, 143) or "INCOMPLETE" in (result or "")[:40]:
        fw = sorted(v.fw.items())
        if fw:
            reason = "; ".join(fw_why(t, x, y, mj)[0] for t, (x, y, mj, _) in fw) + (
                "; tt-umd discovery fails on mixed firmware" if any(x != "unread" for _, (x, _, _, _) in fw) else ""
            )
            nxt = " and ".join(dict.fromkeys(fw_why(t, x, y, mj)[1] for t, (x, y, mj, _) in fw)) + ", then retest"
        else:
            reason, nxt = plain_reason(result) or vt, f"rerun {rerun}"
        out.append(f"Final verdict: INCOMPLETE (not a fault): {reason}; next: {nxt}")
        fwt = {t for t, _ in fw}
        seen = [
            "seen before it stopped, not judged"
            + (" (follows the tray firmware mismatch)" if k[0] == "ethboth" and k[1] in fwt else "")
            + f": {w} | where: {where}"
            for k, (w, where, _) in v.faults
            if k[0] != "fw"
        ]
        v.notes[:0], v.faults = ([f"stopped by: {plain_reason(result) or vt}"] if fw else []) + seen, []
    elif v.faults:
        kinds = list(dict.fromkeys(re.sub(r"[,:(].*", "", w).strip() for _, (w, _, _) in v.faults))
        out.append(f"Final verdict: FAIL: {len(v.faults)} fault{'s' * (len(v.faults) > 1)}: {'; '.join(kinds)}")
        out += families(v)
    elif vt.startswith("KNOWN-DROP") or rc == 12:
        out.append("Final verdict: KNOWN-DROP: a tray known to be dropped was not tested; the others are below")
    elif rc == 0 or vt == "PASS":
        out.append("Final verdict: PASS: no fault found during this run")
    elif v.notes:
        out.append(
            f"Final verdict: no fault found during this run; the {vt or f'exit {rc}'} came only from the notes below"
        )
    else:
        out.append(f"Final verdict: {vt or f'FAIL (exit {rc})'}: no finding line names a part; see the lines above")
    out += [f"  Fault {i}: {w} | where: {where} | do: {do}" for i, (_, (w, where, do)) in enumerate(v.faults[:CAP], 1)]
    out += [f"  ... {len(v.faults) - CAP} more faults in the FAIL lines above"] if len(v.faults) > CAP else []
    return out + [f"  Note {i}: {n}" for i, n in enumerate(v.notes[:CAP], 1)]


def fold(v):
    named, kept, times = {}, [], {}
    for k, (w, where, do) in v.faults:
        if k[0] == "drop" and k[2]:
            new = [b for b in k[2] if b not in named]
            if not new:
                i = named[k[2][0]]
                what = w.split(" chips lost on ", 1)[-1]
                if k[3] or "missing before its glx_reset" in w or "missing in all when it was judged" in w:
                    add = f"; the same chips again: {what}"
                else:
                    times[i] = times.get(i, 1) + 1
                    add = f"; the chips came back and dropped again: {what} (dropped {times[i]} times)"
                kept[i] = (kept[i][0], (kept[i][1][0] + add, *kept[i][1][1:]))
                continue
            named.update(dict.fromkeys(new, len(kept)))
            where = "; ".join(chip(b) for b in new) if len(new) < len(k[2]) else where
        kept.append((k, (w, where, do)))
    down = {frozenset((k[1][0], k[2][0])): w for k, (w, _, _) in kept if k[0] == "eth"}
    cab = {frozenset(k[1:3]) for k, _ in kept if k[0] == "cable"}
    br = [k[1] for k, _ in kept if k[0] == "brick"][:1]
    v.faults = []
    for k, (w, where, do) in kept:
        if k[0] == "eth" and frozenset((k[1][0], k[2][0])) in cab:
            continue
        if k[0] == "cable" and frozenset(k[1:3]) in down:
            w = w.replace(
                "missing or bad",
                "missing or bad: " + down[frozenset(k[1:3])].replace("Ethernet link down", "link down"),
                1,
            )
        if br and k[0] in ("aer", "chunk", "gen", "pcie"):
            do = f"comes with the power brick {br[0]} overheat: fix brick {br[0]} first and rerun; only if it repeats then, {do}"
        v.faults.append((k, (w, where, do)))
    v.faults.sort(key=lambda f: f[0][0] != "brick")


# ── repair recommendation: the part or setting to fix first, per failing tray ───────────────────────────────────────

HEAD = re.compile(r"^(FAIL|INCOMPLETE|PASS|KNOWN-DROP) tray ([1-4])\b(.*)$")
WHY = re.compile(r"^    (.*?) \| .*\| cause: (.*?) -> next: (.*)$")
UNKNOWN = "unknown: see the FAIL lines above"
LOSS = ("the PCIe path (slot, retimer) of {b} on tray {u}, then the tray {u} board", "the tray {u} board carrying {b}")
# (cause pattern, repair template); first match wins. {u} tray, {b} chip BDF(s), {p} part, {x} chip-side ETH speed.
# None: no repair (a warning or not a fault). Order: power, software/firmware, links, PCIe, chip.
RULES = [
    (r"not opened after the reset because tray ([1-4]) lost power", None),
    (r"^POWER-DOWNSTREAM: of the power drop of tray [1-4]", "DOWNSTREAM"),
    (r"none in this run", None),
    (
        r"whole-tray power loss|lost power and its re-power failed|bus-bar signature|^tray 12 V input lost or under-voltage|^POWER: 12 V feed of tray",
        "POWER:{u}",
    ),
    (r"^POWER: tray [1-4] 12 V feed", "POWERFEED"),
    (r"^CONFIG: retimer/ETH speed", "CONFIG"),
    (r"^test infrastructure", "no hardware repair (software: test infrastructure on tray {u})"),
    (r"^firmware version differs", "no hardware repair (firmware: reflash tray {u})"),
    (
        r"warning only|not a fault|not checked|leave the box|read no link map|ETH ports not up \(warning\)|down before the run|not a clean pass",
        None,
    ),
    (r"which fails all its C5 pairs: that chip", "the tray {t} board carrying {c} (all its links fail)"),
    (
        r"^in-box link down or bad.*retimer speed not readable",
        "check the retimer speed setting (expected {x}) of link {p}, then the tray-to-tray connection, then the tray boards",
    ),
    (r"^in-box link down or bad", "the tray-to-tray connection of link {p}, then the tray boards"),
    (r"^link pair hung|^link down or bad", "the cable or port of link {p}"),
    (r"^weak PCIe link|^correctable AER rate outlier|^PCIe path", LOSS[0]),
    (r"^CCL group fault: one bad result from the likely source", "NUMERIC:{s}"),
    (r"^chip compute fault|^chip DRAM fault|^chip read fault", "NUMERIC:{b}"),
    (
        r"^these chips stopped while the others kept going|^chip fell off PCIe|^chip lost after its warm reset|^chip did not take the warm reset|^hang on the named chip\(s\)|^chip slow to finish its warm reset|^chip stopped answering the kernel driver under load|^chip hung for MMIO",
        LOSS[1],
    ),
]
# the first action per fault family; the fleet's (fixed, tried) counts for it come in with --repair-history
FIRST = {
    "power/feed": "replace the bus bar set",
    "math": "stored Vmin 760 mV",
    "math tray": "replace the tray board",
    "DRAM": "flash the current board firmware",
    "link": "reseat the cable or tray-to-tray connection",
    "PCIe": "PCIe Gen4 cap",
    "PCIe loss": "flash the retimer firmware",
}
HUNG = re.compile(r"^these chips stopped|^hang on|^chip stopped answering|^chip hung")
THERMAL = re.compile(
    r"^THERMAL FAIL (host|tray [1-4]|[^:]*): (\d+) SEL thermal trip event\(s\) during the run(?: \((.*)\))?"
)
SENSORS = [  # BMC temperature sensor -> (its part, the first step); first match wins
    (r"PCIe_RT", "PCIe retimers on the PDB", "check fans and airflow, then the PDB"),
    (
        r"PWRBRK(\d+)",
        "power brick {0}, one of the PDB's 54-to-12 V DC-DC modules",
        "check its heatsink, thermal contact and airflow, then replace power brick {0} on the PDB; not a tray reseat",
    ),
    (r"PSU(\d)", "PSU {0}", "check fans and airflow, then PSU {0}"),
    (r"INLET", "the box's air inlet", "check the room's cooling and inlet air, then the fans"),
    (r"OUTLET", "the box's air outlet", "check fans and airflow"),
    (r"tray ([1-4])", "tray {0}", "check fans and airflow, then the tray {0} board"),
]


def thermal_parse(line):
    m = THERMAL.match(line)
    if not m:
        return None
    f = [x.strip() for x in (m.group(3) or "").split("|")]
    sensor = f[3].replace("Temperature ", "") if len(f) > 3 else "a temperature sensor"
    rd = re.search(r"Reading (\d+(?:\.\d+)?) >", m.group(3) or "")
    pb = re.search(r"PWRBRK(\d+)", sensor)
    part, step = sensor, f"check fans and airflow, then the part {sensor} reads"
    for pat, p, s in SENSORS:
        g = re.search(pat, sensor)
        if g:
            part, step = p.format(*g.groups()), s.format(*g.groups())
            break
    return m.group(1), m.group(2), sensor, rd and rd.group(1), pb and pb.group(1), part, step


def bdfs(text, u=None):
    tied = re.findall(rf"({BDF}) \(tray ([1-4])\)", text)
    got = ([b for b, t in tied if u is None or t == u] or [b for b, _ in tied]) if tied else re.findall(BDF, text)
    return list(dict.fromkeys(got))


def loss_bdfs(u, line):
    w = WHY.match(line)
    if not w or HSC_FAIL.search(line):
        return []
    kind = []
    item(str(u), w.group(1), w.group(2), w.group(3), "?", kind)
    return bdfs(w.group(1), str(u)) if kind and kind[0] in LOSS else []


def item(u, part, cause, nxt, commit, kind=None):
    for pat, tpl in RULES:
        if not re.search(pat, cause):
            continue
        if kind is not None:
            kind.append(tpl)
        if tpl is None:
            return ""
        bs = bdfs(part, u) or bdfs(nxt, u) or bdfs(cause, u)
        b = " ".join(bs) if bs else f"a chip of tray {u}"
        if tpl == "CONFIG":
            m = re.search(r"expected (x\d+G)", cause)
            return f"set the retimers of link {part} to {m.group(1) if m else 'x200G'} and rerun, before any cabling or board step"
        if tpl == "POWERFEED":
            m = re.search(r"12 V feed (.+?) / bus bar: HSC (0x[0-9a-f]+) ([A-Z_+]+)", cause)
            src = (
                f"feed {m.group(1)} of tray {u} (HSC {m.group(2)} {m.group(3)} newly set during this run)"
                if m
                else (f"feed of tray {u}")
            )
            return f"bus bar / tray 12 V {src}, then tray {u} board if it persists"
        if tpl.startswith("NUMERIC:"):
            m = re.search(rf"likely source ({BDF})", cause) if tpl == "NUMERIC:{s}" else None
            b = m.group(1) if m else b
            cc = re.search(r"compute core \((\d+),(\d+)\)", cause)
            if cc:
                wc = re.search(r"\(translated worker core (\d+),(\d+)\)", cause)
                return (
                    f"the tray {u} board (UBB-{u}): ASIC {b} compute core logical ({cc.group(1)},{cc.group(2)})"
                    + (f", translated worker core ({wc.group(1)},{wc.group(2)})" if wc else "")
                    + " writes wrong values under matmul load"
                )
            return f"the tray {u} board carrying {b} (numeric fault), only if the same check with commit {commit} on clean hosts shows no similar counts (similar counts there: test or CCL issue, not the box)"
        m = re.search(rf"of ({BDF}) \(tray ([1-4])\)", cause)
        x = re.search(r"chip side (x\d+G)", cause)
        return tpl.format(
            u=u, b=b, p=part, c=m.group(1) if m else b, t=m.group(2) if m else u, x=x.group(1) if x else "x200G"
        )
    return None


def tpl_family(tpl, cause):
    if tpl == LOSS[0]:
        return "PCIe"
    if tpl == LOSS[1]:
        return None if HUNG.match(cause) else "PCIe loss"
    if tpl and tpl.startswith("NUMERIC:"):
        return "DRAM" if cause.startswith("chip DRAM fault") else "math"
    if tpl and re.search(r"\blink\b", tpl) and not tpl.startswith("check the retimer"):
        return "link"
    return None


def lead(r, fam, history):
    _, first, f = min(((r.find(i), i, g) for i, g in fam.items() if i in r), default=(0, "", None))
    if f == "math":  # several bad chips on the first item's tray: the board
        t = re.findall(r"tray ([1-4])", first)[:1]
        bs = {
            b
            for i, g in fam.items()
            if g == "math" and re.findall(r"tray ([1-4])", i)[:1] == t
            for b in re.findall(BDF, i)
        }
        f = "math tray" if t and len(bs) > 1 else "math"
    k, n = history.get(f, (0, 0))
    return f"{FIRST[f]} ({f'fixed {k} of {n} in fleet records' if n else 'no fleet records'}); {r}" if f else r


def recommend(
    rc, verdict, trays_text="", findings_text="", mode="", commit="?", fix_text="", fam=None, rerun="the health check"
):
    """The repair line's text; fam (a dict) gets each item's fault family."""
    fam = {} if fam is None else fam
    rerun += f" --mode={mode}" if mode and mode != "quick" else ""
    items, unknown, heads, named = [], [], {}, set()
    power, loss_of, fold_, off, lost, feed = {}, {}, set(), set(), {}, {}
    u = None
    for l in trays_text.splitlines():
        h = HEAD.match(l)
        if h:
            u = h.group(2)
            heads[u] = (h.group(1), h.group(3).strip())
            continue
        if not (u and heads[u][0] == "FAIL"):
            continue
        m = re.match(rf"\s*\.\.\. \d+ more for tray {u}\b.*?(?:; lost chips: (.*))?$", l)
        if m:
            lost.setdefault(u, set()).update(re.findall(BDF, m.group(1) or ""))
            continue
        w = WHY.match(l)
        if not w:
            continue
        kind = []
        r = item(u, w.group(1), w.group(2), w.group(3), commit, kind)
        hf = HSC_FAIL.search(l)
        if hf and hf.group(1) == u:  # an HSC latch: a feed line; it folds only with OFF and all 8 chips lost
            named.add(u)
            power.setdefault(u, set()).add(hf.group(2))
            if re.search(r"\bOFF\b", hf.group(3)):
                off.add(u)
            if kind and kind[0] == "POWERFEED" and r and r not in feed.setdefault(u, []):
                feed[u].append(r)
            continue
        if r == f"POWER:{u}":
            power.setdefault(u, set())
            fold_.add(u)
            named.add(u)
        elif r == "DOWNSTREAM":  # the dropped tray's feed is the repair
            for t in re.findall(r"tray ([1-4])", w.group(2).split(":", 2)[1]):
                power.setdefault(t, set())
            named.add(u)
        elif kind and kind[0] == "POWERFEED":
            if r not in feed.setdefault(u, []):
                feed[u].append(r)
            named.add(u)
        elif r is None:
            unknown.append(u)
        elif r:
            named.add(u)
            if kind[0] in LOSS:
                lost.setdefault(u, set()).update(bdfs(w.group(1), u))
            if r not in items:
                items.append(r)
                fam[r] = tpl_family(kind[0], w.group(2))
                if kind[0] in LOSS:
                    loss_of[r] = u
    fx = fix_text.splitlines()  # a host link check's 'FIX tray N: **...**', then '  why: ...'
    for i, l in enumerate(fx):
        m = re.match(r"FIX tray ([1-4]): \*\*(.*?)\*\*", l)
        if not m:
            continue
        t = m.group(1)
        named.add(t)
        if "bus bar" in m.group(2):
            power.setdefault(t, set())
            continue
        bs = bdfs(fx[i + 1] if i + 1 < len(fx) and fx[i + 1].startswith("  why:") else "", t)
        r = f"the tray {t} board carrying {' '.join(bs) if bs else 'its missing chips'} or its PCIe path, after the remote recovery steps ({m.group(2)})"
        if r not in items:
            items.append(r)
            fam[r] = "PCIe loss"
    for l in findings_text.splitlines():
        if l.startswith("PSU FINDING") and "the PSU named in the PSU FINDING line" not in items:
            items.append("the PSU named in the PSU FINDING line")
            fam[items[-1]] = None
        t = thermal_parse(l)
        r = t and f"cooling: {t[2]} ({t[5]}) tripped {t[1]} times{f', peak {t[3]} C' if t[3] else ''}: {t[6]}"
        if r and r not in items:
            items.append(r)
            fam[r] = None  # no fleet records for thermal trips: the item names its own step, no prefix
    unknown += [t for t, (k, _) in sorted(heads.items()) if k == "FAIL" and t not in named]
    fold_ |= {t for t in off if len(lost.get(t, ())) >= 8}  # a whole-tray loss folds its PCIe/board-loss lines
    hsc = lambda t: f" (HSC {', '.join(sorted(power[t]))})" if power.get(t) else ""  # noqa: E731
    pw = [
        f
        for t in sorted(set(power) | set(feed))
        for f in feed.get(t)
        or [f"bus bar / tray 12 V feed of tray {t}{hsc(t)}, then the tray {t} board if it still drops"]
    ]
    fam.update(dict.fromkeys(pw, "power/feed"))
    items = pw + [r for r in items if loss_of.get(r) not in fold_]
    unknown = [t for t in unknown if t not in fold_]
    if unknown:
        items.append(f"{UNKNOWN} (tray {', '.join(sorted(set(unknown)))})")
    kd = [t for t, (k, _) in heads.items() if k == "KNOWN-DROP"]
    if items:
        return "; ".join(items + [f"tray {t} (known dropped, not tested)" for t in kd])
    v = verdict or ""
    if (
        v.startswith(("NOT RUN", "INCOMPLETE"))
        or rc in (9, 10, 129, 130, 143)
        or any(k == "INCOMPLETE" for k, _ in heads.values())
    ):
        inc = [f"tray {t}" for t, (k, _) in sorted(heads.items()) if k == "INCOMPLETE"]
        return f"none determined ({v or f'exit {rc}'}); rerun {rerun}" + (f" for {', '.join(inc)}" if inc else "")
    if kd:
        return f"tray {kd[0]} (known dropped, not tested); none for the other trays"
    if rc == 0 or v == "PASS":
        return "none (all 4 trays pass)"
    if not heads:
        return f"none determined ({v or f'exit {rc}'}); fix the cause above and rerun {rerun}"
    return UNKNOWN


# ── the run's report ────────────────────────────────────────────────────────────────────────────────────────────────


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--repair-history", type=Path, help="fleet repair counts, '<family><TAB><fixed><TAB><tried>' lines")


def history(path) -> dict:
    rows = [r.split("\t") for r in (Path(path).read_text().splitlines() if path else []) if not r.startswith("#")]
    return {p[0]: (int(p[1]), int(p[2])) for p in rows if len(p) == 3}


def lines(run_dir, args) -> list[str]:
    """The tray lines with their causes (when the run has a findings.txt), the final verdict and the repair line.
    The run's exit code and verdict come from args (rc, verdict), else from its report (args.output)."""
    d, a = Path(run_dir), vars(args)
    rd = lambda p: Path(p).read_text(errors="replace") if p and Path(p).is_file() else ""  # noqa: E731
    rc, vt, mode, commit, rerun = (
        a.get("rc"),
        a.get("verdict") or "",
        a.get("mode") or "",
        a.get("commit") or "?",
        a.get("rerun") or RERUN,
    )
    if rc is None:
        st = json.loads(rd(a.get("output") or d / "diag_report.json") or "{}").get("overall_status", "")
        rc, vt = int(st == "FAIL"), vt or st
    raw = rd(d / "findings.txt").splitlines()
    stages = {}
    for r in rd(d / "finding_stage.tsv").splitlines():
        stages.setdefault(r.partition("\t")[2], r.partition("\t")[0])
    if a.get("only") == "result":
        ls, st, _ = untag(raw, stages)
        return [result(ls, st, sorted(thru(ls, re.compile(r"^(?:POWER|CHIP) DROP tray ([1-4])"))), a.get("known_tray"))]
    if a.get("only") == "final":  # the tray lines this run already printed, as it saved them
        trays = rd(d / "trays.txt").splitlines()
    else:
        f = d / "findings.txt"
        trays = tray_block(raw, stages, str(f), a.get("known_tray"), a.get("stopped") or "") if f.is_file() else []
        if a.get("only") == "trays":
            return trays
    fam, out = {}, [] if a.get("only") else trays
    out += verdict(rc, vt, "\n".join(raw), a.get("result") or "", commit, mode, rerun)
    r = recommend(rc, vt, "\n".join(trays), "\n".join(raw), mode, commit, rd(a.get("fix")), fam, rerun)
    return out + [f"Repair recommendation: {lead(r, fam, history(a.get('repair_history')))}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--rc", type=int)
    for f in ("--verdict", "--result", "--mode", "--commit", "--fix", "--known-tray", "--stopped", "--rerun"):
        ap.add_argument(f)
    ap.add_argument("--only", choices=("trays", "result", "final"))
    add_args(ap)
    print("\n".join(lines((a := ap.parse_args(argv)).run_dir, a)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
