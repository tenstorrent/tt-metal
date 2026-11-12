#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tray_why.py: the exact failure and likely root cause under each FAIL or WARN tray line of run.sh's verdict (chip, link,
etc.). Reads only the FINDING and WARN lines run.sh already wrote; no device, no new check.
  tray_why.py <findings.txt> <tray 1-4> [--stages <finding_stage.tsv>] [--max N] [--drop-trays "3 4"]
  tray_why.py <findings.txt> --result [--also <numeric.txt>] [--stages ..] [--drop-trays ..] [--known-tray N]
--result: one phrase naming every failed check fl_report will print as a FAIL tray line, for the RESULT line.
--drop-trays: the trays that lost chips (POWER/CHIP DROP); a chunk whose load died then lists every tray's chips in its CHUNK
FAIL line, but it is listed only under those trays, as fl_report counts it.
One indented line per finding that fl_report counts for that tray (same tray match: a finding's ", BMC after: ..." tail
and a QSFP line's " (to <HOST> ...)" peer are not this tray's), FAILs first, at most N (default 10) then one '... more':
  <part> | <stage> | <evidence> | cause: <likely root cause> -> next: <next step>
part: a chip as BDF (tray N), a link as BDF (tray N) port a,b <-> BDF (tray M) port c,d, or the whole tray. stage: from the
finding's own words or a busbar tag in finding_stage.tsv (the bus-bar stage's chunk lines name no stage), else from
finding_stage.tsv (fl_find writes "<phase>\t<line>"), else C4 for run.sh's NUMERIC lines.
Tray numbers only, never UBB. Whole-tray power loss: bus bar first, then the tray."""
import json, os, re, sys

BDF = r"(?:0000:)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])"
CHIP = re.compile(BDF + r" \(tray ([1-4])\)")
ONE = re.compile(BDF + r" \(tray ([1-4])\)(?: chip \d+| tt-smi (\d+))?")   # a chip as findings name it: 02:00.0 (tray 1) tt-smi 1
_ONE = r"(?:0000:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7] \(tray [1-4]\)(?: (?:chip|tt-smi) \d+)?"
LIST = re.compile(r"^((?:[A-Z0-9-]+:? )+)(" + _ONE + r"(?:, " + _ONE + r")*)(?=[ :])")   # a finding's leading chip list
LINK = re.compile(BDF + r" \(tray ([1-4])\) e([0-9,]+) <-> " + BDF + r" \(tray ([1-4])\) e([0-9,]+)")
ONECHIP = re.compile(r"LINK FAIL one chip " + BDF + r" \(tray ([1-4])\)[^:]*: in all (\d+) failing (.*?) pairs \(([^)]*)\): (.*)$")
ONELINK = re.compile(BDF + r" e([0-9,]+) <-> " + BDF + r" e([0-9,]+) ([^;(]*?) \((p\d+)\)")
STAGES = [l.rstrip("\n").split("\t") for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stages.tsv"))
          if l.strip() and not l.startswith("#")]   # the one stage table: number, code, name, what
PHASE = {c: f"stage {n} {nm}" for n, c, nm, *_ in STAGES}
PCIE_WHY = os.environ.get("FLEET_PCIE_WHY", "1") != "0"   # the PCIe wording (all-Fs, power-state I/O, AER FAIL); 0 = off
PCIE_NEXT = ("swap this tray's host PCIe cable with a neighbour tray's; if the errors follow the tray, replace the board")
PHASE.update({"end": "end checks", "linka": "link-A repro", "abortfab": "abort-fabric repro"})
POWER_NEXT = "1. Replace the bus bar -> 2. if the tray still drops, replace tray {u}"
# lib/hsc_check.sh's HSC FAIL lines (a leg's 'latched during <stage>' or, OFF/IOUT_OC only, 'set during <stage>',
# the end read's 'set during the run'); only these name bits newly set during the run. Pre-existing latches are WARN lines;
# unread bits are 'not graded'. HSC_ADD: the OFF/IOUT_OC bits hsc_judge appends to an under-voltage leg line.
HSC_NEW = re.compile(r"^HSC FAIL tray ([1-4]) \([^)]*\) HSC (0x[0-9a-fA-F]{2}): ([A-Z_]+(?:\+[A-Z_]+)*) (?:latched|set) during ")
HSC_ADD = re.compile(r"; ([A-Z_]+(?:\+[A-Z_]+)*) set during ")
HSC_POWER_BITS = ("VIN_UV_WARN", "VIN_UV_FAULT", "OFF", "IOUT_OC")   # CML is a comm fault, never power
HSC_FEED = {"0x40": "A", "0x42": "B"}   # each tray's two HSCs (lib/hsc_check.sh HSC_ADDRS)


DIED = re.compile(r"CHUNK FAIL chunk \d+/\d+ .*no chunk-end readback, the load died or hung")


def tray_lines(lines, u, drop_trays=None, stages=None):
    """The lines fl_report counts for tray u, FAILs first (its nf/nw greps, same sed)."""
    stages = stages or {}
    dstages = {stages.get(l) for l in lines if re.match(r"(POWER|CHIP) DROP tray ", l)}
    hit = re.compile(rf"tray {u}([^0-9]|$)")
    fails, warns = [], []
    for l in lines:
        if not l.strip():
            continue
        if drop_trays and str(u) not in drop_trays and DIED.search(l) \
                and (stages.get(l) is None or stages[l] in dstages):
            continue   # only a chunk of a stage that had a drop; # the load died because another tray lost chips; this tray lost nothing
        if " WARN " in l or l.startswith(("ETH INCOMPLETE ", "HANG INCOMPLETE ")):   # external ports not checked, never a FAIL; a box-wide hang names no tray
            if hit.search(re.sub(r" by tray [1-4](?: and tray [1-4])* power loss", "", re.sub(r" \(to OM-[^)]*\)", "", l))):
                warns.append(l)
        elif hit.search(re.sub(r" \(to OM-[^)]*\)", "", re.sub(r", BMC after: .*", "", l))):
            fails.append(l)
    return fails, warns


def stage_of(l, stages):
    s = l.lower()
    # a chunk the fault came in; never 'before chunk 1' or a hang pinpoint's 'in chunk 1 or before'
    m = re.search(r"(?<!before )chunk (\d+)(?:/(\d+))?(?![\d/]| or before)", l)
    if stages.get(l) == "sweep":   # finding_stage.tsv wins over a 'bus-bar stage' phrase in the text
        return PHASE["sweep"]
    if l.startswith("NUMERIC ") or stages.get(l) == "numeric" or "stage 5 numeric verify" in s or "/busbar/rw" in s:
        # a numeric mismatch is stage 5, never the bus-bar stage
        return PHASE["numeric"] + (" heavy chunk" if "heavy chunk" in s or "/busbar/heavy/" in s else " g3" if "g3" in s or "/busbar/rw" in s
                                   else f" chunk {m.group(1)}" if m else "")
    if "bus-bar stage heavy chunk" in s or "/busbar/heavy/" in s:   # the bus-bar stage's heavy chunk, in quick and full
        return PHASE["busbar"] + " heavy chunk"
    if "bus-bar stage" in s or "/busbar/" in s or stages.get(l) == "busbar" or l.startswith(("CHUNK ", "QUIET ")):   # its CCL-mix chunk lines name no stage
        return PHASE["busbar"] + (f" chunk {m.group(1)}" if m else "")
    if "/c5/" in s:
        return PHASE["C5"]
    if "/sweep/" in s or l.startswith("PINPOINT"):
        return PHASE["sweep"]
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
        return PHASE.get(stages[l], stages[l])
    if l.startswith("NUMERIC "):
        return "C4"   # run.sh's numeric_report.py lines over the C4 tries (fl_report appends them)
    return "stage not recorded"


QSFP = {1: (5, (2, 3)), 2: (1, (2, 3)), 3: (1, (0, 1)), 4: (2, (0, 1)), 5: (3, (0, 1)), 6: (4, (0, 1))}   # fleet_py.QSFP


def tend(b, t, chs):
    """A link end as tray / QSFP-DD port / ASIC / BDF; ETH channels only in
    brackets after it. An end on no faceplate port (an in-box or on-board link) names tray, ASIC and BDF only."""
    cs = [int(c) for c in str(chs).replace(" ", "").split(",") if c]
    a = int(b[1], 16)
    q = sorted({k for c in cs for k, (qa, qc) in QSFP.items() if qa == a and c in qc})
    return (f"tray {t} QSFP-DD port {' and '.join(map(str, q))} / " if q else f"tray {t} ") + f"ASIC {a} / {b} [ETH ch {','.join(map(str, cs))}]"


def has_port(part):  # a link part (tend wording) with a faceplate QSFP-DD port at both ends: a cable to reseat
    return part.count("QSFP-DD port") >= 2


def part_of(l, u):
    """(kind, text): link, chips, tray or unnamed."""
    k = LINK.search(l)
    if k and u in (k.group(2), k.group(5)):
        return "link", f"{tend(k.group(1), k.group(2), k.group(3))} <-> {tend(k.group(4), k.group(5), k.group(6))}"
    m = re.match(r"ETH (?:FAIL|WARN) " + BDF + r" \(tray ([1-4])\) \w+: local ETH ports \[([0-9, ]+)\] (?:not up|down at pre) \(to ([^;]*?)\)(?:;.*)?$", l)
    if m:   # C3's local link check names the far end of each port
        far = {}
        for b, t, pt in re.findall(BDF + r" \(tray ([1-4])\) port (\d+)", m.group(4)):
            far.setdefault(f"{b} (tray {t})", []).append(pt)
        mine = [c for c in m.group(3).replace(" ", "").split(",")]
        return "link", (f"{tend(m.group(1), m.group(2), ','.join(mine))} <-> "
                        + "; ".join(tend(k[:7], k[-2], ",".join(v)) for k, v in far.items()))
    if re.match(rf"(POWER (DROP|FAIL)|SEL (FAIL|WARN)) tray {u}\b", l) or "tray drop" in l:
        return "tray", f"whole tray {u}"
    m = re.match(r"L1 FAIL " + BDF + rf" tray {u} core (\S+)", l)
    if m:   # N1's L1 march: the chip and its core
        return "chips", f"{m.group(1)} (tray {u}) L1 core {m.group(2)}"
    if re.match(rf"CHIP DROP tray {u}\b", l):
        bs = re.findall(r"[0-9a-f]{2}:00\.[0-7]", l.split(", BMC")[0])
        return "chips", f"{' '.join(bs) or 'chips'} (tray {u})"
    cs = [c for c, t in CHIP.findall(re.sub(r" \(to OM-[^)]*\)", "", l)) if t == u]
    if cs and ("UNATTRIBUTED" in l or "candidates:" in l):
        return "unnamed", f"no single chip named ({len(cs)} candidate chip{'s' if len(cs) > 1 else ''} on tray {u})"
    if l.startswith("CHUNK FAIL all "):
        return "unnamed", f"all chips (tray {u} among them)"
    m = re.search(rf"{BDF} \(tray {u}\)[^\[]*?ETH ports? (\[[0-9, ]+\])", l)
    if m:
        return "chips", f"{m.group(1)} (tray {u}) ETH ports {m.group(2)[1:-1].replace(' ', '')}"
    if cs:
        k = {b: n for b, t, n in ONE.findall(l) if t == u and n}
        if k:   # tt-smi indices given (an H3SHAPE HANG's chip list): each chip as BDF (tray N, tt-smi k)
            return "chips", ", ".join(f"{b} (tray {u}" + (f", tt-smi {k[b]})" if b in k else ")") for b in dict.fromkeys(cs))
        return "chips", " ".join(dict.fromkeys(cs)) + f" (tray {u})"
    return "tray", f"tray {u}"


STAGE_DIR = {PHASE["busbar"]: "busbar", PHASE["busbar"] + " heavy": "busbar/heavy", PHASE["numeric"]: "busbar",
             PHASE["numeric"] + " heavy": "busbar/heavy", PHASE["numeric"] + " g3": "busbar/rw", PHASE["sweep"]: "sweep", PHASE["C5"]: "c5"}


def path_of(l, findings, stage):
    """The capture the finding names, else the stage's own dir (C4: run.sh's --hold-on-hang snapshots)."""
    m = re.search(r"(?:snapshot|triage saved|saved|probe:?|triage) (/[^\s);,]+)", l) or re.search(r"\((/[^\s);,]+)", l)
    if m:
        return m.group(1)
    fl = os.path.dirname(os.path.abspath(findings))
    if stage == "C4":
        return os.path.join(os.path.dirname(fl), "heavy", "hold_try*")
    return os.path.join(fl, STAGE_DIR.get(stage.split(" chunk")[0].split(" at ")[0], ""))


def evidence(l, u=None):
    e = re.sub(r" ?\(?(?<![\w.])/[A-Za-z_~][^\s);,]*\)?", "", l.split(", BMC after:")[0])   # paths go in the next step
    e = re.sub(r"; candidates: .*|, candidates: .*", "", e)
    e = re.sub(r"(\d+ chips lost): [0-9a-f:. ]+", r"\1", e)
    e = LINK.sub("", e, count=1)
    e = re.sub(r"(local ETH ports \[[0-9, ]+\] (?:not up|down at pre)) \(to [^;]*\)", r"\1", e)   # the far end is in the part column
    m = LIST.match(e)   # the part: its first chip goes, and of a longer list tray u's chips; another tray's peer stays
    if m:
        cs = [x.group(0) for x in ONE.finditer(m.group(2))]
        keep = [x for x in cs if ONE.match(x).group(2) != u] if u and len(cs) > 1 else cs[1:]
        e = m.group(1) + ", ".join(keep) + e[m.end():]
    e = re.sub(r"\((?:snapshot|triage saved|probe:?) (?=\(its error)", "", e)   # path gone, its '(its error' stays
    e = re.sub(r"[(;,]?\s*(?:\b(?:snapshot|triage saved|probe:?)|(?<![\w-])triage)\s*(?=[);,]|$)\)?", "", e)
    e = re.sub(r"\s+", " ", e).replace(" :", ":").replace(" ,", "").strip(" ,;:")
    if "vr_pg 00" in l:
        e += " (tray VR power-good low)"
    return e if len(e) <= 140 else e[:137] + "..."


def ccl_source(l, lines=()):
    """The likely source chips of a CCL group finding (chunk_sched.group_lines): [(bdf, tray)] its 'likely
    source' names, else a member that another FAIL line fails outside the CCL ([heavy], C4, [rw]); [] when none."""
    m = re.search(r"[;,] likely sources? (.*)$", l)
    if m:
        return list(dict.fromkeys(CHIP.findall(m.group(1))))
    mem = set(CHIP.findall(l))
    src = []
    for o in lines:
        f = re.match(r"NUMERIC FAIL " + BDF + r" \(tray ([1-4])\)(?: chip \S+)?(?: \[([^\]]*)\])?:", o)
        if f and "CCL group fault" not in o and (f.group(1), f.group(2)) in mem \
                and not (f.group(3) and set(f.group(3).split(",")) <= {"h3", "h3-ref"}):
            src.append((f.group(1), f.group(2)))
    return list(dict.fromkeys(src))


def cause(l, kind, u, findings, stage="", lines=()):
    """(cause, next) in plain words."""
    s = l.lower()
    warn = " WARN " in l
    if l.startswith("ETH INCOMPLETE ") and "start reset skipped; ETH not-active-at-open not judged" in l:
        return ("ETH core not active again at mesh open with no start reset before it: start reset skipped; ETH not-active-at-open not judged, "
                "not checked, not a fault",
                "none on this box; judged only by a run whose C2 start reset (tt-smi -glx_reset) ran")   # no 'rerun'
    if l.startswith("HANG PINPOINT") and "whole box" in l:   # fl_loads' chunk_sched.py stalled line
        return ("every chip stalled together (one mesh: launch counts move in lockstep), so no chip or tray stopped first: "
                "not one tray's board; hardware or software not determined",
                "repeat the stage on this host with tt-triage/exalens there to read the hung state (run.sh --hold keeps it)")
    if l.startswith("HANG PINPOINT"):
        return ("these chips stopped while the others kept going: chip/board or its links",
                "read the hang capture of these chips; rerun the stage on this host")
    if kind == "tray" and l.startswith("POWER FAIL") and "reset chain broken by its power loss" in l:
        return (f"tray {u} lost power and its re-power failed, so the reset chain broke there: the other trays' chips stay on "
                "PCIe but do not open (ENODEV); their boards are not at fault", POWER_NEXT.format(u=u))
    if warn and "not opened: reset chain broken by tray" in l:   # on PCIe, ENODEV, another tray without power
        t = re.search(r"broken by (tray [1-4](?: and tray [1-4])*) power loss", l).group(1)
        return (f"not opened after the reset because {t} lost power and broke the reset chain; this chip stays on PCIe "
                "(ENODEV): not this tray's board", f"fix {t} first (see its FAIL line), then reset and recheck this chip")
    hf = HSC_NEW.match(l)   # a power latch newly set during this run is a POWER cause naming
    # its feed and HSC; it comes before the whole-tray rule, which stays the fallback for lines it does not parse
    if hf and not warn:
        bs = hf.group(3).split("+") + [b for m in HSC_ADD.findall(l[hf.end():]) for b in m.split("+")]
        pw = [b for b in dict.fromkeys(bs) if b in HSC_POWER_BITS]
        if pw:
            a = hf.group(2).lower()
            f = HSC_FEED.get(a, f"at HSC {a}")
            uv = "an under-voltage latch" if any(b.startswith("VIN_UV") for b in pw) else "a power latch"
            return (f"POWER: tray {hf.group(1)} 12 V feed {f} / bus bar: HSC {a} {'+'.join(pw)} newly set during this run "
                    f"(clear at its start; {uv} from this run, not a pre-existing one)",
                    f"1. Check the bus bar and 12 V feed {f} of tray {hf.group(1)} (HSC {a}) -> 2. replace the bus bar if the "
                    f"latch sets again -> 3. replace tray {hf.group(1)} only if it persists")
    if (kind == "tray" and l.startswith("HSC FAIL") and re.search(r"\b(VIN_UV_WARN|VIN_UV_FAULT|OFF)\b", l)
            and re.search(r"latched during|set during", l)):   # a newly set HSC latch is the 12 V-lost evidence
        return ("tray 12 V input lost or under-voltage: HSC OFF/under-voltage latch newly set during the run (12 V-lost evidence)",
                POWER_NEXT.format(u=u))
    if kind == "tray" and l.startswith("SEL WARN ") and "frb3" in s and "during the run" in s:   # SEL alone
        return ("SEL FRB3 processor init failure during the run with no 12 V-lost evidence (no chip lost, no newly set "
                "HSC OFF/under-voltage latch, no CPLD 12 V loss): a warning only, not a power fault",
                f"none from this line; a lost chip or a newly set HSC latch on tray {u} would have its own line")
    if kind == "tray" and (l.startswith(("POWER DROP", "POWER FAIL")) or "frb3" in s or "tray drop" in s):
        if warn:
            return "earlier whole-tray power loss in the BMC log (bus-bar signature), none in this run", POWER_NEXT.format(u=u) + " if it drops in a run"
        return "whole-tray power loss", POWER_NEXT.format(u=u)
    if l.startswith("FAIL MCE GUARD") and "kmd burst" in l:   # the guard's monitor under our load
        return ("chip stopped answering the kernel driver under load (a burst of kmd 'Failed to set initial power state'; "
                "the MCE guard cut it): chip/board or its power, not determined",
                f"read mce_guard/ and the capture; rerun run.sh on this host; same chip again: replace the board of tray {u}")
    pw = PCIE_WHY and re.search(r"Setting power state failed on device \d+: Input/output error|Failed to set initial power state", l)
    if pw and not l.startswith("FAIL MCE GUARD"):   # a power-state write the chip never took is PCIe
        return ("PCIe fault: the chip did not take a power-state write over PCIe (UMD/kmd I/O error), not a compute fault: "
                "its PCIe path (board, retimer or cable, or the host slot)", PCIE_NEXT)
    if PCIE_WHY and re.match(r"AER FAIL .*: correctable \+\d+ ", l):   # one root port's corrected AER, rule 1 or a stall
        return ("the chip's PCIe path: board, retimer or cable, or the host slot"
                + (" (likely the stall's cause: the only port with errors)" if "likely " in l else ""), PCIE_NEXT)
    if l.startswith("FAIL MCE GUARD") and "hung for MMIO" in l:   # the guard cut a chip that hung under load
        return ("chip hung for MMIO under load (config reads ffffffff or an ARC timeout; the MCE guard cut it): CHIP/BOARD",
                f"read mce_guard/ and the capture; rerun run.sh on this host; same chip again: replace the board of tray {u}")
    if l.startswith("FAIL MCE GUARD") and "AER storm" in l:
        return ("PCIe path: an AER storm on its root port under load (the MCE guard cut the chip): slot, retimer or board",
                f"read the AER/lspci capture; reseat tray {u}; if it repeats, replace the board of tray {u}")
    hang = not warn and re.search(r"\b(hang|hung)\b(?![- ]in)", s)
    o = ONECHIP.search(l)
    if o:   # every failing C5 pair has this chip at one end, the far-end chips pass their other links
        e = re.search(r"active ethernet core (\d+)-25", l)
        c = (f"the chip itself, not the cables: its own ETH core {e.group(1)}-25 (channel {int(e.group(1)) - 20}) does not come up, so "
             if e else "the chip or its ETH side, not the cables: ") + "every failing link has it at one end and the far-end chips pass their other links"
        return c, f"reset and rerun C5; if it fails again, replace the board of tray {u} (chip {o.group(1)}); logs {path_of(l, findings, stage)}"
    if kind == "link" and not hang:   # a pair line of a chip that a one-chip line names: that chip, not this link
        for x in lines:
            o = ONECHIP.search(x)
            if o and o.group(1) in (LINK.search(l).group(1), LINK.search(l).group(4)):
                return (f"one of the links of {o.group(1)} (tray {o.group(2)}), which fails all its C5 pairs: that chip, not this link",
                        f"see the one-chip FAIL of {o.group(1)} (tray {o.group(2)})")
    if "incomplete (test-infra)" in s:   # the pair could not start (job's process/thread limit), the link was not tested
        return ("test infrastructure: the pair could not start (out of processes/threads: the job's ulimit -u or cgroup "
                "pids.max); not a chip or link fault", "rerun with a higher process limit; this link was not tested")
    if warn and "still hidden" in s and "port_status mask" in s:   # run.sh --repro link-a left its mask on
        return ("in-box ETH link hidden by the link-A repro's port_status mask, no reset after it: not a fault",
                "undo: tt-smi -glx_reset (a reset of this host's chips, then the root-port rescan and recount)")
    if warn and "reset marker slow to clear" in s:   # tt-umd's 'Still waiting' after 5 s of its 10 s budget
        return ("chip slow to finish its warm reset (marker still set after 5 s)" + ("; it then timed out, see its RESET FAIL line"
                if "timed out" in s else ", cleared in time: no fault proven"),
                f"watch it; a marker timeout (RESET FAIL) that repeats: tt-smi reset it again, then the board of tray {u}")
    if kind == "link" and hang and not warn:
        return "link pair hung: cable, connector or an end chip", f"reseat or replace the cable; triage {path_of(l, findings, stage)}"
    if l.startswith("ETH INCOMPLETE "):   # cluster ports that leave the box, held at pre, no reset judged them
        w = re.search(r"; not checked: (.*)$", l)
        return ("external ETH port, not an in-box link on this platform; not checked" + (f": {w.group(1)}" if w else ""),
                "none on this box; it is judged once C2's reset can run")
    if "not checked" in s:
        return "numeric not checked (the readback failed): cause not determined", f"rerun the {stage or 'stage'}; read {path_of(l, findings, stage)}"
    if hang and kind != "link" or kind == "unnamed":
        p = path_of(l, findings, stage)
        if kind == "unnamed":
            return "hang or failure with no chip named", f"read the hold capture {p}; rerun the stage alone to name the chip"
        return "hang on the named chip(s): chip/board or its links, not determined", f"read the hold capture {p}; rerun the stage"
    if warn and "down at pre" in s and "recovered by" in s:   # a pre-run in-box port the start reset fixed
        cab = has_port(part_of(l, u)[1])   # no cable advice for a link with no QSFP-DD port
        return ("in-box ETH link down before the run, back after the start reset (C2): left down by an earlier run or a "
                "reset", "none now; " + ("reseat or replace that QSFP-DD cable" if cab else "this link has no cable: the "
                f"board of tray {u}") + " if it goes down again later in a run")
    if kind == "link" or re.match(r"(ETH|LINK|PINPOINT LINK) FAIL|C[34] FAIL mesh placement", l):
        if warn:   # rc 143/137 is the pair job killed at its time limit, not a link fault
            rc = re.search(r"pair run failed, rc (\d+)", l)
            part = part_of(l, u)[1] if kind == "link" else ""
            if rc and rc.group(1) in ("143", "137"):
                return (f"not a link fault: the link pair's test job was killed at its time limit (rc {rc.group(1)}), so the "
                        "link was not judged", "none for the hardware; rerun the link pair (C5) with a longer time limit")
            return ("link run failed without a hang or mismatch: cause not determined",
                    "rerun the link pair (C5); " + ("if it fails again with a hang or data mismatch, reseat or replace that "
                    "QSFP-DD cable" if has_port(part) else "this link has no cable (in-box or on-board): if it fails again "
                    "with a hang or data mismatch, the tray board"))
        q = bool(re.search(r"QSFP (ETH )?ports", l.split(";")[0]))   # this line's own port kind, not a gate why in its tail
        nv = re.search(r"; no C2 verdict \((.*)\)$", l)
        to = re.search(r"local ETH ports \[[0-9, ]*\] not up \(to (" + BDF + r" \(tray [1-4]\))", l)
        if nv and not q:   # in-box port down before the run that no reset rechecked
            return (f"in-box ETH link{' to ' + to.group(1) if to else ''} down before the run, not rechecked: {nv.group(1)}",
                    f"rerun once C2's reset can run; if it stays down after a reset, reseat or replace the cable, then the board of tray {u}")
        if not q and kind == "link" and not has_port(part_of(l, u)[1]):   # an in-box link has no cable
            return ("in-box link down or bad (no QSFP-DD port, no cable): the tray-to-tray connection or a tray board",
                    f"no cable to swap: reseat the trays at both ends; if it stays down after a reset, the board of tray {u}")
        return ("link down or bad: QSFP-DD cable or connector" + (" (QSFP to another host)" if q else ""),
                "reseat or replace that QSFP-DD cable" + ("" if q else f"; if it stays down after a reset, the board of tray {u}"))
    if warn and l.startswith("ETH WARN") and "leave the box" in l:   # standalone galaxy, ports 10-13
        return "ETH ports leave the box, not cabled on a standalone galaxy (warning)", "none: the port leaves the box (no cable on a standalone galaxy)"
    if warn and l.startswith("ETH WARN") and "not judged: no link map" in l:   # map asked for, none read
        return ("inter-host QSFP ETH ports down, not judged: the run read no link map (warning)",
                "none until a link discovery reads; then judged from the link map")
    if warn and l.startswith("ETH WARN") and "dropped by our own reset on" in l:   # any chip count
        return ("inter-host QSFP port up after the C2 reset and dropped by our own later reset: test-side, not this "
                "host's fault (warning)", "none on this host")
    if warn and l.startswith("ETH WARN") and "far end down or resetting" in l:   # one port, two or more trays
        return ("inter-host QSFP port down on chips of several trays at once: far end down or resetting, not this host's "
                "fault (warning)", "none on this host; recheck once the far host is up")
    if warn and l.startswith("ETH WARN"):
        return "ETH ports not up (warning)", "recheck after a reset; reseat or replace the cable if they stay down"
    if l.startswith("RESET FAIL") and "not recovered by tray re-power/glx_reset" in l and "cannot open it" in l:
        return ("chip lost after its warm reset and not recovered by its tray re-power or glx_reset: chip/board (hardware), "
                "not a load or CCL fault", f"power-cycle the box; if it stays unopenable, replace the board of tray {u}")
    if l.startswith("RESET FAIL") and "not recovered by tray re-power/glx_reset" in l and "tt-umd discovery fails on it" in l:
        return ("chip lost after its warm reset and not recovered by its tray re-power or glx_reset: it is on PCIe and opens, "
                "but tt-umd discovery fails on it: chip/board (hardware), not a load or CCL fault",
                f"power-cycle the box; if tt-umd discovery still fails on it, replace the board of tray {u}")
    if ("no such device" in s or "missing after" in s or "not answering" in s or "unknownx" in s or l.startswith("CHIP DROP")
            or "lost from pcie" in s):
        return ("chip fell off PCIe: chip/board or its PCIe path (slot, retimer)",
                f"read the AER/lspci capture; same chip again: replace the board of tray {u}; all of tray {u} at once: bus bar first")
    if "failed on it" in s and l.startswith("RESET"):
        return "chip did not take the warm reset: chip/board", f"tt-smi reset it again; if it repeats, replace the board of tray {u}"
    w = re.search(r"corrected AER peak (\d+)/min \(over ([0-9.]+)/min", l) if warn and "weak PCIe link" in l else None
    if w:   # the bus-bar/sustained per-minute AER poll (e.g. 5,230 GHES corrected in one run, dropped 2 h later)
        return (f"weak PCIe link: {w.group(1)} corrected AER errors a minute under load on its root port (warn over "
                f"{w.group(2)}/min): slot, retimer or board, an early sign of a chip that falls off PCIe",
                f"read the stage's aer_rate.txt and the lspci capture; reseat tray {u}; if it repeats, replace the board of tray {u}")
    w = re.match(r"AER FAIL .*? correctable kernel AER events (during the run \(.*?\)|over 7 days|since repair .*?)(, last [^=]*?)? = ", l)
    if w:   # C1's journal outlier: name its window, so a past AER storm is not read as a new fault
        return (f"correctable AER rate outlier {w.group(1)}{w.group(2) or ''} in the kernel journal"
                f"{'' if w.group(1).startswith('during the run') else ', not only this run'}: "
                "PCIe path (chip, slot, retimer or link)", f"read the AER/lspci capture; reseat tray {u}; if it repeats, "
                f"replace the board of tray {u}")
    if "SETUP/SOFTWARE: fabric setup failed" in l:   # fl_setup_err's label, a fabric-setup error, not this tray's hardware
        return ("test infrastructure: fabric setup failed (setup or software: fabric descriptors or the tt-metal tree), "
                "not this tray's hardware", "fix the setup error the run's log names, then run run.sh again")
    if re.match(r"(PCIE|AER) FAIL|PREFLIGHT FAIL .*PCIe", l):
        return "PCIe path (link speed/width or uncorrectable errors): slot, retimer or board", f"reseat tray {u}; if it repeats, replace the board of tray {u}"
    if l.startswith("NUMERIC FAIL") and "CCL group fault" in l:   # one bad result spread by an all-gather
        src = ccl_source(l, lines)
        if not src:
            return ("CCL group fault: one bad result spread over the group's all-gather, source not isolated (no chip of "
                    "the group failed a check outside the CCL)",
                    "rerun the numeric pinpoint on every chip of the group; replace only the board of a chip that fails it")
        n = ", ".join(f"{b} (tray {t})" for b, t in src)
        ts = sorted({t for _, t in src})
        return (f"CCL group fault: one bad result from the likely source {n} spread over the group's all-gather"
                + ("" if u in ts else f" (tray {u}'s chips only received it)") + f": chip/board of {n}",
                f"rerun the numeric pinpoint on {n}; if it repeats, replace the board of tray {'/'.join(ts)}")
    if re.match(r"(NUMERIC|CHUNK|VAE) FAIL|C4v? FAIL", l) and "not checked" not in s:
        side = compute_side(l)
        if side:   # a matmul's bank list follows its output columns: the words' lane or column skew names compute
            return (f"chip compute fault (or its power under load): chip/board; {side}, not a DRAM bank",
                    f"rerun the numeric pinpoint on this chip; if it repeats, replace the board of tray {u}")
        rf = read_fault(l, u)
        if rf:   # one bad read across a whole core column is no single core's fault
            return rf
        cc = re.search(r"compute core \((\d+),(\d+)\)", l)
        if cc:   # the owning core of the wrong outputs, no DRAM readback failed
            bd = re.search(r"([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])", l)
            a_ = f"ASIC {bd.group(1)} " if bd else ""
            el = re.search(r"element \((\d+),(\d+)\)", l)
            wc = re.search(r"compute core \(\d+,\d+\) \(translated worker core (\d+),(\d+)\)", l)
            return (f"chip compute fault in {a_}compute core ({cc.group(1)},{cc.group(2)})"
                    + (f" (translated worker core {wc.group(1)},{wc.group(2)})" if wc else "")
                    + (f", element ({el.group(1)},{el.group(2)}) of its output tiles" if el else "") + ": tray board",
                    f"replace the tray {u} board ({a_}compute core ({cc.group(1)},{cc.group(2)}))")
        # a tile index mod 8 is no DRAM bank; the rw split's per-bank counts are a DRAM readback, so they stay
        d = "dram" in s.replace("in-dram matmul", "") or ("banks [" in s and " rw)" in s)
        return (f"chip {'DRAM' if d else 'compute'} fault{decoupled_bank(l) if d else ''}: chip/board",
                f"rerun the numeric pinpoint on this chip; if it repeats, replace the board of tray {u}")
    if l.startswith("QUIET FAIL"):
        return "chip stayed above idle clock in the quiet: cause not determined", "read tt-smi telemetry of that chip; rerun"
    if PCIE_WHY and l.startswith("PREFLIGHT FAIL") and re.search(r"_ver 255\.255\.255\.255\b", l):   # all-Fs reads
        return ("chip not answering over PCIe (all-Fs reads), not a firmware mismatch",
                "glx_reset and recount; if it repeats after power cycles, the tray's PCIe path")
    if l.startswith("PREFLIGHT FAIL") and "_ver" in s:
        return "firmware version differs from the other chips", f"reflash tray {u} to the box's firmware bundle"
    if l.startswith("LOAD LOW WARN "):   # the stage never loaded this tray to its chips' own limits
        return (f"the stage's load stayed under 90% of tray {u}'s chips' own power limits for most of it: a clean result "
                "there is not a clean pass, a weak tray may not have been loaded hard enough to drop",
                "no fault proven; the load mix toward each chip's own limit is the tuning step")
    if l.startswith("LOAD UNREAD WARN "):
        return (f"tray {u}'s chip power or power limit was not read for enough of the load: whether it was loaded to its "
                "limit is not known (neither a pass nor a failed check)", f"read tray {u}'s hwmon power1_input and power1_max; rerun")
    if warn:
        return "warning only, no fault proven", "watch it in the next run"
    return "cause not determined", f"rerun run.sh --heavy-logs on this host; read {findings}"


LANE, COL_SKEW, CONC = 0.9, 2.0, 0.8   # as pinpoint_verdict.py: one lane parity's share, top column group vs its even share


def compute_side(l):
    """A NUMERIC FAIL line whose bad words sit on one lane parity or one output column group -> the evidence phrase, else
    None. One bank with ', columns decoupled from banks' stays a DRAM cause (pinpoint_verdict.py)."""
    m = re.search(r"lanes even/odd (\d+)/(\d+)", l)
    if "columns decoupled from banks" in l:
        b = re.search(r"banks \[([\d, ]+)\]", l)
        v = [int(x) for x in b.group(1).split(",")] if b else []
        if v and sum(v) and max(v) >= CONC * sum(v):
            return None
    if m:
        e, o = int(m.group(1)), int(m.group(2))
        if e + o and max(e, o) >= LANE * (e + o):
            return f"bad words on {'odd' if o > e else 'even'} lanes {max(e, o)}/{e + o}, they follow output columns"
    c = re.search(r"output column groups \[([\d, ]+)\]", l)
    if c:
        v = [int(x) for x in c.group(1).split(",")]
        if len(v) > 1 and sum(v) and max(v) >= COL_SKEW * sum(v) / len(v):
            return f"bad words in output column group {v.index(max(v))} {round(100 * max(v) / sum(v))}% of {len(v)} groups"
    return None


def decoupled_bank(l):
    """' in GDDR bank N (...)' when the columns were decoupled from banks and one bank holds CONC of the bad words, else ''.
    The only bank this line names: a tile column mod 8 is never one."""
    b = re.search(r"banks \[([\d, ]+)\]", l)
    v = [int(x) for x in b.group(1).split(",")] if b and "columns decoupled from banks" in l else []
    if not v or not sum(v) or max(v) < CONC * sum(v):
        return ""
    return f" in GDDR bank {v.index(max(v))} ({max(v)}/{sum(v)} bad words, columns decoupled from banks)"


def tile_cols(l):
    """The bad output tile columns a NUMERIC FAIL line names: its positions list, else its first bad tile."""
    p = re.search(r"element row,element col\) (\[\[[\d, \[\]]*\]\])", l)
    if p:
        return [int(t[3]) for t in json.loads(p.group(1))]
    t = re.search(r"of tile \d+,(\d+)|\(tile \d+,(\d+),", l)
    return [int(t.group(1) or t.group(2))] if t else []


def column_read(l):
    """(tile column, core x, cores) when one bad read reached every core of one core column: the bad tiles in one tile
    column, the bad cores in one core column and in each of its rows (the line lists at most 8 cores), else None."""
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


def read_fault(l, u):
    """(cause, next) for one bad read that reached every core of one core column: a shared input read, not one
    compute core; else None."""
    bd, cr = re.search(r"FAIL " + BDF, l), column_read(l)
    if not bd or not cr:
        return None
    return (f"chip read fault: one bad read of tile column {cr[0]} reached every core of core column {cr[1]} (compute cores "
            + ", ".join(f"({cr[1]},{y})" for y in cr[2]) + "), not one compute core: chip/board",
            f"rerun the numeric pinpoint on ASIC {bd.group(1)} with the DRAM readback split and the decoupled arm to tell "
            f"its memory from its NoC; if it repeats, replace the tray {u} board")


def result(lines, stages=None, drop_trays=None, known=None):
    """The failed checks fl_report prints as FAIL tray lines, as one phrase for run.sh's RESULT line (never
    'no loss' over a FAIL tray), or '' when every tray passes. known: the --known-dropped-tray, whose own lines do not count."""
    stages, seen, out, trays = stages or {}, set(), [], []
    for u in "1234":
        if known and u == str(known):
            continue
        for l in tray_lines(lines, u, drop_trays, stages)[0]:
            ts = set(re.findall(r"tray ([1-4])(?![0-9])", re.sub(r", BMC after: .*", "", l)))
            if known and ts == {str(known)}:
                continue
            if not l.startswith("NUMERIC FAIL"):
                trays.append(f"tray {u}")   # every tray with a FAIL line, both ends of a link; the line counts once
            if l in seen:
                continue
            seen.add(l)
            if l.startswith("NUMERIC FAIL"):
                k = re.search(r"(\d+/\d+) (checks|launches)", l)
                me = re.search(r"max abs err (\S+), wrong elements (\d+), (all finite|non-finite values present)", l)
                ev = ", ".join(x for x in ((f"{k.group(1)} {k.group(2)}" if k else ""), stage_of(l, stages),
                                           (f"max abs err {me.group(1)}, {me.group(2)} wrong elements, {me.group(3)}" if me else "")) if x)
                chips = ", ".join(f"{b} tray {t}" for b, t in dict.fromkeys(CHIP.findall(l))) or f"tray {u}"
                out.append(f"numeric fault {chips}" + (f" ({ev})" if ev else ""))
            else:
                out.append(f"tray {u}")
    num = [o for o in out if o.startswith("numeric fault")]
    other = list(dict.fromkeys(trays))
    n = len(out) - len(num)
    return "; ".join(num + ([f"{n} failed check{'s' * (n > 1)} on {', '.join(other)}"] if n else []))


def why(lines, u, stages=None, cap=10, findings="findings.txt", drop_trays=None):
    u = str(u)
    stages = stages or {}
    fails, warns = tray_lines(lines, u, drop_trays, stages)
    # the tray's power loss first, within the cap, then a load hang's pinpoint
    fails.sort(key=lambda l: (not l.startswith("POWER "), not l.startswith("HANG PINPOINT")))
    out = []
    for l in fails + warns:
        l = re.sub(r"UBB([0-3])", lambda m: f"tray {int(m.group(1)) + 1}", l.rstrip("\n"))   # old lines: tray numbers only
        kind, part = part_of(l, u)
        st = stage_of(l, stages)
        c, n = cause(l, kind, u, findings, st, lines)
        o = ONECHIP.search(l)
        if o:   # its links one per line under it: both ends, eth channels
            ev = f"all {o.group(3)} failing {o.group(4)} pairs have this chip at one end ({o.group(5)})"
            out.append(f"    {part} | {st} | {ev} | cause: {c} -> next: {n}")
            out += [f"      link {a} e{ca} <-> {b} e{cb}: {w}" for a, ca, b, cb, w, _ in ONELINK.findall(o.group(6))]
            t = re.search(r"tt-metal errors on this chip in .*?(?= \(/[^)]*\)$|$)", o.group(6))
            if t:
                out.append(f"      {t.group(0).strip()}")
            continue
        out.append(f"    {part} | {st} | {evidence(l, u)} | cause: {c} -> next: {n}")
    if len(out) > cap:
        out = out[:cap] + [f"    ... {len(out) - cap} more for tray {u} in {findings}"]
    return out


def main(a):
    if len(a) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    f, u = a[0], a[1]
    if u == "--result":   # tray_why.py <findings.txt> --result [--also <numeric.txt>] [--stages ..] [--drop-trays ..] [--known-tray N]
        u = "0"
    st, cap = None, 10
    if "--stages" in a:
        st = a[a.index("--stages") + 1]
    if "--max" in a:
        cap = int(a[a.index("--max") + 1])
    dt = a[a.index("--drop-trays") + 1].split() if "--drop-trays" in a else None
    stages = {}
    if st and os.path.exists(st):
        for r in open(st, errors="replace"):
            p, _, l = r.rstrip("\n").partition("\t")
            stages.setdefault(l, p)
    try:
        lines = open(f, errors="replace").read().splitlines()
    except OSError:
        return 0
    if u == "0":
        if "--also" in a:   # run.sh's numeric_report.py lines, as fl_report adds them
            try:
                lines += [l for l in open(a[a.index("--also") + 1], errors="replace").read().splitlines()
                          if re.match(r"NUMERIC (FAIL|WARN) ", l)]
            except OSError:
                pass
        r = result(lines, stages, dt, a[a.index("--known-tray") + 1] if "--known-tray" in a else None)
        if r:
            print(r)
        return 0
    for o in why(lines, u, stages, cap, f, dt):
        print(o)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
