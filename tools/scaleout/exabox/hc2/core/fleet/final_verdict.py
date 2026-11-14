#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The run's plain-words final verdict.

It reads the run's finding lines (FL_OUT/findings.txt: 'ETH FAIL ...', 'NUMERIC FAIL ...', ...) and its RESULT line and
prints, just before the 'Repair recommendation:' line:

  Final verdict: FAIL: 2 faults: Ethernet link down; wrong compute results
    Fault 1: Ethernet link down, still down after the start (C2) and end (C9) resets | where: in-box link between
      tray 2 and tray 4 (no QSFP-DD port): tray 2 ASIC 4 / 44:00.0 [ETH ch 4,5] <-> tray 4 ASIC 4 / 84:00.0 [ETH ch 4,5]
      | do: ...
    Note 1: old PCIe errors ... none during this run, not a failure

A chip is named by tray 1-4, ASIC 1-8 (the bus number's low digit) and BDF; an Ethernet link end by tray / faceplate
QSFP-DD port / ASIC / BDF, with the ETH channel numbers only in brackets (channels are not ports).
An INCOMPLETE run is never a fault, and kernel-log PCIe (AER) history from before the run is at most a note.

  final_verdict.py --rc N --verdict TEXT [--findings FILE] [--result FILE|--result-text T] [--commit C] [--mode M]
  final_verdict.py --log run.log      # an older run's log: its RESULT, VERDICT and finding lines

Any read or parse error prints 'Final verdict: see the FAIL lines above' (the run's last lines are always printed).
"""
import argparse
import re
import sys

BDF = r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
TRAY = {0x0: 1, 0x4: 2, 0xC: 3, 0x8: 4}  # bus high digit -> tray (fleet_py.tbdf)
# faceplate QSFP-DD port -> (ASIC, ETH channels); the same map as fleet_py.QSFP (tt-metal board.cpp UBB_BLACKHOLE)
QSFP = {1: (5, (2, 3)), 2: (1, (2, 3)), 3: (1, (0, 1)), 4: (2, (0, 1)), 5: (3, (0, 1)), 6: (4, (0, 1))}
# faceplate QSFP-DD ports 7-14 -> (ASICs, ETH channel): the inter-host ports, channel 8 or 9 of two ASICs each (board.cpp)
QSFP_X = {
    7: ((1, 2), 8),
    8: ((5, 6), 8),
    9: ((3, 4), 8),
    10: ((7, 8), 8),
    11: ((1, 2), 9),
    12: ((5, 6), 9),
    13: ((3, 4), 9),
    14: ((7, 8), 9),
}
RESET = {"c2": "start (C2)", "c9": "end (C9)"}
CAP = 12  # faults printed; the rest are counted


def tray_asic(b):
    bus = int(b[:2], 16)
    return TRAY.get(bus >> 4), bus & 0xF


def chip(b, idx=None):
    t, a = tray_asic(b)
    return f"tray {t} ASIC {a} / {b}" + (f" (tt-smi chip {idx})" if idx is not None else "")


def chans(s):
    return [int(x) for x in re.findall(r"\d+", s)]


def smi_ch(p):
    """ETH channel of tt-smi ETH port p, as fleet_py.ethboth maps them: ports 0-7 = ch 0-7, ports 10-13 = ch 8-11;
    ports 8 and 9 have no channel (None)."""
    return p if p < 8 else (p - 2 if 10 <= p <= 13 else None)


def ports(b, chs):
    """Faceplate QSFP-DD ports of chip b's ETH channels chs; [] when none of them is cabled to the faceplate."""
    _, a = tray_asic(b)
    got = {p for c in chs for p, (pa, pc) in QSFP.items() if pa == a and c in pc}
    return sorted(got | {p for c in chs for p, (pa, pc) in QSFP_X.items() if a in pa and c == pc})


def reset_words(labels):
    """'start (C2) reset', 'end (C9) reset' or 'start (C2) and end (C9) resets' of reset labels (c2, c9)."""
    ws = [RESET.get(x, x.upper()) for x in sorted(set(labels), key=lambda x: (x not in RESET, x))]
    return " and ".join(ws) + (" resets" if len(ws) > 1 else " reset")


def end(b, chs):
    t, a = tray_asic(b)
    ps = ports(b, chs)
    ch = f" [ETH ch {','.join(map(str, chs))}]" if chs else ""
    if ps:
        return f"tray {t} QSFP-DD port {' and '.join(map(str, ps))} / ASIC {a} / {b}{ch}"
    return f"tray {t} ASIC {a} / {b}{ch}"


def end1(b, chs):
    """end() of a lone link end; one with no faceplate port says so."""
    return end(b, chs) + ("" if ports(b, chs) else " (no QSFP-DD port)")


def link_where(a, ca, b, cb):
    """(where, kind) of the link a[ca] <-> b[cb]; kind: cable, inbox, onboard."""
    ta, tb = tray_asic(a)[0], tray_asic(b)[0]
    pa, pb = ports(a, ca), ports(b, cb)
    w = f"{end(a, ca)} <-> {end(b, cb)}"
    if pa and pb:
        return "QSFP-DD cable " + w, "cable"
    if ta == tb:
        return f"on-board link inside tray {ta} (no QSFP-DD port): " + w, "onboard"
    return f"in-box link between tray {ta} and tray {tb} (no QSFP-DD port): " + w, "inbox"


def link_do(kind, a, b, ca=(), cb=()):
    ta, tb = tray_asic(a)[0], tray_asic(b)[0]
    if kind == "cable":
        pa, pb = ports(a, ca), ports(b, cb)
        return (
            f"reseat, then replace, the QSFP-DD cable from tray {ta} port {'/'.join(map(str, pa))} to tray {tb} port "
            f"{'/'.join(map(str, pb))}; if it stays down with a good cable, the port on one of the two trays"
        )
    if kind == "onboard":
        return f"no cable to swap: the tray {ta} board (link between ASIC {tray_asic(a)[1]} and ASIC {tray_asic(b)[1]})"
    return (
        f"no cable to swap: reseat trays {ta} and {tb}; if it stays down, the tray-to-tray connection or one of the two "
        f"tray boards (swap a tray to see which end follows)"
    )


def resets(rest):
    r = rest.lower()
    if "no c2 verdict" in r or "reset skipped" in r:
        return None
    c2 = "still down after the c2 reset" in r
    c9 = "c9 reset: still down" in r or "still down after the c9 reset" in r
    if c2 and c9:
        return "still down after the start (C2) and end (C9) resets"
    if c2:
        return "still down after the start (C2) reset"
    if c9:
        return "still down after the end (C9) reset"
    if "back after" in r or "came back" in r:
        return "came back after a reset (flaky)"
    return "down"


class V:
    def __init__(self):
        self.faults, self.notes, self.seen, self.fw, self.down = [], [], set(), {}, {}

    def fault(self, key, what, where, do):  # a later line on the same part (the C9 recheck) replaces the earlier one
        if key in self.seen:
            self.faults = [(k, f) if k != key else (k, (what, where, do)) for k, f in self.faults]
            return
        self.seen.add(key)
        self.faults.append((key, (what, where, do)))

    def note(self, key, text):
        if key in self.seen:
            return
        self.seen.add(key)
        self.notes.append(text)


def eth(v, l):
    m = re.match(
        rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\) (\w+): local ETH ports \[([0-9, ]*)\] (?:enabled but )?not up"
        rf" \(to ((?:[^()]|\([^()]*\))*)\)(.*)$",
        l,
    )
    if m:
        a, lc, peers, rest = (
            m.group(2),
            chans(m.group(4)),
            re.findall(rf"({BDF}) \(tray [1-4]\) port (\d+)", m.group(5)),
            m.group(6),
        )
        by = {}
        for c, (b, pc) in zip(lc, peers):
            by.setdefault(b, ([], []))
            by[b][0].append(c)
            by[b][1].append(int(pc))
        st = resets(rest)
        for b, (ca, cb) in by.items():
            x, y = sorted([(a, tuple(ca)), (b, tuple(cb))])
            where, kind = link_where(x[0], list(x[1]), y[0], list(y[1]))
            if st is None:
                why = re.search(r"\(C2 reset skipped: ([^:)]*)", rest)
                v.note(
                    ("eth", x, y),
                    f"Ethernet link down before the run, not judged: the start (C2) reset was skipped"
                    f"{' (' + why.group(1) + ')' if why else ''} | where: {where} | do: rerun run.sh with the start reset",
                )
            elif m.group(1) == "FAIL":
                v.fault(("eth", x, y), f"Ethernet link down, {st}", where, link_do(kind, x[0], y[0], x[1], y[1]))
        return True
    m = re.match(
        rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\) (\w+): (\d+) (QSFP cables?|wrap cables?) missing or bad: (.*?)"
        rf"(?: \(layout ([\w-]+)\))?(;.*)?$",
        l,
    )
    if m:
        wrap = m.group(5).startswith("wrap")
        lay = m.group(7) or "?"
        for a, b, ta, tb in re.findall(rf"({BDF})<->({BDF}) \(tray ([1-4])<->([1-4])\)", m.group(6)):

            def port(x):
                asic = tray_asic(x)[1]
                p = [k for k, (pa, _) in QSFP.items() if pa == asic]
                return ("2" if wrap else "2 or 3") if len(p) > 1 else (str(p[0]) if p else "?")

            ends = f"tray {ta} QSFP-DD port {port(a)} / ASIC {tray_asic(a)[1]} / {a} <-> tray {tb} QSFP-DD port {port(b)} / ASIC {tray_asic(b)[1]} / {b}"
            v.fault(
                ("cable", a, b),
                f"{'wrap ' if wrap else ''}QSFP-DD cable missing or bad (cabling judged against layout {lay})",
                ends,
                f"check that cable is plugged in at both ends, then reseat or replace it (cabling, for the DC)",
            )
        return True
    return False


def numeric(v, l):
    m = re.match(rf"^NUMERIC (FAIL|WARN) ({BDF}) \(tray [1-4]\) chip (\d+) \[([^\]]*)\]: (.*)$", l)
    if not m:
        return False
    b, idx, kind, t = m.group(2), m.group(3), m.group(4), m.group(5)
    k = (
        re.search(r"(\d+/\d+) checked outputs mismatched", t)
        or re.search(r"(\d+ mismatching chunks of \d+)", t)
        or re.search(r"(\d+/\d+) (?:launches )?mismatched", t)
    )
    hist = re.search(r"DRAM bank histogram \[([0-9, ]+)\]", t)
    banks = [str(i) for i, n in enumerate(chans(hist.group(1))) if n] if hist else re.findall(r"DRAM bank (\d)", t)[:1]
    stage = re.search(r"\((stage \d+[^:)]*)", t)
    known = re.search(r"matches (SYS-\d+ \([^)]*\))", t)
    what = "wrong compute results (numeric check" + (f", {kind.split(',')[0]} load" if kind else "") + ")"
    if k:
        what += f": {k.group(1).replace('mismatching chunks', 'chunks wrong')}" + (
            "" if "chunks" in k.group(1) else " outputs wrong"
        )
    where = (
        chip(b, idx) + (f", DRAM bank {','.join(banks)}" if banks else "") + (f" ({stage.group(1)})" if stage else "")
    )
    do = (
        f"hardware only if the same check on the same commit stays clean on clean hosts; then the "
        f"tray {tray_asic(b)[0]} board carrying {b}"
    ) + (f"; matches known issue {known.group(1)}" if known else "")
    if m.group(1) == "FAIL":
        v.fault(("num", b), what, where, do)
    return True


def eth_more(v, l):
    """ETH lines eth() does not read: ports down before the run on one chip, QSFP ports to another host, an ETH core
    not active again at mesh open. Each end is named by tray / QSFP-DD port / ASIC / BDF."""
    m = re.match(
        rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: enabled ETH ports \[([0-9, ]*)\] down before the run"
        rf" and still down after the (\w+) reset",
        l,
    )
    if m:  # tt-smi ETH port numbers: shown as ETH channels the way fleet_py maps them (ports 10-13 = ch 8-11)
        b, ps = m.group(2), tuple(chans(m.group(3)))
        t, _ = tray_asic(b)
        if m.group(1) == "FAIL":
            k = ("ethboth", t, ps)  # the same ports down on more chips of the tray, or rechecked after C9: one fault
            bs, rs = v.down.setdefault(k, ([], set()))
            bs.append(b) if b not in bs else None
            rs.add(m.group(4).lower())
            cs = [c for c in map(smi_ch, ps) if c is not None]
            lab = "; ".join(
                x
                for x in (
                    f"ETH ch {','.join(map(str, cs))}" if cs else "",
                    "tt-smi ETH port " + ",".join(str(p) for p in ps if smi_ch(p) is None) + " (no channel)"
                    if len(cs) < len(ps)
                    else "",
                )
                if x
            )
            f = (
                f"Ethernet ports down before the run, still down after the {reset_words(rs)}",
                f"tray {t} "
                + ", ".join(f"ASIC {tray_asic(x)[1]} / {x}" for x in bs)
                + f" [{lab}]"
                + ("" if any(ports(x, cs) for x in bs) else " (no QSFP-DD port)"),
                f"glx_reset and retest; if they stay down, the tray {t} board",
            )
            if k in v.seen:
                v.faults = [(kk, f if kk == k else ff) for kk, ff in v.faults]
            else:
                v.fault(k, *f)
        return True
    m = re.match(
        rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: inter-host QSFP ETH ports \[[0-9, ]*\] \(ch \[([0-9, ]*)\]\)"
        rf" down before the run and still down after the (\w+) reset",
        l,
    )
    if m:  # fleet_py.ethboth on a node whose link map shows inter-host links: channels 8, 9 = QSFP-DD ports 7-14
        b, cs = m.group(2), chans(m.group(3))
        if m.group(1) == "FAIL":
            v.fault(
                ("qsfp", b, tuple(cs)),
                f"QSFP-DD link to another host down before the run, still down after the "
                f"{reset_words([m.group(4).lower()])}",
                "QSFP-DD cable " + end1(b, cs) + " <-> another host",
                f"reseat, then replace, that QSFP-DD cable; if it stays down with a good cable, the port on tray "
                f"{tray_asic(b)[0]} or on the far host",
            )
        return True
    m = re.match(
        rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: QSFP ETH ports \[([0-9, ]*)\] not up"
        rf"(?: \(to ([\w-]+) tray [1-4] ({BDF}) ports \[([0-9, ]*)\]\))?(.*)$",
        l,
    )
    if m:
        b, cs = m.group(2), chans(m.group(3))
        far = f" <-> {m.group(4)} {end1(m.group(5), chans(m.group(6)))}" if m.group(4) else " <-> another host"
        idle = "up and idle" in m.group(7)
        if m.group(1) == "FAIL":
            v.fault(
                ("qsfp", b, tuple(cs)),
                "QSFP-DD link to another host down" + (" while the far end is up and idle" if idle else ""),
                "QSFP-DD cable " + end1(b, cs) + far,
                f"reseat, then replace, that QSFP-DD cable; if it stays down with a good cable, the port on tray "
                f"{tray_asic(b)[0]} or on the far host",
            )
        return True
    m = re.match(
        rf"^ETH (FAIL|WARN) ({BDF}) \(tray [1-4]\)(?: \w+)?: ETH core ([0-9-]+) \(channel (\d+)[^)]*\) not active"
        rf" again at mesh open(.*)$",
        l,
    )
    if m:
        b, c = m.group(2), int(m.group(4))
        if m.group(1) == "FAIL":
            v.fault(
                ("ethcore", b, c),
                f"Ethernet core {m.group(3)} not active again when the mesh opened",
                end1(b, [c]),
                f"glx_reset and retest; if it repeats, the tray {tray_asic(b)[0]} board carrying {b}",
            )
        return True
    return False


def fw_why(t, x, y, mj):
    """(reason, next step) of a tray firmware mismatch in plain words; x 'unread', mj e.g. '12 of 32' or '' (majority)."""
    if x == "unread":
        return (
            f"tray {t}'s firmware version could not be read while the other chips run {y}",
            f"check that tray {t}'s chips answer, reflash tray {t} to {y} if its firmware differs",
        )
    if mj:
        return (
            f"tray {t} runs firmware {x}; the chips run mixed firmware with no version on most of them ({y} on {mj} chips)",
            f"reflash every chip to one firmware version",
        )
    return f"tray {t} runs firmware {x} while the other chips run {y}", f"reflash tray {t} to {y}"


def preflight(v, l):
    m = re.match(
        rf"^PREFLIGHT (FAIL|WARN) ({BDF}) \(tray ([1-4])\) chip \d+: tt_fw_bundle_ver ([\w.]+), (\d+ of \d+) chips"
        rf" ([\w.]+)( \(no majority\))?",
        l,
    )
    if not m:
        return False
    t = int(m.group(3))
    x, y, mj, bs = v.fw.get(t, (m.group(4), m.group(6), m.group(5) if m.group(7) else "", []))
    v.fw[t] = (x, y, mj, bs + [m.group(2)])
    k = ("fw", t)
    why, do = fw_why(t, x, y, mj)
    w = (f"firmware mismatch: {why}", f"tray {t} ({len(v.fw[t][3])} chips)", f"{do}, then retest")
    if k in v.seen:
        v.faults = [(kk, w if kk == k else f) for kk, f in v.faults]
    else:
        v.fault(k, *w)
    return True


def chunk(v, l):
    m = re.match(r"^CHUNK FAIL chunk (\d+/\d+)(?: \([^)]*\))?: no chunk-end readback, the load died or hung(.*)$", l)
    if not m:
        return False
    tr = sorted(set(re.findall(r"tray ([1-4]):", m.group(2))))
    v.fault(
        ("chunk", m.group(1)),
        f"the load died or hung (chunk {m.group(1)} gave no end readback; numeric not checked)",
        ("trays " + ", ".join(tr)) if tr else "the box",
        "see the cause line and the Repair recommendation",
    )
    return True


def aer(v, l):
    m = re.match(rf"^AER (FAIL|WARN) ({BDF}) \(tray [1-4]\):? (.*)$", l)
    if not m:
        return False
    b, t = m.group(2), m.group(3)
    n = re.search(r"(\d+) (correctable|uncorrectable)", t)
    if re.search(r"history before the run|not only this run|over \d+ days|before the run", t):
        v.note(
            ("aer", b),
            f"old PCIe (AER) errors on {chip(b)} from the kernel log before this run"
            f"{' (' + n.group(0) + ' events)' if n else ''}, none during this run: not a failure",
        )
    elif m.group(1) == "FAIL":
        v.fault(
            ("aer", b),
            f"PCIe errors during the run (AER{': ' + n.group(0) if n else ''})",
            chip(b),
            f"the PCIe path of {b} (slot, retimer, cable), then the tray {tray_asic(b)[0]} board",
        )
    return True


def pcie(v, l):
    m = re.match(
        rf"^PCIE (FAIL|WARN) ({BDF}) \(tray [1-4]\)(.*?)(?:, root port ([0-9a-f:.]+))?(?:, host [\w-]+)?: (.*)$", l
    )
    if not m or m.group(1) != "FAIL":
        return bool(m)
    b, t = m.group(2), m.group(5)
    rp = f" (root port {m.group(4)})" if m.group(4) else ""
    sp = re.search(r"root port link ([\w.]+ x\d+)", t)
    if "off pcie" in t.lower():
        what = "chip off PCIe (config space reads ffff)" + (f", link at {sp.group(1)}" if sp else "")
        do = f"glx_reset and recount; if still missing, power cycle the box; then the tray {tray_asic(b)[0]} board or its PCIe slot"
    else:
        what = "weak PCIe link" + (f" ({sp.group(1)})" if sp else f": {t[:80]}")
        do = f"the PCIe path of {b} (slot, retimer), then the tray {tray_asic(b)[0]} board"
    v.fault(("pcie", b), what, chip(b) + rp, do)
    return True


def link(v, l):
    m = re.match(
        rf"^LINK (FAIL|HANG) ({BDF}) \(tray [1-4]\) e([0-9,]+) <-> ({BDF}) \(tray [1-4]\) e([0-9,]+): (.*)$", l
    )
    if m:
        a, ca, b, cb, t = m.group(2), chans(m.group(3)), m.group(4), chans(m.group(5)), m.group(6)
        x, y = sorted([(a, tuple(ca)), (b, tuple(cb))])
        where, kind = link_where(x[0], list(x[1]), y[0], list(y[1]))
        w = re.search(r"(\d+) words wrong", t)
        what = (
            "Ethernet link hung (no progress)"
            if m.group(1) == "HANG"
            else "wrong data over an Ethernet link" + (f" ({w.group(1)} words wrong)" if w else "")
        )
        v.fault(("link", x, y), what, where, link_do(kind, x[0], y[0], x[1], y[1]))
        return True
    m = re.match(rf"^LINK FAIL one chip ({BDF}) \(tray [1-4]\) tt-smi (\d+): (.*)$", l)
    if m:
        b = m.group(1)
        v.fault(
            ("chip", b),
            "every link test through this one chip failed",
            chip(b, m.group(2)),
            f"the tray {tray_asic(b)[0]} board carrying {b}",
        )
        return True
    return False


def thermal(v, l):
    m = re.match(
        r"^THERMAL FAIL (host|tray [1-4]|[^:]*): (\d+) SEL thermal trip event\(s\) during the run(?: \((.*)\))?", l
    )
    if not m:
        return False
    f = [x.strip() for x in (m.group(3) or "").split("|")]
    sensor = f[3].replace("Temperature ", "") if len(f) > 3 else "a temperature sensor"
    rd = re.search(r"Reading (\d+)", m.group(3) or "")
    pb = re.search(r"PWRBRK(\d+)", sensor)
    part = f"power brick {pb.group(1)}" if pb else sensor
    ev = f"BMC sensor {sensor}" + (f" read {rd.group(1)} C, above its upper critical limit" if rd else "")
    if pb:
        v.fault(
            ("brick", pb.group(1)),
            f"power brick {pb.group(1)} overheated during the run ({ev})",
            f"power brick {pb.group(1)}" + ("" if m.group(1) == "host" else f" ({m.group(1)})"),
            f"brick {pb.group(1)}'s heatsink and thermal contact, then replace brick {pb.group(1)}; not a tray reseat",
        )
        return True
    v.fault(
        ("thermal", sensor),
        f"over-temperature trip during the run ({m.group(2)} BMC event(s))",
        f"{m.group(1)}, {ev}",
        f"check the cooling and airflow of {part}, then {part} itself; the chips' own checks are listed separately",
    )
    return True


def generic(v, l):
    m = re.match(r"^((?:[A-Z][A-Z0-9-]*\s)+?)(FAIL|HANG)\b:?\s*(.*)$", l)
    if not m:
        return False
    t = m.group(3)
    bs = list(dict.fromkeys(re.findall(BDF, t)))
    tr = re.findall(r"tray ([1-4])", t)
    where = "; ".join(chip(b) for b in bs[:4]) or (f"tray {tr[0]}" if tr else "the box")
    v.fault(
        ("gen", l[:120]),
        f"{m.group(1).strip()} {m.group(2).lower()}: {t[:160]}",
        where,
        "see the Repair recommendation line" + (f" (tray {tray_asic(bs[0])[0]})" if bs else ""),
    )
    return True


def plain_reason(res):
    r = re.sub(r"^RESULT:\s*", "", res or "").strip()
    r = re.sub(r"^INCOMPLETE:\s*", "", r)
    r = r.split("; chips counted")[0]
    m = re.match(r"stopped by (SIG\w+) during (\w+) \((\d+) s in\)(?:, last step: (.*))?", r)
    if m:
        return (
            f"stopped from outside ({m.group(1)}) {int(m.group(3)) // 60} min in, during " f"{m.group(4) or m.group(2)}"
        )
    return r[:200]


def verdict(rc, vtext="", findings="", result="", commit="?", mode=""):
    v = V()
    for l in findings.splitlines():
        l = l.strip()
        if not l:
            continue
        for f in (eth, eth_more, preflight, chunk, numeric, aer, pcie, link, thermal):
            if f(v, l):
                break
        else:
            generic(v, l)
    fold(v)
    vt = (vtext or "").strip()
    rerun = "run.sh" + (f" --mode={mode}" if mode and mode not in ("quick", "repro") else "")
    out = []
    inc = vt.startswith("INCOMPLETE") or rc in (9, 129, 130, 143) or "INCOMPLETE" in (result or "")[:40]
    if vt.startswith("NOT RUN") or rc in (6, 10):
        out.append(
            f"Final verdict: NOT RUN: {vt[8:].strip(' ()') or f'exit {rc}'}: no hardware result; do: "
            f"{'wait for the other user, then ' if rc == 10 else ''}rerun {rerun}"
        )
    elif inc:  # never called a fault: what it saw before it stopped is listed as notes
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
            f"seen before it stopped, not judged"
            + (" (follows the tray firmware mismatch)" if fwt and (k[0] == "ethboth" and k[1] in fwt) else "")
            + f": {w} | where: {where}"
            for k, (w, where, _) in v.faults
            if k[0] != "fw"
        ]
        if fw:
            seen.insert(0, f"stopped by: {plain_reason(result) or vt}")
        v.notes[:0], v.faults = seen, []
    elif v.faults:
        kinds = list(dict.fromkeys(re.sub(r"[,:(].*", "", w).strip() for _, (w, _, _) in v.faults))
        out.append(f"Final verdict: FAIL: {len(v.faults)} fault{'s' if len(v.faults) > 1 else ''}: {'; '.join(kinds)}")
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
    for i, (_, (w, where, do)) in enumerate(v.faults[:CAP], 1):
        out.append(f"  Fault {i}: {w} | where: {where} | do: {do}")
    if len(v.faults) > CAP:
        out.append(f"  ... {len(v.faults) - CAP} more faults in the FAIL lines above")
    for i, n in enumerate(v.notes[:CAP], 1):
        out.append(f"  Note {i}: {n}")
    return out


def fold(v):
    """One fault per part: a missing-cable line and the Ethernet-link-down line of the same chips are one fault; a power
    brick overheat goes first, and the load and PCIe faults of that run are named as coming with it."""
    down = {frozenset((k[1][0], k[2][0])): w for k, (w, _, _) in v.faults if k[0] == "eth"}
    cab = {frozenset(k[1:3]) for k, _ in v.faults if k[0] == "cable"}
    v.faults = [
        (
            k,
            (
                w.replace(
                    "missing or bad",
                    "missing or bad: " + down[frozenset(k[1:3])].replace("Ethernet link down", "link down"),
                    1,
                )
                if k[0] == "cable" and frozenset(k[1:3]) in down
                else w,
                where,
                do,
            ),
        )
        for k, (w, where, do) in v.faults
        if not (k[0] == "eth" and frozenset((k[1][0], k[2][0])) in cab)
    ]
    br = [f for f in v.faults if f[0][0] == "brick"]
    if br:
        n = br[0][0][1]
        v.faults = br + [
            (
                k,
                (
                    w,
                    where,
                    f"comes with the power brick {n} overheat: fix brick {n} first and rerun; "
                    f"only if it repeats then, {do}"
                    if k[0] in ("aer", "chunk", "gen", "pcie")
                    else do,
                ),
            )
            for k, (w, where, do) in v.faults
            if k[0] != "brick"
        ]


def from_log(text):
    """(rc, verdict, findings, result) of an older run.log: its finding lines, with say()'s time stamp taken off."""
    ls = text.splitlines()
    ri = max([i for i, l in enumerate(ls) if l.startswith("RESULT")] or [0])
    res = ls[ri] if ls and ls[ri].startswith("RESULT") else ""
    vt = next((l[9:].strip() for l in reversed(ls) if l.startswith("VERDICT: ")), "")
    m = re.search(r"exit (\d+)", vt)
    rc = int(m.group(1)) if m else (0 if vt == "PASS" else 8)
    ls2 = [re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d UTC [\w.-]+: ", "", l) for l in ls]  # say()'s stamp
    fs = [l for l in ls2 if re.match(r"^[A-Z][A-Z0-9-]*(?: [A-Z][A-Z0-9-]*)* (FAIL|WARN|HANG) ", l)]
    return rc, vt, "\n".join(dict.fromkeys(fs)), res


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--rc", type=int, default=0)
    a.add_argument("--verdict", default="")
    a.add_argument("--findings", default="")
    a.add_argument("--result", default="")
    a.add_argument("--result-text", default="")
    a.add_argument("--commit", default="?")
    a.add_argument("--mode", default="")
    a.add_argument("--log", default="")
    o = a.parse_args(argv)

    def rd(p):
        try:
            return open(p, errors="replace").read() if p else ""
        except OSError:
            return ""

    try:
        if o.log:
            t = rd(o.log)
            if not re.search(r"^(RESULT|VERDICT|Repair recommendation):", t, re.M):  # a run still going, or cut off
                st = [l for l in t.splitlines() if l.startswith("STAGE ")]
                at = f"log ends after {st[-1].split(' VERDICT')[0]}" if st else "log has no stage line"
                print(
                    f"Final verdict: NO RESULT YET: {at}; the run is still going or was cut off; "
                    "do: wait for it to end, or rerun run.sh"
                )
                return
            rc, vt, fs, res = from_log(t)
        else:
            rc, vt, fs, res = o.rc, o.verdict, rd(o.findings), (rd(o.result).strip() or o.result_text)
        print("\n".join(verdict(rc, vt, fs, res, o.commit, o.mode)))
    except Exception:  # the run's last lines are always printed
        print("Final verdict: see the FAIL lines above")


if __name__ == "__main__":
    sys.exit(main())
