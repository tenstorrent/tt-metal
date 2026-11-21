# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Phase 6: the ETH status read (every on-host ETH link, read twice with tt-triage).

``tools/triage/triage.py`` runs ``check_eth_status`` over every chip twice,
ETH_STATUS_INTERVAL_S apart, and the two reads are judged against the BH Galaxy
UBB port map (``galaxy.UBB_ETH_PORTS``):

* links that are down or unreadable, eth cores whose heartbeat stopped, links
  that went away between the reads;
* error counters rising between the reads (uncorrected FEC, CRC) and links that
  retrained during the check;
* built-in links (board traces, linking boards) that are not linked, a fault on
  any Galaxy, and QSFP ports with only part of their lanes linked.

The QSFP cabling is described, never judged: it differs from host to host, and a
port linked on no UBB has no cable, a cable to another host or dead cables, which
nothing here can tell apart. A port linked on some UBBs and not on the others is
a WARN, since UBBs are normally cabled alike.

Nothing in this phase resets or writes to the unit; it reads the links as found.
That is also why it reads the PCIe state from sysfs before opening any chip, and
stops when one has dropped off: opening it could hang the read, and a reset to
recover it would wipe the evidence.
"""
from __future__ import annotations

import csv
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .common import emit_result, emit_running, log, run_with_timeout, write_command_log
from .galaxy import EXPECTED_CHIP_COUNT, QSFP_PORT_IDS, UBB_ETH_PORT_OF, UBB_ETH_PORTS, format_bdf, parse_bdf_physical
from .naming import blamed, link_end, port_name as _port_name
from .model import FAIL, PASS, SKIP, WARN, Check, Phase

TRIAGE_SCRIPT = "tools/triage/triage.py"
TRIAGE_REQUIREMENTS = "tools/triage/requirements.txt"

# What triage imports beyond the standard library. tools/triage/requirements.txt
# brings all of it: create_venv.sh installs it into python_env, the exabox-tools
# image into its venv.
TRIAGE_IMPORTS = "import ttexalens, capnp, docopt, rich"

# The first read also maps each triage chip id to its PCI address, and so to its
# UBB and ASIC slot; the second only has to read the links again.
FIRST_READ_SCRIPTS = ("device_info", "check_eth_status")
SECOND_READ_SCRIPTS = ("check_eth_status",)

# Seconds between the two reads. The FEC, CRC and retrain counters only go up
# until the chips are reset, so their totals say little on their own; what they
# did over this interval is the reading.
ETH_STATUS_INTERVAL_S = 30

# A backstop per read, not its expected duration. A chip that stops answering
# surfaces as exalens' own register timeout well inside this.
ETH_STATUS_READ_TIMEOUT_S = 300

# Per link. Rates are over the interval between the two reads.
UNCORR_FAIL_PER_S = 100.0  # uncorrected FEC codewords: the link corrupts data
UNCORR_WARN_PER_S = 10.0
CRC_WARN_PER_S = 1.0  # received frames that failed the Ethernet CRC
UNCORR_WARN_TOTAL = 1_000_000  # since the last reset; judged only when one read is all there is
RETRAIN_WARN_TOTAL = 3  # since the last reset

# Port states check_eth_status gives a port that is not meant to carry a link:
# "Unused" on Blackhole, "Disabled" / "Not connected" on Wormhole.
IDLE_PORT_STATES = frozenset({"unused", "disabled", "not connected"})
# Port states that leave the port itself unjudged.
ODD_PORT_STATES = frozenset({"Unknown", "Invalid", "Unsupported FW"})

TT_PCI_VENDOR = "0x1e52"
PCI_COMMAND_MEMORY = 0x2  # Command register bit: memory space decoding on

# What a built-in port is, for the operator deciding what to reseat. Every
# non-QSFP port that is not a trace is a linking board.
BUILTIN_KIND = {"TRACE": "on-board trace, nothing to reseat"}

# Names for the QSFP cabling found, keyed by the ports linked on all 4 UBBs.
# tt-metal's own cablings of one Galaxy (tools/scaleout/node/node.cpp: Y torus
# on QSFP 1-2, X torus on QSFP 3-6; the Z ports 7-14 in
# tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus_z_ports.textproto),
# plus a Galaxy inside a sub-torus cluster, whose QSFP 6-8, 11 and 12 lead to
# other hosts. A name describes; it never sets a status, so a cabling with no
# name here is not a finding.
QSFP_CABLINGS: dict[tuple[int, ...], str] = {
    (): "a mesh",
    (1, 2): "a Y torus",
    (3, 4, 5, 6): "an X torus",
    (1, 2, 3, 4, 5, 6): "an XY torus",
    tuple(range(1, 15)): "an XY torus plus the Z ports",
    (1, 2, 3, 4, 5, 9, 10, 13, 14): "a Galaxy inside a sub-torus cluster (QSFP 6-8, 11, 12 lead to other hosts)",
}

TOPOLOGY_CHECKS = (
    "eth_status_builtin_links",
    "eth_status_cable_lanes",
    "eth_status_cable_trays_alike",
    "eth_status_cabling",
)

# Rows of the summary's WORST LINKS table, kept in eth_status_fec_uncorrected's data.
WORST_LINKS = 5


def _findings(status: str, items: list[str]) -> list[dict]:
    """A check's offending items, one line each: ``data["findings"]`` on every
    check here, which is what the flash summary lists under its verdict."""
    return [{"status": status, "text": t} for t in items]


# ─────────────────────────────────────────────────────────────────────────────
# PCIe, from sysfs: nothing here opens a chip
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PcieFunction:
    bdf: str
    # Read live from config space: 0xFFFF once the chip no longer answers. Both
    # are None when config space could not be read at all.
    vendor: int | None
    command: int | None  # the PCI Command register

    @property
    def unverified(self) -> bool:
        return self.vendor is None or self.command is None

    @property
    def dropped(self) -> bool:
        return not self.unverified and (self.vendor == 0xFFFF or not self.command & PCI_COMMAND_MEMORY)


def read_pcie_functions(sys_root: Path = Path("/sys")) -> list[PcieFunction]:
    """Every Tenstorrent PCI function.

    The ``vendor`` attribute is cached at enumeration, so it still names a chip
    that has since dropped off the bus; config space is read live. A function
    whose config space cannot be read is kept, as unverified: triage opens every
    chip, so one that cannot be checked must stop the read like one that failed.
    """
    found = []
    for dev in sorted((sys_root / "bus" / "pci" / "devices").glob("*")):
        try:
            if (dev / "vendor").read_text().strip().lower() != TT_PCI_VENDOR:
                continue
        except OSError:
            continue
        try:
            with (dev / "config").open("rb") as f:
                config = f.read(6)
        except OSError:
            config = b""
        if len(config) == 6:
            found.append(
                PcieFunction(dev.name, int.from_bytes(config[0:2], "little"), int.from_bytes(config[4:6], "little"))
            )
        else:
            found.append(PcieFunction(dev.name, None, None))
    return found


def pcie_gate_check(functions: list[PcieFunction]) -> Check:
    """Whether the chips can be opened. A missing chip is pcie_enum_count's
    finding, so here it only narrows what the topology checks can judge."""
    dropped = [f for f in functions if f.dropped]
    unverified = [f for f in functions if f.unverified]
    data = {
        "present": len(functions),
        "expected": EXPECTED_CHIP_COUNT,
        "dropped": [{"bdf": f.bdf, "vendor": f"0x{f.vendor:04x}", "command": f"0x{f.command:04x}"} for f in dropped],
        "unverified": [f.bdf for f in unverified],
        "blamed_chips": blamed(f.bdf for f in dropped + unverified),
    }
    unread = f"{len(unverified)} more could not be checked: {_listing([format_bdf(f.bdf) for f in unverified])}"
    if not functions:
        return Check(
            name="eth_status_pcie_gate",
            status=SKIP,
            details="no Tenstorrent PCI function under /sys/bus/pci/devices; nothing to read",
            data=data,
            ip="pcie",
        )
    if dropped:
        listing = [
            f"{format_bdf(f.bdf)} "
            + ("not answering" if f.vendor == 0xFFFF else f"memory decoding off (Command=0x{f.command:04x})")
            for f in dropped
        ]
        data["findings"] = _findings(FAIL, [f"{item} on PCIe: do NOT reset, report it" for item in listing])
        return Check(
            name="eth_status_pcie_gate",
            status=FAIL,
            details=(
                f"{len(dropped)} of {len(functions)} chip(s) dropped off PCIe: {_listing(listing)}"
                + (f"; {unread}" if unverified else "")
                + ". Links not read and the unit left as found: report it before resetting, a reset wipes the evidence"
            ),
            data=data,
            ip="pcie",
        )
    if unverified:
        listing = [format_bdf(f.bdf) for f in unverified]
        data["findings"] = _findings(
            WARN,
            [f"{item}: PCI config space unreadable, so whether it is safe to open is not known" for item in listing],
        )
        return Check(
            name="eth_status_pcie_gate",
            status=WARN,
            details=(
                f"PCI config space of {len(unverified)} of {len(functions)} chip(s) could not be read: "
                f"{_listing(listing)}. Whether they are safe to open is not known, so the links were not read"
            ),
            data=data,
            ip="pcie",
        )
    details = f"{len(functions)}/{EXPECTED_CHIP_COUNT} chips on PCIe, all answering with memory decoding on"
    if len(functions) < EXPECTED_CHIP_COUNT:
        details += "; the links to a missing chip cannot come up, so which links are there is not judged"
    return Check(name="eth_status_pcie_gate", status=PASS, details=details, data=data, ip="pcie")


def sysfs_chip_bdfs(sys_root: Path = Path("/sys")) -> dict[int, str]:
    """Triage chip id -> PCI address, from /dev/tenstorrent numbering.

    The fallback for a read whose device_info table is missing or partial. UMD
    numbers the chips 0, 1, 2 ... in /dev/tenstorrent order, skipping the gaps a
    missing chip leaves.
    """
    nodes: dict[int, str] = {}
    for p in (sys_root / "class" / "tenstorrent").glob("tenstorrent!*"):
        try:
            nodes[int(p.name.rsplit("!", 1)[1])] = (p / "device").resolve().name
        except (ValueError, OSError):
            continue
    return {chip: bdf for chip, (_, bdf) in enumerate(sorted(nodes.items()))}


# ─────────────────────────────────────────────────────────────────────────────
# triage --llm-output
# ─────────────────────────────────────────────────────────────────────────────

# The line triage prints above each script's table: "check_eth_status.py:", or
# with --print-script-times "check_eth_status.py [1.23s]:".
SCRIPT_LINE_RE = re.compile(r"^([a-z_]+)\.py\b.*:$")
# An eth core's Loc cell: "<noc0 x-y> (e0,<channel>)".
ETH_LOC_RE = re.compile(r"(\d+-\d+)(?:\s*\(e\d+,(\d+)\))?")

Tables = dict[str, tuple[list[str], list[list[str]]]]


def parse_triage_tables(text: str) -> Tables:
    """The CSV tables ``triage.py --llm-output`` prints, as {script: (header, rows)}.

    Indented lines are a script's own pass/fail and warning lines. Triage's
    ERROR/WARNING lines are not indented, so a table can pick up a stray row;
    the readers below keep only the rows that parse.
    """
    tables: Tables = {}
    current: tuple[list[str], list[list[str]]] | None = None
    for line in text.splitlines():
        m = SCRIPT_LINE_RE.match(line)
        if m:
            current = tables[m.group(1)] = ([], [])
        elif current is not None and line.strip() and not line[0].isspace():
            header, rows = current
            cells = [v.strip() for v in next(csv.reader([line]))]
            if header:
                rows.append(cells)
            else:
                header.extend(cells)
    return tables


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value, 16) if value.lower().startswith("0x") else int(value)
    except ValueError:
        return None


def _columns(tables: Tables, script: str):
    header, rows = tables.get(script, ([], []))
    col = {name.lower(): i for i, name in enumerate(header)}

    def cell(row: list[str], name: str) -> str | None:
        i = col.get(name)
        if i is None or i >= len(row) or row[i] in ("", "N/A"):
            return None
        return row[i]

    return header, col, rows, cell


@dataclass(frozen=True)
class EthLink:
    """One eth core's row of check_eth_status."""

    dev: int
    noc: str  # noc0 "x-y": with dev, what identifies the core across the two reads
    channel: int | None
    port: str
    link: str | None  # live PCS state; older triage gives port and RX state only
    heartbeat: str | None
    rx: str | None
    retrain: int | None
    crc: int | None
    uncorr: int | None
    err_stat: int | None

    @property
    def idle(self) -> bool:
        return self.port.lower() in IDLE_PORT_STATES


def parse_eth_links(tables: Tables) -> dict[tuple[int, str], EthLink]:
    _header, col, rows, cell = _columns(tables, "check_eth_status")
    if "dev" not in col or "loc" not in col:
        return {}
    links: dict[tuple[int, str], EthLink] = {}
    for row in rows:
        dev = _int(cell(row, "dev"))
        loc = ETH_LOC_RE.fullmatch(cell(row, "loc") or "")
        if dev is None or loc is None:
            continue
        links[dev, loc.group(1)] = EthLink(
            dev=dev,
            noc=loc.group(1),
            channel=int(loc.group(2)) if loc.group(2) else None,
            port=cell(row, "port status") or "",
            link=cell(row, "link"),
            heartbeat=cell(row, "heartbeat"),
            rx=cell(row, "rx link up"),
            retrain=_int(cell(row, "retrain count")),
            crc=_int(cell(row, "crc errors")),
            uncorr=_int(cell(row, "uncorr cw")),
            err_stat=_int(cell(row, "err_stat")),
        )
    return links


def parse_chip_bdfs(tables: Tables) -> dict[int, str]:
    """Triage chip id -> PCI address, from device_info."""
    header, _col, rows, cell = _columns(tables, "device_info")
    out: dict[int, str] = {}
    for row in rows:
        if len(row) != len(header):
            continue
        chip, bdf = _int(cell(row, "chip")), cell(row, "pci bdf") or ""
        if chip is not None and parse_bdf_physical(bdf) is not None:
            out[chip] = bdf
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Judging two reads
# ─────────────────────────────────────────────────────────────────────────────


def _listing(items: list[str], limit: int = 6) -> str:
    shown = "; ".join(items[:limit])
    return shown + (f"; ... and {len(items) - limit} more" if len(items) > limit else "")


def _ranges(ns: Sequence[int]) -> str:
    """(1, 2, 4, 5, 6) -> '1,2,4-6'."""
    ns, out, i = sorted(ns), [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        out.append(str(ns[i]) if j == i else f"{ns[i]},{ns[j]}" if j == i + 1 else f"{ns[i]}-{ns[j]}")
        i = j + 1
    return ",".join(out)


def _ubbs(trays: Sequence[int]) -> str:
    return ", ".join(f"UBB{t}" for t in sorted(trays))


def _where(dev: int, channel: int | None, chip_bdfs: dict[int, str]) -> str:
    """'UBB2/U6 ch4 (TRACE P9, dev 13)', or 'dev 13 ch4' when the chip's place is not known."""
    ch = "" if channel is None else f" ch{channel}"
    pos = parse_bdf_physical(chip_bdfs.get(dev, ""))
    if pos is None:
        return f"dev {dev}{ch}"
    tray, asic = pos
    port = UBB_ETH_PORT_OF.get((asic, channel)) if channel is not None else None
    return f"UBB{tray}/U{asic}{ch} ({f'{_port_name(*port)}, ' if port else ''}dev {dev})"


def _link_state_checks(
    now: dict[tuple[int, str], EthLink], before: dict[tuple[int, str], EthLink] | None, where, ends
) -> list[Check]:
    """*where* names a link in words, *ends* gives the ``link_ends`` entries of a list of links."""
    live = [l for l in now.values() if not l.idle]
    down: list[tuple[EthLink, str]] = []
    if before is not None:
        down += [(before[k], f"{where(before[k])}: gone between the reads") for k in sorted(before.keys() - now.keys())]
    for k in sorted(now):
        l = now[k]
        if "unreadable" in (l.port.lower(), (l.link or "").lower()):
            down.append((l, f"{where(l)}: eth core unreadable"))
        elif l.idle:
            continue
        elif l.link is not None:
            if l.link.lower() != "up":
                down.append((l, f"{where(l)}: link down"))
        else:
            reasons = (["port down"] if l.port.lower() == "down" else []) + (
                ["RX link down"] if (l.rx or "up").lower() != "up" else []
            )
            if reasons:
                down.append((l, f"{where(l)}: {', '.join(reasons)}"))
    chips = len({l.dev for l in live})
    down_ends, down = ends([l for l, _ in down]), [t for _, t in down]
    links_down = Check(
        name="eth_status_links_down",
        status=FAIL if down else PASS,
        details=(
            f"{len(down)} of {len(live)} link(s) down or unreadable: {_listing(down)}"
            if down
            else f"{len(live)}/{len(live)} links up on {chips} chips"
            + (", none lost between the reads" if before is not None else "")
        ),
        data={
            "links": len(live),
            "chips": chips,
            "down": down,
            "findings": _findings(FAIL, down),
            "link_ends": down_ends,
        },
        ip="eth",
    )

    dead = [l for l in now.values() if (l.heartbeat or "").lower() == "down"]
    stopped = [where(l) for l in dead]
    beating = sum(1 for l in now.values() if l.heartbeat is not None)
    heartbeat = Check(
        name="eth_status_heartbeat",
        status=FAIL if stopped else PASS if beating else SKIP,
        details=(
            f"{len(stopped)} eth core(s) stopped, heartbeat not moving: {_listing(stopped)}"
            if stopped
            else f"heartbeat moving (or core released to metal) on {beating} eth cores"
            if beating
            else "check_eth_status reported no heartbeat"
        ),
        data={
            "stopped": stopped,
            "findings": _findings(FAIL, [f"{w}: heartbeat not moving, the eth core has stopped" for w in stopped]),
            "link_ends": ends(dead),
        },
        ip="eth",
    )

    oddly = [l for l in now.values() if l.port in ODD_PORT_STATES]
    odd = [f"{where(l)}: port {l.port}" for l in oddly]
    port_state = Check(
        name="eth_status_port_state",
        status=WARN if odd else PASS,
        details=(
            f"{len(odd)} port(s) in a state that cannot be judged: {_listing(odd)}"
            if odd
            else f"no port {', '.join(sorted(ODD_PORT_STATES))}"
        ),
        data={"odd": odd, "findings": _findings(WARN, odd), "link_ends": ends(oddly)},
        ip="eth",
    )
    return [links_down, heartbeat, port_state]


def _counter_checks(
    now: dict[tuple[int, str], EthLink],
    before: dict[tuple[int, str], EthLink] | None,
    interval_s: float | None,
    where,
) -> list[Check]:
    # Per link: (uncorrected FEC/s, CRC/s, retrains during the check). A counter
    # that went down means the chip was reset between the reads, and nothing is
    # read from that link's deltas.
    rates: dict[tuple[int, str], tuple[float | None, float | None, int | None]] = {}
    reset_between = 0
    if before is not None and interval_s:
        for k, l in now.items():
            a = before.get(k)
            if a is None:
                continue
            d = {
                f: getattr(l, f) - getattr(a, f)
                for f in ("uncorr", "crc", "retrain")
                if getattr(l, f) is not None and getattr(a, f) is not None
            }
            if any(v < 0 for v in d.values()):
                reset_between += 1
                continue
            rates[k] = (
                d["uncorr"] / interval_s if "uncorr" in d else None,
                d["crc"] / interval_s if "crc" in d else None,
                d.get("retrain"),
            )
    span = f"over {interval_s:.0f}s" if interval_s else ""
    reset_note = f"; {reset_between} link(s) reset between the reads, not rated" if reset_between else ""

    def rate(k: tuple[int, str], i: int) -> float | None:
        r = rates.get(k)
        return r[i] if r else None

    checks = []
    uncorr = {k: r[0] for k, r in rates.items() if r[0] is not None}
    totals = [l.uncorr for l in now.values() if l.uncorr is not None]
    worst = sorted(now.items(), key=lambda kl: (rate(kl[0], 0) or 0, kl[1].uncorr or 0), reverse=True)
    data = {
        "max_per_s": max(uncorr.values(), default=None),
        "max_total": max(totals, default=None),
        "interval_s": interval_s,
        "reset_between": reset_between,
        "worst_links": [
            {
                "link": where(l),
                "uncorr_per_s": rate(k, 0),
                "crc_per_s": rate(k, 1),
                "uncorr_total": l.uncorr,
                "retrains": l.retrain,
                "err_stat": l.err_stat,
            }
            for k, l in worst[:WORST_LINKS]
        ]
        if totals
        else [],
    }
    if not totals:
        status, details = SKIP, "check_eth_status reported no FEC counters"
    elif uncorr:
        ranked = sorted(uncorr.items(), key=lambda kv: -kv[1])
        bad = [f"{where(now[k])}: {v:,.0f}/s" for k, v in ranked if v >= UNCORR_FAIL_PER_S]
        high = [f"{where(now[k])}: {v:,.1f}/s" for k, v in ranked if UNCORR_WARN_PER_S <= v < UNCORR_FAIL_PER_S]
        data.update(fail=bad, warn=high)
        data["findings"] = _findings(
            FAIL,
            [
                f"{where(now[k])}: uncorrected FEC errors rising {v:,.0f}/s, the link corrupts data"
                for k, v in ranked
                if v >= UNCORR_FAIL_PER_S
            ],
        ) + _findings(
            WARN,
            [
                f"{where(now[k])}: uncorrected FEC errors rising {v:,.1f}/s"
                for k, v in ranked
                if UNCORR_WARN_PER_S <= v < UNCORR_FAIL_PER_S
            ],
        )
        if bad:
            status = FAIL
            details = f"uncorrected FEC errors rising on {len(bad)} link(s), these links corrupt data: {_listing(bad)}"
            if high:
                details += f"; {len(high)} more >= {UNCORR_WARN_PER_S:g}/s"
        elif high:
            status = WARN
            details = f"uncorrected FEC errors rising on {len(high)} link(s): {_listing(high)}"
        else:
            worst_k, worst = ranked[0]
            status = PASS
            details = f"max {worst:,.1f}/s {span} on {len(uncorr)} links (worst {where(now[worst_k])})"
        details += reset_note
    else:
        many_links = [l for l in now.values() if (l.uncorr or 0) >= UNCORR_WARN_TOTAL]
        many = [f"{where(l)}: {l.uncorr:,}" for l in many_links]
        data.update(warn=many)
        data["findings"] = _findings(
            WARN,
            [
                f"{where(l)}: {l.uncorr:,} uncorrected FEC errors since the last reset, read once so not known "
                f"whether still rising"
                for l in many_links
            ],
        )
        status = WARN if many else PASS
        details = (
            f"{len(many)} link(s) with >= {UNCORR_WARN_TOTAL:,} uncorrected FEC errors since the last reset; "
            f"one read only, so whether they are still rising is not known: {_listing(many)}"
            if many
            else f"one read only, no rate: max {max(totals):,} since the last reset"
        ) + reset_note
    checks.append(Check(name="eth_status_fec_uncorrected", status=status, details=details, data=data, ip="eth"))

    crc = {k: r[1] for k, r in rates.items() if r[1] is not None}
    rising = [(k, v) for k, v in sorted(crc.items(), key=lambda kv: -kv[1]) if v >= CRC_WARN_PER_S]
    high = [f"{where(now[k])}: {v:,.1f}/s" for k, v in rising]
    if not any(l.crc is not None for l in now.values()):
        status, details = SKIP, "check_eth_status reported no CRC counters"
    elif not crc:
        status, details = SKIP, "one read only: a CRC rate needs two"
    else:
        ranked = sorted(crc.items(), key=lambda kv: -kv[1])
        status = WARN if high else PASS
        details = (
            f"CRC errors (bad frames) rising on {len(high)} link(s): {_listing(high)}"
            if high
            else f"max {ranked[0][1]:,.1f}/s {span} on {len(crc)} links"
        ) + reset_note
    checks.append(
        Check(
            name="eth_status_crc_errors",
            status=status,
            details=details,
            data={
                "max_per_s": max(crc.values(), default=None),
                "warn": high,
                "findings": _findings(
                    WARN, [f"{where(now[k])}: CRC errors (bad frames) rising {v:,.1f}/s" for k, v in rising]
                ),
            },
            ip="eth",
        )
    )

    flapping = {k: r[2] for k, r in rates.items() if r[2]}
    # since_reset is every link at the threshold; the WARNs leave out the
    # flapping ones, which already have a FAIL of their own.
    at_threshold = [(k, l) for k, l in sorted(now.items()) if (l.retrain or 0) >= RETRAIN_WARN_TOTAL]
    since_reset = [f"{where(l)}: {l.retrain} since the last reset" for _, l in at_threshold]
    many_links = [l for k, l in at_threshold if k not in flapping]
    many = [f"{where(l)}: {l.retrain} since the last reset" for l in many_links]
    flap = [f"{where(now[k])}: {n}x" for k, n in sorted(flapping.items())]
    retrain_findings = _findings(
        FAIL,
        [
            f"{where(now[k])}: retrained {n}x during the check, the link is flapping"
            for k, n in sorted(flapping.items())
        ],
    ) + _findings(WARN, [f"{where(l)}: {l.retrain} retrains since the last reset" for l in many_links])
    if not any(l.retrain is not None for l in now.values()):
        status, details = SKIP, "check_eth_status reported no retrain counts"
    elif flap:
        status = FAIL
        details = f"{len(flap)} link(s) retrained during the check, they are flapping: {_listing(flap)}"
    elif many:
        status = WARN
        details = f"{len(many)} link(s) with >= {RETRAIN_WARN_TOTAL} retrains since the last reset: {_listing(many)}"
    else:
        status = PASS
        details = (
            "no link retrained during the check" if rates else "one read only: retrains during the check not seen"
        ) + f", none with >= {RETRAIN_WARN_TOTAL} since the last reset"
    checks.append(
        Check(
            name="eth_status_retrains",
            status=status,
            details=details,
            data={"during": flap, "since_reset": since_reset, "findings": retrain_findings},
            ip="eth",
        )
    )

    latched_links = [l for l in now.values() if l.err_stat]
    latched = [f"{where(l)}: 0x{l.err_stat:x}" for l in latched_links]
    if not any(l.err_stat is not None for l in now.values()):
        status, details = SKIP, "check_eth_status reported no ERR_STAT"
    else:
        status = WARN if latched else PASS
        details = (
            f"{len(latched)} link(s) with a latched link error: {_listing(latched)}" if latched else "none latched"
        )
    checks.append(
        Check(
            name="eth_status_err_stat",
            status=status,
            details=details,
            data={
                "latched": latched,
                "findings": _findings(
                    WARN, [f"{where(l)}: ERR_STAT 0x{l.err_stat:x} latched (a link error)" for l in latched_links]
                ),
            },
            ip="eth",
        )
    )
    return checks


def _topology_checks(now: dict[tuple[int, str], EthLink], chip_bdfs: dict[int, str], chips_on_pcie: int) -> list[Check]:
    live = [l for l in now.values() if not l.idle]
    slot: dict[tuple[int, int], int] = {}
    for dev, bdf in chip_bdfs.items():
        pos = parse_bdf_physical(bdf)
        if pos is not None:
            slot[pos] = dev
    if chips_on_pcie < EXPECTED_CHIP_COUNT:
        why = f"only {chips_on_pcie} of {EXPECTED_CHIP_COUNT} chips on PCIe, and the links to a missing chip cannot come up"
    elif any(l.channel is None for l in live):
        why = "triage printed no eth channel numbers"
    elif set(slot) != {(t, a) for t in range(1, 5) for a in range(1, 9)}:
        why = f"the chips' PCI addresses do not map onto 4 UBBs x 8 ASICs ({len(slot)} placed)"
    else:
        why = ""
    if why:
        return [
            Check(name=n, status=SKIP, details=f"which links are there is not judged: {why}", ip="eth")
            for n in TOPOLOGY_CHECKS
        ]

    linked = {(l.dev, l.channel) for l in live}
    builtin_off: list[str] = []
    builtin_findings: list[str] = []
    half: list[str] = []
    half_findings: list[str] = []
    qsfp_lanes: dict[int, dict[int, int]] = {}
    bi_have = bi_total = 0
    for (ptype, pid), lanes in sorted(UBB_ETH_PORTS.items()):
        for tray in range(1, 5):
            off = [(asic, ch) for asic, ch in lanes if (slot[tray, asic], ch) not in linked]
            lanes_txt = ", ".join(f"U{asic} ch{ch}" for asic, ch in off)
            port = f"UBB{tray} {_port_name(ptype, pid)}"
            if ptype == "QSFP_DD":
                qsfp_lanes.setdefault(pid, {})[tray] = len(lanes) - len(off)
                if 0 < len(off) < len(lanes):
                    have = f"{len(lanes) - len(off)} of {len(lanes)} lanes"
                    half.append(f"{port}: {have} ({lanes_txt} not)")
                    half_findings.append(
                        f"{port}: only {have} linked ({lanes_txt} not), the cable is there and part of it is dead"
                    )
            else:
                bi_total += len(lanes)
                bi_have += len(lanes) - len(off)
                if off:
                    kind = BUILTIN_KIND.get(ptype, "linking board")
                    builtin_off.append(f"{port}: {lanes_txt} ({kind})")
                    builtin_findings.append(f"{port}: {lanes_txt} not linked ({kind})")

    uneven: list[str] = []
    for pid in QSFP_PORT_IDS:
        n = len(UBB_ETH_PORTS["QSFP_DD", pid])
        full = [t for t, k in qsfp_lanes[pid].items() if k == n]
        none = [t for t, k in qsfp_lanes[pid].items() if k == 0]
        if full and none:
            uneven.append(f"QSFP P{pid}: linked on {_ubbs(full)}, not on {_ubbs(none)}")
    even = tuple(p for p in QSFP_PORT_IDS if all(k == len(UBB_ETH_PORTS["QSFP_DD", p]) for k in qsfp_lanes[p].values()))

    on_cables = sum(sum(per_tray.values()) for per_tray in qsfp_lanes.values())
    place = {dev: pos for pos, dev in slot.items()}
    unnamed = sorted(
        f"dev {l.dev} ch{l.channel}"
        for l in live
        if (place.get(l.dev, (None, None))[1], l.channel) not in UBB_ETH_PORT_OF
    )
    found = f"QSFP {_ranges(even)} linked on all 4 UBBs" if even else "No QSFP port linked inside this host"
    if half or uneven:
        shape = "QSFP ports not linked alike on every UBB, see eth_status_cable_lanes and eth_status_cable_trays_alike"
    elif even in QSFP_CABLINGS:
        shape = f"{found}, as in {QSFP_CABLINGS[even]}"
    else:
        shape = f"{found}: not a cabling with a name here, which says nothing about whether it is the intended one"
    counts = f"{len(live)} on-host links: {bi_have}/{bi_total} built-in + {on_cables} on cables"
    if unnamed:
        counts += f" + {len(unnamed)} on channels with no port name ({_listing(unnamed, 4)})"

    return [
        Check(
            name="eth_status_builtin_links",
            status=FAIL if builtin_off else PASS,
            details=(
                f"{bi_total - bi_have} of {bi_total} built-in links not linked, a fault on any Galaxy: "
                f"{_listing(builtin_off)}"
                if builtin_off
                else f"{bi_have}/{bi_total} built-in links linked (board traces and linking boards, 4 UBBs)"
            ),
            data={
                "linked": bi_have,
                "total": bi_total,
                "not_linked": builtin_off,
                "findings": _findings(FAIL, builtin_findings),
            },
            ip="eth",
        ),
        Check(
            name="eth_status_cable_lanes",
            status=FAIL if half else PASS,
            details=(
                f"{len(half)} QSFP port(s) with only part of their lanes linked: the cable is there and "
                f"part of it is dead: {_listing(half)}"
                if half
                else "no QSFP port linked in part"
            ),
            data={
                "partial": half,
                "findings": _findings(FAIL, half_findings),
            },
            ip="eth",
        ),
        Check(
            name="eth_status_cable_trays_alike",
            status=WARN if uneven else PASS,
            details=(
                f"{len(uneven)} QSFP port(s) linked on some UBBs and not on the others; UBBs are normally "
                f"cabled alike, so a cable is probably dead or missing: {_listing(uneven)}"
                if uneven
                else "no QSFP port linked on some UBBs and not on the others"
            ),
            data={
                "uneven": uneven,
                "findings": _findings(
                    WARN,
                    [f"{item}: UBBs are normally cabled alike, a cable is probably dead or missing" for item in uneven],
                ),
            },
            ip="eth",
        ),
        Check(
            name="eth_status_cabling",
            status=PASS,
            details=f"{shape}. {counts}",
            data={
                "linked_on_all_ubbs": list(even),
                "name": QSFP_CABLINGS.get(even) if not (half or uneven) else None,
                "lanes_per_port": {
                    str(pid): {f"UBB{t}": k for t, k in trays.items()} for pid, trays in qsfp_lanes.items()
                },
                "description": shape,
                "links": len(live),
                "builtin": bi_have,
                "builtin_total": bi_total,
                "on_cables": on_cables,
                "unnamed_channels": unnamed,
            },
            ip="eth",
        ),
    ]


def judge_eth_status(
    first: dict[tuple[int, str], EthLink],
    second: dict[tuple[int, str], EthLink] | None,
    interval_s: float | None,
    chip_bdfs: dict[int, str],
    chips_on_pcie: int,
) -> list[Check]:
    """The ETH checks from one or two reads. Pure: no hardware, so stored reads are judged the same."""
    now = second or first
    before = first if second else None

    def where(l: EthLink) -> str:
        return _where(l.dev, l.channel, chip_bdfs)

    def ends(links: list[EthLink]) -> list[dict]:
        return [
            {"a": link_end(chip_bdfs[l.dev], l.channel)} for l in links if l.dev in chip_bdfs and l.channel is not None
        ]

    return (
        _link_state_checks(now, before, where, ends)
        + _counter_checks(now, before, interval_s, where)
        + _topology_checks(now, chip_bdfs, chips_on_pcie)
    )


# ─────────────────────────────────────────────────────────────────────────────
# The phase
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TriageRead:
    name: str
    command: list[str]
    started: float
    duration_s: float
    rc: int
    timed_out: bool
    csv_path: Path
    log_path: Path
    stdout: str
    stderr: str
    tables: Tables
    links: dict[tuple[int, str], EthLink]


def resolve_triage_python(tt_metal: Path) -> str:
    """The repo venv's python, where create_venv.sh installs the triage
    requirements, else the one running the suite (the image's venv)."""
    venv = tt_metal / "python_env" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def triage_command(python: str, script: Path, scripts: Sequence[str], no_inspector: Path) -> list[str]:
    """The triage command line, with Inspector kept out of it.

    Triage tries the Inspector RPC server first and its saved logs second, and
    either would tell check_eth_status which eth cores a metal run found trained
    rather than the links as they are now. It has no switch to turn Inspector
    off, so both sources are pointed at nothing: TCP port 0 can never have a
    listener, so the RPC connect fails at once, and ``no_inspector`` must be a
    path that does not exist. Rank offsets are disabled so neither moves.
    """
    return [
        python,
        str(script),
        *(f"--run={s}" for s in scripts),
        "--dev=all",
        "--llm-output",
        "--inspector-rpc-host=127.0.0.1",
        "--inspector-rpc-port=0",
        "--inspector-disable-rank",
        f"--inspector-log-path={no_inspector}",
    ]


def _run_read(name: str, cmd: list[str], env: dict, cwd: Path, logs_dir: Path) -> TriageRead:
    csv_path = logs_dir / f"{name}.csv"
    log_path = logs_dir / f"{name}.log"
    # A read left by an earlier run would be judged as this one.
    csv_path.unlink(missing_ok=True)
    emit_running(name)
    started = time.time()
    try:
        rc, out, err, timed_out = run_with_timeout(cmd, ETH_STATUS_READ_TIMEOUT_S, cwd=str(cwd), env=env)
    except OSError as e:
        rc, out, err, timed_out = 127, "", repr(e), False
    dt = time.time() - started
    try:
        csv_path.write_text(out)
    except OSError as e:
        log(f"  could not write {csv_path}: {e!r}")
    write_command_log(log_path, shlex.join(cmd), rc, dt, f"(in {csv_path.name})" if out else "", err)
    tables = parse_triage_tables(out)
    links = parse_eth_links(tables)
    emit_result(name, PASS if links else WARN, suffix=f"({dt:.1f}s, {len(links)} eth cores)")
    return TriageRead(name, cmd, started, dt, rc, timed_out, csv_path, log_path, out, err, tables, links)


def _tail(read: TriageRead, n: int = 3) -> str:
    lines = [ln.strip() for ln in (read.stdout + "\n" + read.stderr).splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:300]


def run_eth_status(
    tt_metal: Path,
    phase: Phase,
    dry_run: bool,
    logs_dir: Path,
    *,
    sys_root: Path = Path("/sys"),
    interval_s: float = ETH_STATUS_INTERVAL_S,
) -> None:
    """Read every on-host ETH link twice and judge the two reads.

    * **SKIP** — no reading taken: triage is not in this checkout or cannot
      import its requirements, no chip is on PCIe, or one has dropped off it
      (``eth_status_pcie_gate`` then FAILs and says which).
    * **WARN** on ``eth_status_read`` — triage ran but listed no eth core, or
      only the first read did. Lost coverage, not a statement about the links.
    * Every other check is a finding about the links; see judge_eth_status.
    """
    script = tt_metal / TRIAGE_SCRIPT
    if not script.is_file():
        phase.add(Check(name="eth_status_read", status=SKIP, details=f"not present: {script}", ip="other"))
        return
    python = resolve_triage_python(tt_metal)
    # Never created: see triage_command. Without it triage would load
    # /tmp/tt-metal/inspector, left by the last metal run on this host.
    no_inspector = logs_dir / "eth_status_no_inspector"
    reads = (("eth_status_read_1", FIRST_READ_SCRIPTS), ("eth_status_read_2", SECOND_READ_SCRIPTS))
    commands = [triage_command(python, script, scripts, no_inspector) for _, scripts in reads]

    log("--- eth status ---")
    for (name, _), cmd in zip(reads, commands):
        log(f"  {name}: {shlex.join(cmd)}")

    if dry_run:
        for name, _ in reads:
            print(f"  {name:30} (dry-run)")
        plan = f" ; sleep {interval_s:.0f} ; ".join(shlex.join(cmd) for cmd in commands)
        phase.add(Check(name="eth_status_read", status=SKIP, details=f"(dry) {plan}", ip="other"))
        return

    functions = read_pcie_functions(sys_root)
    gate = pcie_gate_check(functions)
    phase.add(gate)
    if gate.status != PASS:
        reason = {
            FAIL: "a chip dropped off PCIe",
            WARN: "the PCIe state of a chip could not be read",
        }.get(gate.status, "no chip on PCIe")
        phase.add(Check(name="eth_status_read", status=SKIP, details=f"links not read: {reason}", ip="other"))
        return

    try:
        rc, _out, err, _timed_out = run_with_timeout([python, "-c", TRIAGE_IMPORTS], 120)
    except OSError as e:
        rc, err = 127, repr(e)
    if rc != 0:
        reason = next((ln.strip() for ln in reversed(err.splitlines()) if ln.strip()), f"rc={rc}")
        phase.add(
            Check(
                name="eth_status_read",
                status=SKIP,
                details=f"triage cannot run under {python}: {reason[:200]}. Install {TRIAGE_REQUIREMENTS} "
                f"(create_venv.sh does)",
                ip="other",
            )
        )
        return

    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(tt_metal)
    # Every chip is read, whatever the caller left visible to metal.
    env.pop("TT_METAL_VISIBLE_DEVICES", None)
    logs_dir.mkdir(parents=True, exist_ok=True)

    done: list[TriageRead] = []
    for i, ((name, _), cmd) in enumerate(zip(reads, commands)):
        if i:
            log(f"  waiting {interval_s:.0f}s before the second read, to see what the error counters do meanwhile")
            time.sleep(interval_s)
        done.append(_run_read(name, cmd, env, tt_metal, logs_dir))
        if not done[-1].links:
            break

    first = done[0]
    second = done[1] if len(done) > 1 and done[1].links else None
    interval = second.started - first.started if second else None
    data = {
        "python": python,
        "reads": [
            {
                "name": r.name,
                "command": shlex.join(r.command),
                "rc": r.rc,
                "timed_out": r.timed_out,
                "duration_s": r.duration_s,
                "eth_cores": len(r.links),
                "csv": str(r.csv_path),
                "log": str(r.log_path),
                "stderr_tail": r.stderr[-2000:],
            }
            for r in done
        ],
        "interval_s": interval,
    }

    if not first.links:
        timed = f" (timed out after {ETH_STATUS_READ_TIMEOUT_S}s)" if first.timed_out else ""
        phase.add(
            Check(
                name="eth_status_read",
                status=WARN,
                details=f"rc={first.rc} but triage listed no eth core{timed}: {_tail(first)} log={first.log_path}",
                data=data,
                ip="other",
            )
        )
        return

    chip_bdfs = parse_chip_bdfs(first.tables)
    data["chips_mapped_by"] = "device_info"
    if not {l.dev for l in first.links.values()} <= chip_bdfs.keys():
        chip_bdfs = sysfs_chip_bdfs(sys_root)
        data["chips_mapped_by"] = "/dev/tenstorrent order"
    data["chip_bdfs"] = {str(k): v for k, v in sorted(chip_bdfs.items())}

    if second is None:
        status = WARN
        details = f"second read listed no eth core, judged on the first alone: {_tail(done[-1])}"
    else:
        status = PASS
        details = f"2 reads {interval:.0f}s apart, {len(first.links)} and {len(second.links)} eth cores"
    details += f"; chips placed by {data['chips_mapped_by']}; logs={logs_dir}/eth_status_read_*.csv"
    phase.add(Check(name="eth_status_read", status=status, details=details, data=data, ip="other"))

    for c in judge_eth_status(first.links, second.links if second else None, interval, chip_bdfs, len(functions)):
        phase.add(c)
