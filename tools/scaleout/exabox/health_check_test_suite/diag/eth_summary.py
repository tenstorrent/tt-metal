# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The ETH link summary a tier with ``eth_summary`` prints at the end of a run.

The checks and the JSON report are for the dashboard and the ticket. This is for
the person who ran the tier: one screen that says whether the links are healthy
and, when they are not, which ones to look at. It is rendered from the report
alone, so it says nothing the report does not.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from .eth_status import CRC_WARN_PER_S, RETRAIN_WARN_TOTAL, UNCORR_WARN_PER_S, UNCORR_WARN_TOTAL
from .galaxy import EXPECTED_CHIP_COUNT, UBB_ETH_PORTS
from .model import FAIL, PASS, SKIP, WARN

SUMMARY_FILE = "eth_status_summary.txt"

TITLE = f"{'=' * 16} ETH LINK SUMMARY {'=' * 16}"
WIDTH = 100
MAX_LISTED = 8

STATUS_WORD = {PASS: "OK", WARN: "** WARNING **", FAIL: "** PROBLEM **", SKIP: "NOT CHECKED"}
SEVERITY = (FAIL, WARN, PASS, SKIP)

# The QSFP ports by what tt-metal cables them for on one Galaxy (QSFP_CABLINGS in eth_status.py).
QSFP_GROUPS = (((1, 2), "Y torus"), ((3, 4, 5, 6), "X torus"), (tuple(range(7, 15)), "Z ports"))

FW_FIELDS = (("fw_bundle_version", "FW bundle"), ("cm_fw", "CM FW"), ("eth_fw", "ETH FW"), ("gddr_fw", "GDDR FW"))

NOT_CHECKED = (
    "whether the cabling is the one this host is meant to have: only the people who cabled it know",
    "a QSFP port linked on no UBB: no cable, a cable to another host and dead cables look the same from here",
    "links to other hosts: check_eth_status lists on-host links only",
    "where each cable goes: the far end of a link is not read, so cables in the wrong ports can look healthy",
    "traffic: a fabric router can wedge on a link that still reads healthy",
)


def _worst(statuses) -> str:
    present = set(statuses)
    return next((s for s in SEVERITY if s in present), SKIP)


def _checks_by_name(report: dict) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for phase in report.get("phases", {}).values():
        for c in phase.get("checks", []):
            found.setdefault(c.get("name", ""), c)
    return found


def _one_line(check: dict, limit: int = 160) -> str:
    details = (check.get("details") or "").splitlines()
    line = f"{check.get('name')}: {details[0] if details else check.get('status')}"
    return line if len(line) <= limit else line[: limit - 3] + "..."


def collect_findings(report: dict) -> tuple[list[str], list[str]]:
    """(problems, warnings) from every phase that gates: each item a check lists
    in ``data["findings"]``, else one line for the check itself."""
    problems: list[str] = []
    warnings: list[str] = []
    for phase in report.get("phases", {}).values():
        if phase.get("gates", True) is False:
            continue
        for c in phase.get("checks", []):
            if c.get("status") not in (FAIL, WARN):
                continue
            items = (c.get("data") or {}).get("findings")
            if items:
                for f in items:
                    (problems if f.get("status") == FAIL else warnings).append(str(f.get("text", "")))
            else:
                (problems if c["status"] == FAIL else warnings).append(_one_line(c))
    return problems, warnings


def _bullets(items: list[str]) -> list[str]:
    out = [f"    - {t}" for t in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        out.append(f"    ... and {len(items) - MAX_LISTED} more")
    return out


def _qsfp_state(per_ubb: dict[str, int], lanes: int) -> str:
    full = sorted(u for u, k in per_ubb.items() if k == lanes)
    part = sorted(u for u, k in per_ubb.items() if 0 < k < lanes)
    none = sorted(u for u, k in per_ubb.items() if k == 0)
    if len(none) == len(per_ubb):
        return "not linked inside this host"
    if len(full) == len(per_ubb):
        return f"linked on all {len(per_ubb)} UBBs"
    bits = (
        f"linked on {', '.join(full)}" if full else "",
        f"in part on {', '.join(part)}" if part else "",
        f"not on {', '.join(none)}" if none else "",
    )
    return ", ".join(b for b in bits if b)


def _table(report: dict, checks: dict[str, dict], interval: float | None, links_read: bool) -> list[str]:
    rows: list[tuple[str, str, str, str]] = []

    def add(label: str, result: str, expected: str, status: str) -> None:
        rows.append((label, result, expected, STATUS_WORD.get(status, status)))

    gate, enum = checks.get("eth_status_pcie_gate"), checks.get("pcie_enum_count")
    gate_data = (gate or {}).get("data") or {}
    if "present" in gate_data:
        ok = gate_data["present"] - len(gate_data.get("dropped") or []) - len(gate_data.get("unverified") or [])
        statuses = [gate["status"]] + ([enum["status"]] if enum else [])
        add("chips on PCIe", f"{ok} of {gate_data['expected']} ok", str(gate_data["expected"]), _worst(statuses))
    elif enum:
        actual = (enum.get("data") or {}).get("actual", "?")
        add("chips on PCIe (tt-smi)", f"{actual} of {EXPECTED_CHIP_COUNT}", str(EXPECTED_CHIP_COUNT), enum["status"])

    def row(name: str, label: str, expected: str, result) -> None:
        c = checks.get(name)
        if c is not None:
            add(label, "-" if c["status"] == SKIP else result(c.get("data") or {}), expected, c["status"])

    if links_read:
        row("eth_status_links_down", "links down or unreadable", "0", lambda d: str(len(d.get("down", []))))
        row("eth_status_heartbeat", "eth cores stopped", "0", lambda d: str(len(d.get("stopped", []))))
        row("eth_status_port_state", "ports in an odd state", "0", lambda d: str(len(d.get("odd", []))))

        fec = checks.get("eth_status_fec_uncorrected")
        if fec is not None:
            d = fec.get("data") or {}
            if d.get("max_per_s") is not None:
                add(
                    "uncorrected FEC errors",
                    f"max {d['max_per_s']:,.1f}/s",
                    f"< {UNCORR_WARN_PER_S:g}/s per link",
                    fec["status"],
                )
            elif d.get("max_total") is not None:
                add(
                    "uncorrected FEC errors",
                    f"max {d['max_total']:,} total",
                    f"< {UNCORR_WARN_TOTAL:,} total",
                    fec["status"],
                )
            else:
                add("uncorrected FEC errors", "-", f"< {UNCORR_WARN_PER_S:g}/s per link", fec["status"])
        row(
            "eth_status_crc_errors",
            "CRC errors (bad frames)",
            f"< {CRC_WARN_PER_S:g}/s per link",
            lambda d: f"max {d['max_per_s']:,.1f}/s" if d.get("max_per_s") is not None else "-",
        )

        retrains = checks.get("eth_status_retrains")
        if retrains is not None and retrains["status"] == SKIP:
            add("retrains", "-", "0", SKIP)
        elif retrains is not None:
            d = retrains.get("data") or {}
            during, since = d.get("during") or [], d.get("since_reset") or []
            if interval:
                add("retrains during the check", f"{len(during)} link(s)", "0", FAIL if during else PASS)
            else:
                add("retrains during the check", "one read only", "0", SKIP)
            add(
                "retrains since last reset",
                f"{len(since)} link(s) at >= {RETRAIN_WARN_TOTAL}",
                f"< {RETRAIN_WARN_TOTAL} per link",
                WARN if since else PASS,
            )
        row("eth_status_err_stat", "ERR_STAT latched", "0", lambda d: f"{len(d.get('latched', []))} link(s)")
        row(
            "eth_status_builtin_links",
            "built-in links (boards)",
            "all",
            lambda d: f"{d.get('linked')} of {d.get('total')} linked",
        )
        row(
            "eth_status_cable_lanes",
            "cable ports linked in part",
            "0",
            lambda d: f"{len(d.get('partial', []))} port(s)",
        )
        row(
            "eth_status_cable_trays_alike",
            "cable ports, UBBs alike",
            "yes",
            lambda d: "yes" if not d.get("uneven") else f"no: {len(d['uneven'])} port(s)",
        )
        links = (checks.get("eth_status_links_down") or {}).get("data", {}).get("links")
        if links is not None:
            add("on-host links in total", str(links), "-", "not judged")

    fw = [
        checks[f"{field}_consistent"]
        for field, _ in FW_FIELDS
        if f"{field}_consistent" in checks and checks[f"{field}_consistent"]["status"] != SKIP
    ]
    if fw:
        mixed = [c["name"].removesuffix("_consistent") for c in fw if c["status"] != PASS]
        add(
            "firmware across chips",
            "same" if not mixed else "mixed: " + ", ".join(mixed),
            "same",
            _worst(c["status"] for c in fw),
        )

    covered = {"pcie_enum_count"} | {f"{field}_consistent" for field, _ in FW_FIELDS}
    rest = [c for c in report.get("phases", {}).get("snapshot", {}).get("checks", []) if c.get("name") not in covered]
    if rest:
        bad = [c for c in rest if c.get("status") in (FAIL, WARN)]
        fails = sum(c["status"] == FAIL for c in bad)
        result = "no fault" if not bad else f"{fails} fail, {len(bad) - fails} warn"
        add("other tt-smi snapshot checks", result, "no fault", _worst(c["status"] for c in bad) if bad else PASS)

    lines = [f"  {'CHECK':<30} {'RESULT':<24} {'EXPECTED':<20} STATUS", "  " + "-" * 86]
    lines += [f"  {label:<30} {result:<24} {expected:<20} {status}" for label, result, expected, status in rows]
    builtin = checks.get("eth_status_builtin_links")
    if links_read and builtin is not None and builtin["status"] == SKIP:
        reason = (builtin.get("details") or "").split("not judged: ", 1)[-1]
        lines.append(f"  Which links are there was NOT judged: {reason}.")
    return lines


def _cabling(cabling: dict) -> list[str]:
    d = cabling.get("data") or {}
    lanes = d.get("lanes_per_port") or {}
    lines = [
        "",
        "  CABLING FOUND (QSFP ports linked inside this host: described, not judged)",
        f"    built-in links (board traces, linking boards)   {d.get('builtin')} of {d.get('builtin_total')} linked",
    ]
    for ports, role in QSFP_GROUPS:
        states = [_qsfp_state(lanes.get(str(p), {}), len(UBB_ETH_PORTS["QSFP_DD", p])) for p in ports]
        i = 0
        while i < len(ports):
            j = i
            while j + 1 < len(ports) and states[j + 1] == states[i]:
                j += 1
            which = f"port  {ports[i]}" if i == j else f"ports {ports[i]}-{ports[j]}"
            lines.append(f"    QSFP {which:<11} ({role})   {states[i]}")
            i = j + 1
    lines.append(f"    => {d.get('description')}")
    unnamed = d.get("unnamed_channels") or []
    if unnamed:
        lines.append(f"    {len(unnamed)} link(s) on channels with no port name here: {', '.join(unnamed[:8])}")
    lines += [
        "    A port not linked inside this host has no cable, a cable to another host, or dead cables:",
        "    this check cannot tell which.",
    ]
    return lines


def _worst_links(fec: dict | None) -> list[str]:
    worst = ((fec or {}).get("data") or {}).get("worst_links") or []
    if not worst:
        return []

    def rate(v) -> str:
        return "-" if v is None else f"{v:,.1f}"

    lines = [
        "",
        "  WORST LINKS (by uncorrected FEC errors)",
        f"    {'link':<34} {'uncorr/s':>9} {'crc/s':>8} {'uncorr total':>13} {'retrains':>9}  ERR_STAT",
    ]
    for w in worst:
        lines.append(
            f"    {w.get('link', ''):<34} {rate(w.get('uncorr_per_s')):>9} {rate(w.get('crc_per_s')):>8} "
            f"{w.get('uncorr_total') or 0:>13,} {w.get('retrains') or 0:>9}  0x{w.get('err_stat') or 0:x}"
        )
    return lines


def render_eth_summary(report: dict, report_path: Path | None = None, summary_path: Path | None = None) -> str:
    checks = _checks_by_name(report)
    read = checks.get("eth_status_read") or {}
    read_data = read.get("data") or {}
    interval = read_data.get("interval_s")
    links_read = "eth_status_links_down" in checks

    started = (report.get("started_utc") or "")[:19]
    reads = f"2 reads {interval:.0f} s apart" if interval else "1 read" if links_read else "links not read"
    machine = f"Blackhole Galaxy, board {report.get('detected_board_rev') or 'rev not determined'}"
    if report.get("tt_smi_version"):
        machine += f", tt-smi {report['tt_smi_version']}"
    lines = [
        TITLE,
        f"  host {report.get('host')}   utc {started}Z   tier {report.get('tier')}   {reads}",
        f"  machine {machine}",
        "",
    ]

    problems, warnings = collect_findings(report)
    overall = report.get("overall_status")
    gate_failed = (checks.get("eth_status_pcie_gate") or {}).get("status") == FAIL
    links_failed = any(
        c.get("status") == FAIL
        for name, c in checks.items()
        if name.startswith("eth_status_") and name not in ("eth_status_pcie_gate", "eth_status_read")
    )
    if overall == FAIL:
        why = (
            "a chip dropped off PCIe, the links were not read"
            if gate_failed
            else "links broken or corrupting data"
            if links_failed
            else "no ETH link broken, but other checks failed"
            if links_read
            else "other checks failed, and the links were not read"
        )
        head = f"PROBLEM - {why} ({len(problems)} found)"
    elif not links_read:
        reason = _one_line(read)[len("eth_status_read: ") :] if read else "the ETH status phase did not run"
        reason = reason.removeprefix("links not read: ")
        head = f"NOT CHECKED - the links were not read: {reason}"
    elif overall == WARN:
        head = f"WARNING - nothing proven broken, worth a look ({len(warnings)} found)"
    else:
        head = "HEALTHY - no fault found" + (", no errors rising" if interval else "")
    lines.append(f"  VERDICT: {head}")
    lines += _bullets(problems or warnings)
    if problems and warnings:
        lines.append("    also worth a look:")
        lines += _bullets(warnings)
    if gate_failed:
        lines.append("    => Do NOT reset yet: report the chip(s) above first, a reset wipes the evidence")

    cabling = checks.get("eth_status_cabling")
    if cabling is not None and cabling["status"] == PASS:
        d = cabling.get("data") or {}
        text = (
            f"{d.get('links')} on-host links: {d.get('builtin')}/{d.get('builtin_total')} built-in + "
            f"{d.get('on_cables')} on cables. {d.get('description')}."
        )
        lines.append(textwrap.fill(text, WIDTH, initial_indent="  LINKS:   ", subsequent_indent=" " * 11))
    elif links_read:
        d = (checks.get("eth_status_links_down") or {}).get("data") or {}
        lines.append(f"  LINKS:   {d.get('links')} on-host links on {d.get('chips')} chips")

    lines.append("")
    lines += _table(report, checks, interval, links_read)
    if cabling is not None and cabling["status"] == PASS:
        lines += _cabling(cabling)
    lines += _worst_links(checks.get("eth_status_fec_uncorrected"))

    fw_bits = []
    for field, label in FW_FIELDS:
        c = checks.get(f"{field}_consistent")
        if c is not None and c["status"] == PASS:
            fw_bits.append(f"{label} {(c.get('data') or {}).get('value')}")
        elif c is not None and c["status"] == WARN:
            fw_bits.append(f"{label} mixed")
    if fw_bits:
        lines += ["", "  firmware  " + ", ".join(fw_bits)]

    lines += ["", "  NOT CHECKED HERE"] + [f"    - {t}" for t in NOT_CHECKED]

    lines.append("")
    if report_path is not None:
        lines.append(f"  report     {report_path}")
    csvs = [r["csv"] for r in read_data.get("reads", []) if r.get("csv")]
    if csvs:
        lines.append(f"  raw reads  {Path(csvs[0]).parent}/eth_status_read_*.csv")
    if summary_path is not None:
        lines.append(f"  summary    {summary_path}")
    lines.append("=" * len(TITLE))
    return "\n".join(lines) + "\n"
