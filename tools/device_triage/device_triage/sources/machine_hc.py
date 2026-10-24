# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Machine HC runs (`hc_logs/<host>/<date>/<job>/`) and the BMC reboot collection,
which stores the same diag_report.json layout under `<host>/<date>/<HH:MM:SS>/`."""

import json
import re
from pathlib import Path

from .. import hosts
from ..config import BMC_REBOOT_ROOT, MAX_EVIDENCE_CHARS
from ..logscan import iso_ts, read_lines, scan_lines
from ..model import Event, Run, UnitResult
from ..signatures import RESET_CHECK_RE, classify_check

TRIAGE_ROW_RE = re.compile(r"^\s*(\d+)\s+(0000:[0-9a-f]{2}:00\.\d)\s+u(\d)c(\d)\b", re.I)
KLOG_SECTION = "-- PCIe, IOMMU and machine check faults"
STATUS_RANK = {"SKIP": 0, "PASS": 1, "WARN": 2, "FAIL": 3}


def _triage_table(run_dir: Path) -> tuple[dict[str, str], dict[int, str], list[str], int]:
    """BDF -> chip and ordinal -> chip from triage_host_side.txt, plus its kernel-fault lines."""
    bdf_map: dict[str, str] = {}
    ordinals: dict[int, str] = {}
    klog: list[str] = []
    klog_start = 0
    path = run_dir / "logs" / "triage_host_side.txt"
    if not path.exists():
        return bdf_map, ordinals, klog, klog_start
    in_klog = False
    for i, line in enumerate(read_lines(str(path))):
        m = TRIAGE_ROW_RE.match(line)
        if m:
            chip = hosts.chip_label(m.group(3), m.group(4))
            bdf_map[m.group(2).lower()] = chip
            ordinals[int(m.group(1))] = chip
        if line.startswith("-- "):
            in_klog = line.startswith(KLOG_SECTION)
            klog_start = i + 1 if in_klog else klog_start
            continue
        if in_klog:
            klog.append(line)
    return bdf_map, ordinals, klog, klog_start


def _check_log(run_dir: Path, check: dict) -> str | None:
    data = check.get("data") if isinstance(check.get("data"), dict) else {}
    candidates = [data.get("log_file"), str(run_dir / "logs" / f"{check.get('name')}.log")]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def _strongest(events: list[Event]) -> Event | None:
    """The evidence that best explains a failed test: first fail-level hit, else first warn."""
    for severity in ("fail", "warn"):
        for ev in sorted(events, key=lambda e: e.line or 0):
            if ev.severity == severity and ev.category not in ("unknown",):
                return ev
    return None


def parse_run(run_dir_str: str) -> UnitResult:
    run_dir = Path(run_dir_str)
    res = UnitResult(unit=run_dir_str)
    report = json.loads((run_dir / "diag_report.json").read_text(errors="replace"))
    is_bmc = run_dir_str.startswith(str(BMC_REBOOT_ROOT))
    source = "bmc_reboot" if is_bmc else "machine_hc"
    host = hosts.normalize_host(report.get("host")) or run_dir.parts[-3]
    run_id = f"{'bmc' if is_bmc else 'mhc'}:{host}/{run_dir.parts[-2]}/{run_dir.parts[-1]}"
    ts_start, ts_end = iso_ts(report.get("started_utc")), iso_ts(report.get("ended_utc"))
    report_path = str(run_dir / "diag_report.json")

    phases = report.get("phases") or {}
    if isinstance(phases, list):
        phases = {ph.get("name", str(i)): ph for i, ph in enumerate(phases)}
    checks = [
        (pn, c)
        for pn, ph in phases.items()
        if isinstance(ph, dict)
        for c in ph.get("checks") or []
        if isinstance(c, dict)
    ]

    bdf_map, ordinals, klog, klog_start = _triage_table(run_dir)
    for _, c in checks:
        for bdf, chip in hosts.bdf_chip_pairs(str(c.get("details") or "")):
            bdf_map.setdefault(bdf, chip)
    chip_ordinal = {chip: o for o, chip in ordinals.items()}
    res.chip_map = [(host, bdf, chip, chip_ordinal.get(chip)) for bdf, chip in bdf_map.items()]

    worst: dict[str, str] = {}
    for _, c in checks:
        name, status = c.get("name"), c.get("status")
        if STATUS_RANK.get(status, 0) > STATUS_RANK.get(worst.get(name), -1):
            worst[name] = status

    scanned: set[str] = set()
    failing, warning = [], []
    for phase, c in checks:
        name = c.get("name", "")
        for f in classify_check(phase, c, bdf_map, worst):
            ev = Event(
                host=host,
                ts=ts_start,
                source=source,
                origin=f.origin,
                category=f.category,
                signal=f.signal,
                severity=f.severity,
                detail=f.detail[:MAX_EVIDENCE_CHARS],
                chips=",".join(sorted(f.chips)),
                scope=hosts.scope_of(f.chips),
                run_id=run_id,
                path=report_path,
            )
            is_test = phase == "tests"
            if f.severity != "info" and (is_test or RESET_CHECK_RE.match(name)):
                log = _check_log(run_dir, c)
                if log and log not in scanned:
                    scanned.add(log)
                    log_events = scan_lines(
                        read_lines(log),
                        default_host=host,
                        base_ts=ts_start,
                        source=source,
                        origin=f"log:{name}",
                        path=log,
                        run_id=run_id,
                        bdf_map=bdf_map,
                        ordinals=ordinals,
                    )
                    res.events.extend(log_events)
                    best = _strongest(log_events) if is_test else None
                    if best:
                        ev.category, ev.signal = best.category, f"{name}:{best.signal}"
                        ev.chips = ev.chips or best.chips
            res.events.append(ev)
            if f.severity == "fail":
                failing.append(name)
            elif f.severity == "warn":
                warning.append(name)

    if klog and worst.get("triage_hostside_kernel_log") in ("WARN", "FAIL"):
        klog_events = scan_lines(
            klog,
            default_host=host,
            base_ts=ts_start,
            source=source,
            origin="log:kernel",
            path=str(run_dir / "logs" / "triage_host_side.txt"),
            run_id=run_id,
            bdf_map=bdf_map,
            ordinals=ordinals,
        )
        for ev in klog_events:
            ev.line = (ev.line or 0) + klog_start
        res.events.extend(klog_events)

    reason = ""
    if is_bmc and (run_dir / "reason.txt").is_file():
        reason = (run_dir / "reason.txt").read_text(errors="replace").strip()[:MAX_EVIDENCE_CHARS]
        res.events.append(
            Event(
                host=host,
                ts=ts_start,
                source=source,
                origin="bmc_reboot",
                category="reset",
                signal="bmc_reboot",
                severity="info",
                detail=reason or "BMC reboot",
                run_id=run_id,
                path=str(run_dir / "reason.txt"),
            )
        )

    effective = "FAIL" if failing else "WARN" if warning else "PASS"
    summary = ", ".join(dict.fromkeys(failing)) or ", ".join(dict.fromkeys(warning)) or "clean"
    res.runs.append(
        Run(
            run_id=run_id,
            source=source,
            kind=str(report.get("tier") or ""),
            hosts=[host],
            ts_start=ts_start,
            ts_end=ts_end,
            status=effective,
            summary=summary[:MAX_EVIDENCE_CHARS],
            path=run_dir_str,
            extra={
                "overall_status": report.get("overall_status"),
                "tt_smi": report.get("tt_smi_version"),
                "board_rev": report.get("detected_board_rev"),
                "tool_version": report.get("tool_version"),
                "post_reboot": "_post-reboot-" in run_dir.name,
                "reason": reason,
            },
        )
    )
    return res
