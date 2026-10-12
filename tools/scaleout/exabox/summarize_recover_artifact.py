#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Missing cables and error categories from the logs of a recover run."""

from __future__ import annotations

import re
from pathlib import Path

from analyze_validation_results import PATTERNS, LogAnalysis, parse_missing_connections, unique_missing_connections

DISCOVERY_START = "Running Physical Discovery"
MAX_LISTED = 64
ATTEMPT_START = re.compile(r"Recovery attempt \d+ of \d+\s*$")
# Checked in order, first match per line wins, so specific causes come before their knock-on errors.
RECOVER_ERRORS = {
    "post_reset_failed": re.compile(r"POST_RESET failed for device"),
    "reset_failed": re.compile(r"Reset failed \| Exit code|Reset failed on one or more hosts"),
    "port_down_failed": re.compile(r"cross-host port down FAILED"),
    "version_check_failed": re.compile(
        r"version check failed|KMD version .* < required|could not read /sys/class/tenstorrent/\S*tt_fw_bundle_ver"
    ),
    "chips_missing": re.compile(
        r"Didn't find all \d+ chips|Failed to find all \d+ PCIe devices|Query mappings failed on device"
    ),
    "chip_init_failed": re.compile(r"Error when re-initializing chips|Chip initialization failed|UmdException"),
    "disk_full": re.compile(r"No space left on device"),
    "mpi_timeout": re.compile(r"MPIEXEC_TIMEOUT|job execution time limit"),
    "mpi_daemon_lost": re.compile(r"failure of the TCP network|daemon has unexpectedly failed"),
}
# recover.sh retries these within the attempt, so they only end a run when nothing else went wrong.
SOFT_ERRORS = {"port_down_failed"}
ANALYZER_ERRORS = {k: v for k, v in PATTERNS.items() if k not in ("healthy", "recovered_connections")}
ERROR_PATTERNS = {
    **RECOVER_ERRORS,
    **ANALYZER_ERRORS,
    "killed": re.compile(r"\bKilled\s+mpirun"),
    "mpi_abort": re.compile(r"MPI_ABORT was invoked|prterun has exited due to process rank"),
}


def _logs_newest_first(artifact_dir: str | None) -> list[Path]:
    if not artifact_dir:
        return []
    try:
        logs = [p for p in Path(artifact_dir).glob("*.log") if p.is_file()]
        return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def _read(log: Path) -> str | None:
    try:
        return log.read_text(errors="replace")
    except OSError:
        return None


def _format_endpoint(ep: tuple) -> str:
    host, tray, port_type, port_id = ep
    return f"{host} T{tray} {port_type}/{port_id}"


def _last_discovery(text: str) -> str | None:
    start = text.rfind(DISCOVERY_START)
    return None if start < 0 else text[start:]


def missing_cables(artifact_dir: str | None) -> list[str] | None:
    """Missing cables seen by the last discovery in the newest log, or None without a discovery.

    Retried attempts re-run discovery, so only the last one reflects how the run ended.
    """
    for log in _logs_newest_first(artifact_dir):
        text = _read(log)
        last = None if text is None else _last_discovery(text)
        if last is None:
            continue
        found = parse_missing_connections(last)
        ports = [
            c
            for c in unique_missing_connections([LogAnalysis(filepath=str(log), missing_connections=found)])
            if c[0] == "port"
        ]
        return [f"{_format_endpoint(a)} <-> {_format_endpoint(b)}" for _, a, b in ports]
    return None


def missing_cable_labels(artifact_dir: str | None) -> dict[str, str]:
    """``missing_cables`` count and, when non-zero, a newline-separated ``missing_cables_list``."""
    cables = missing_cables(artifact_dir)
    if cables is None:
        return {}
    labels = {"missing_cables": str(len(cables))}
    if cables:
        listed = cables[:MAX_LISTED]
        if len(cables) > MAX_LISTED:
            listed.append(f"... {len(cables) - MAX_LISTED} more")
        labels["missing_cables_list"] = "\n".join(listed)
    return labels


def recover_errors(artifact_dir: str | None) -> tuple[list[str], str | None] | None:
    """Error categories in the newest log and the one the run ended on, or None without a readable log.

    The categories are in order of first appearance. The run ended on the first error of the last
    attempt that had any: later lines in that attempt are usually its consequences (MPI abort, cleanup).
    Soft errors are skipped there unless they are all the attempt had.
    """
    logs = _logs_newest_first(artifact_dir)
    text = _read(logs[0]) if logs else None
    if text is None:
        return None
    seen: list[str] = []
    attempt_errors: list[str] = []
    last_failing: list[str] = []
    for line in text.splitlines():
        if ATTEMPT_START.search(line):
            attempt_errors = []
            continue
        key = next((k for k, pattern in ERROR_PATTERNS.items() if pattern.search(line)), None)
        if key is None:
            continue
        if key not in seen:
            seen.append(key)
        attempt_errors.append(key)
        last_failing = attempt_errors
    hard = [k for k in last_failing if k not in SOFT_ERRORS]
    return seen, (hard or last_failing or [None])[0]


def recover_error_labels(artifact_dir: str | None) -> dict[str, str]:
    """``errors`` (comma-separated categories) and ``final_error`` (what the run ended on) when any were seen."""
    found = recover_errors(artifact_dir)
    if not found or not found[0]:
        return {}
    seen, ended_on = found
    return {"errors": ",".join(seen), "final_error": ended_on}
