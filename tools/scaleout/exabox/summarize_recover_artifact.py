#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Missing port/cable connections from the last physical discovery of a recover run."""

from __future__ import annotations

from pathlib import Path

from analyze_validation_results import LogAnalysis, parse_missing_connections, unique_missing_connections

DISCOVERY_START = "Running Physical Discovery"
MAX_LISTED = 64


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
    if not artifact_dir:
        return None
    try:
        logs = sorted((p for p in Path(artifact_dir).glob("*.log") if p.is_file()), key=lambda p: p.stat().st_mtime)
    except OSError:
        return None
    for log in reversed(logs):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        last = _last_discovery(text)
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
