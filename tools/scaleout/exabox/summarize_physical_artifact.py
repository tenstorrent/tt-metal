#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Infer physical analyzer results from an artifact directory or wrapper log.

Used by ``report_cluster_health.py`` so callers do not have to copy
``Success Rate`` / exit code off ``analyze_validation_results.py`` stdout.
Does not print the analyzer report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_validation_results import (
    analyze_log_file,
    calculate_metrics,
    get_cluster_info,
    validate_results,
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SUCCESS_RATE = re.compile(r"Success Rate:\s*([\d.]+)\s*%", re.IGNORECASE)
_SUCCESS_RATE_SENTENCE = re.compile(
    r"(?:healthy with|success rate)\s+([\d.]+)\s*%",
    re.IGNORECASE,
)
_WRAPPER_LOG_NAME = re.compile(
    r"^(physical_validation|fabric_tests|dispatch_tests|recover)[-_]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PhysicalSummary:
    """Quiet physical grade for one artifact tree."""

    pass_pct: float | None = None
    analyzer_code: int | None = None
    hosts: str = ""


def _strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def parse_success_rate(text: str) -> float | None:
    """Return a 0–100 success rate parsed from analyzer or wrapper text."""
    cleaned = _strip_ansi(text)
    match = _SUCCESS_RATE.search(cleaned)
    if match is None:
        match = _SUCCESS_RATE_SENTENCE.search(cleaned)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value < 0 or value > 100:
        return None
    return round(value, 2)


def _is_wrapper_log(name: str) -> bool:
    return bool(_WRAPPER_LOG_NAME.match(name))


def _iteration_logs(directory: Path) -> list[Path]:
    logs = sorted(path for path in directory.glob("*.log") if path.is_file())
    iterations = [path for path in logs if "cluster_validation_iteration_" in path.name]
    if iterations:
        return iterations
    return [path for path in logs if not _is_wrapper_log(path.name)]


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _wrapper_candidates(artifact: Path) -> list[Path]:
    candidates: list[Path] = []
    if artifact.is_file():
        candidates.append(artifact)
        search_dirs = [artifact.parent, artifact.parent / "logs", artifact.parent.parent / "logs"]
    else:
        search_dirs = [
            artifact,
            artifact / "logs",
            artifact.parent,
            artifact.parent / "logs",
        ]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        candidates.extend(sorted(directory.glob("physical_validation-*.log")))
        candidates.extend(sorted(directory.glob("physical_validation_*.log")))
    return _unique_paths(candidates)


def _grade_logs(log_files: list[Path]) -> PhysicalSummary:
    analyses = [analyze_log_file(str(path)) for path in log_files]
    analyses = [item for item in analyses if item.content]
    if not analyses:
        return PhysicalSummary()
    metrics = calculate_metrics(analyses)
    _message, analyzer_code = validate_results(analyses, metrics)
    info = get_cluster_info(analyses)
    hosts = ",".join(info["hosts"])
    rate = metrics.get("success_rate")
    pass_pct = None
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        pass_pct = round(float(rate), 2)
        if pass_pct < 0 or pass_pct > 100:
            pass_pct = None
    return PhysicalSummary(pass_pct=pass_pct, analyzer_code=analyzer_code, hosts=hosts)


def _summary_from_wrapper_text(text: str) -> PhysicalSummary:
    return PhysicalSummary(pass_pct=parse_success_rate(text))


def summarize_physical_artifact(artifact: str | Path) -> PhysicalSummary:
    """Grade physical logs under ``artifact`` without printing the analyzer report.

    Prefers in-process re-grade of iteration logs. If those are missing, parses
    ``Success Rate: N%`` from a sibling ``physical_validation-*.log``.
    """
    path = Path(artifact)
    if path.is_dir():
        logs = _iteration_logs(path)
        if logs:
            return _grade_logs(logs)
    elif path.is_file() and path.suffix == ".log" and not _is_wrapper_log(path.name):
        return _grade_logs([path])

    for wrapper in _wrapper_candidates(path):
        try:
            text = wrapper.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        summary = _summary_from_wrapper_text(text)
        if summary.pass_pct is not None:
            return summary
    return PhysicalSummary()


def as_pass_pct(value: Any) -> float | None:
    """Normalize a CLI / record ``pass_pct`` or return None if unusable."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    if number < 0 or number > 100:
        return None
    return round(number, 2)
