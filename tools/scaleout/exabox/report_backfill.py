#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover leftover validation runs under an artifact tree for backfill.

Prefers a trailing wrapper summary JSON object, then wrapper-log fields. Does not grep MPI / GTest bodies.
Large files are scanned in bounded chunks (head for metadata, reverse tail for outcomes).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from analyze_host_health_results import DIAG_REPORT_NAME, parse_diag_report

_HEAD_BYTES = 65536
_CHUNK_BYTES = 65536
_CHUNK_OVERLAP = 2048
# Allow a host/timestamp prefix; keep end-of-line so MPI bodies are not grepped.
_ANALYSIS_RC = re.compile(r"Analysis exit code:\s*(-?\d+)\s*$", re.MULTILINE)
_HOSTS_LINE = re.compile(r"^HOSTS=(.*)$", re.MULTILINE)
_OUTPUT_DIR_LINE = re.compile(r"^OUTPUT_DIR=(.*)$", re.MULTILINE)
_FILENAME_TS = re.compile(r"(\d{8}T\d{6}Z)")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WRAPPER_HEADER = re.compile(
    r"^=== (Physical Validation|Fabric Tests|Dispatch Tests|Recover) - ",
    re.MULTILINE,
)
# Last terminal recover event wins. Do not treat "Recovery completed at …" as success.
# Host-prefixed wrapper lines: "[host][ts] Recovery attempt 1 of 1 failed (exit code 1)."
_RECOVERY_EVENT = re.compile(
    r"Recovery succeeded on attempt"
    r"|Recovery attempt .+? failed \(exit code (-?\d+)\)"
    r"|Recover completed successfully"
    r"|Recover failed(?:\s|$|\()"
)

# Wrapper logs only. Never match cluster_validation_iteration_*.log.
_LOG_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("physical", ("physical_validation-*.log",)),
    ("fabric", ("fabric_tests*.log",)),
    ("dispatch", ("dispatch_tests-*.log",)),
    ("recover", ("recover-*.log", "recover_*.log")),
)


@dataclass(frozen=True)
class Leftover:
    test_type: str
    hosts: str
    analyzer_code: int | None
    artifact_dir: str
    ts: str
    mtime: datetime
    source: Path
    incomplete: bool = False
    incomplete_reason: str = ""
    duration_s: float | None = None
    labels: dict[str, str] = field(default_factory=dict)


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def parse_window_date(value: str | None, flag: str) -> date | None:
    if value is None:
        return None
    if not _DATE_RE.match(value):
        raise ValueError(f"{flag}: expected YYYY-MM-DD, got {value!r}")
    return date.fromisoformat(value)


def mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def in_mtime_window(mtime: datetime, from_date: date | None, to_date: date | None) -> bool:
    day = mtime.astimezone(timezone.utc).date()
    if from_date is not None and day < from_date:
        return False
    if to_date is not None and day > to_date:
        return False
    return True


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_head(path: Path, nbytes: int = _HEAD_BYTES) -> str:
    with open(path, "rb") as handle:
        return handle.read(nbytes).decode("utf-8", errors="replace")


def _iter_tail_windows(path: Path) -> Iterator[str]:
    """Yield overlapping windows from EOF toward the start. Memory is one chunk."""
    size = path.stat().st_size
    if size <= 0:
        return
    with open(path, "rb") as handle:
        pos = size
        carry = b""
        while pos > 0:
            read_size = min(_CHUNK_BYTES, pos)
            pos -= read_size
            handle.seek(pos)
            chunk = handle.read(read_size) + carry
            yield chunk.decode("utf-8", errors="replace")
            carry = chunk[:_CHUNK_OVERLAP]
            if pos == 0:
                break


def _last_match_in_windows(path: Path, pattern: re.Pattern[str]) -> re.Match[str] | None:
    """Return the last (closest-to-EOF) match without loading the whole file."""
    size = path.stat().st_size
    if size <= _HEAD_BYTES * 2:
        matches = list(pattern.finditer(_read_text(path)))
        return matches[-1] if matches else None
    for window in _iter_tail_windows(path):
        matches = list(pattern.finditer(window))
        if matches:
            return matches[-1]
    return None


_WRAPPER_SENTINEL_KEYS = ("analysis_exit_code", "return_code", "output_dir", "checked_at")


def extract_trailing_json(text: str) -> dict[str, Any] | None:
    """Return the last JSON object in text, if it looks like a wrapper summary.

    Walk candidate ``{`` … last-``}`` spans through ``json.loads`` so braces
    inside strings do not throw off a manual depth count.
    """
    end = text.rfind("}")
    if end < 0:
        return None
    found: dict[str, Any] | None = None
    search_from = 0
    while search_from <= end:
        start = text.find("{", search_from, end)
        if start < 0:
            break
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        if isinstance(obj, dict) and any(k in obj for k in _WRAPPER_SENTINEL_KEYS):
            found = obj
        search_from = start + 1
    return found


def load_sidecar_json(log_path: Path) -> dict[str, Any] | None:
    sidecar = log_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        obj = json.loads(_read_text(sidecar))
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = None
    for match in pattern.finditer(text):
        pass
    if match is None:
        return None
    return match.group(1).strip()


def parse_compact_ts(value: str) -> str | None:
    """Turn YYYYMMDDTHHMMSSZ or RFC3339 into schema UTC ``...Z``."""
    text = value.strip()
    if text.endswith("Z") and "-" in text and ":" in text:
        try:
            dt = datetime.fromisoformat(text[:-1] + "+00:00")
        except ValueError:
            pass
        else:
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    squeezed = text.replace("-", "").replace(":", "")
    match = _FILENAME_TS.search(squeezed)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{raw[9:11]}:{raw[11:13]}:{raw[13:15]}Z"


def ts_from_filename(path: Path) -> str | None:
    match = _FILENAME_TS.search(path.name)
    if not match:
        return None
    return parse_compact_ts(match.group(1))


def _hosts_csv(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        hosts = [str(h).strip() for h in value if str(h).strip()]
        return ",".join(hosts) if hosts else None
    text = str(value).strip()
    return text or None


def _int_field(obj: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
    return None


def looks_like_wrapper(head: str, test_type: str) -> bool:
    match = _WRAPPER_HEADER.search(head)
    if match is None:
        return False
    kind = match.group(1)
    expected = {
        "physical": "Physical Validation",
        "fabric": "Fabric Tests",
        "dispatch": "Dispatch Tests",
        "recover": "Recover",
    }
    return kind == expected.get(test_type)


def parse_recover_code_from_text(text: str) -> int | None:
    """Return recover RC from the last terminal event, or None if incomplete."""
    match = None
    for match in _RECOVERY_EVENT.finditer(text):
        pass
    if match is None:
        return None
    body = match.group(0)
    if "succeeded" in body or "completed successfully" in body:
        return 0
    fail_rc = match.group(1)
    if fail_rc is not None:
        return int(fail_rc)
    return 1


def parse_recover_code(path: Path) -> int | None:
    size = path.stat().st_size
    if size <= _HEAD_BYTES * 2:
        return parse_recover_code_from_text(_read_text(path))
    match = _last_match_in_windows(path, _RECOVERY_EVENT)
    if match is None:
        return None
    return parse_recover_code_from_text(match.group(0))


def parse_analysis_exit_code(path: Path, head: str) -> int | None:
    footer = _last_match(_ANALYSIS_RC, head)
    if footer is not None and path.stat().st_size <= _HEAD_BYTES:
        return int(footer)
    match = _last_match_in_windows(path, _ANALYSIS_RC)
    if match is None:
        return None
    return int(match.group(1))


def leftover_from_log(test_type: str, log_path: Path, root: Path) -> Leftover | None:
    try:
        head = _read_head(log_path)
    except OSError as exc:
        _warn(f"skipping {log_path}: unreadable ({exc})")
        return None

    if not looks_like_wrapper(head, test_type):
        _warn(f"skipping {log_path}: not a {test_type} wrapper header")
        return None

    tail_for_json = head
    if log_path.stat().st_size > _HEAD_BYTES:
        with open(log_path, "rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - _HEAD_BYTES))
            tail_for_json = handle.read().decode("utf-8", errors="replace")
    payload = load_sidecar_json(log_path) or extract_trailing_json(tail_for_json)

    hosts = None
    analyzer_code: int | None = None
    artifact_dir: str | None = None
    ts: str | None = None

    if payload:
        hosts = _hosts_csv(payload.get("hosts") or payload.get("hosts_list"))
        if test_type == "recover":
            analyzer_code = _int_field(payload, "return_code", "analyzer_code")
            status = payload.get("status")
            if analyzer_code is None and status == "success":
                analyzer_code = 0
            elif analyzer_code is None and status in ("failed", "error"):
                analyzer_code = 1
        else:
            analyzer_code = _int_field(payload, "analysis_exit_code", "analyzer_code")
        for key in ("output_dir", "artifact_uri", "log_file"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                artifact_dir = value.strip()
                if key != "log_file":
                    break
        checked = payload.get("checked_at")
        if isinstance(checked, str):
            ts = parse_compact_ts(checked)

    if hosts is None:
        hosts = _hosts_csv(_last_match(_HOSTS_LINE, head))
    if analyzer_code is None and test_type != "recover":
        analyzer_code = parse_analysis_exit_code(log_path, head)
    if analyzer_code is None and test_type == "recover":
        analyzer_code = parse_recover_code(log_path)
    if artifact_dir is None:
        output_dir = _last_match(_OUTPUT_DIR_LINE, head)
        if output_dir:
            artifact_dir = output_dir
    if artifact_dir is None:
        artifact_dir = str(log_path)
    if ts is None:
        ts = ts_from_filename(log_path)
    if ts is None:
        ts = mtime_utc(log_path).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not hosts:
        _warn(f"skipping {log_path}: no hosts")
        return None

    incomplete = analyzer_code is None
    incomplete_reason = "missing_terminal_outcome" if incomplete else ""
    if incomplete:
        reason = (
            "incomplete wrapper (no recover outcome)"
            if test_type == "recover"
            else "incomplete wrapper (no analysis exit code)"
        )
        _warn(f"{log_path}: {reason}; emitting degraded record")

    artifact_path = Path(artifact_dir)
    if not artifact_path.is_absolute():
        artifact_path = (root / artifact_dir).resolve()
        artifact_dir = str(artifact_path)

    return Leftover(
        test_type=test_type,
        hosts=hosts,
        analyzer_code=analyzer_code,
        artifact_dir=artifact_dir,
        ts=ts,
        mtime=mtime_utc(log_path),
        source=log_path,
        incomplete=incomplete,
        incomplete_reason=incomplete_reason,
    )


def leftover_from_diag_report(path: Path, root: Path) -> Leftover | None:
    try:
        extract = parse_diag_report(path)
    except ValueError as exc:
        _warn(f"skipping {path}: {exc}")
        return None
    if extract.dry_run:
        _warn(f"skipping {path}: dry-run diag report")
        return None
    ts = extract.ts
    if ts is None:
        ts = mtime_utc(path).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact_dir = extract.artifact_dir
    artifact_path = Path(artifact_dir)
    if not artifact_path.is_absolute():
        artifact_dir = str((root / artifact_dir).resolve())
    if extract.incomplete:
        _warn(f"{path}: incomplete diag report (no overall_status); emitting degraded record")
    return Leftover(
        test_type="host",
        hosts=extract.hosts,
        analyzer_code=extract.analyzer_code,
        artifact_dir=artifact_dir,
        ts=ts,
        mtime=mtime_utc(path),
        source=path,
        incomplete=extract.incomplete,
        incomplete_reason=extract.incomplete_reason,
        duration_s=extract.duration_s,
        labels=dict(extract.labels),
    )


def leftover_key(leftover: Leftover) -> tuple[str, str, str, str, int | None]:
    return (
        leftover.test_type,
        leftover.hosts,
        leftover.artifact_dir,
        leftover.ts,
        leftover.analyzer_code,
    )


def _iter_log_dirs(root: Path, recursive: bool) -> list[Path]:
    if not recursive:
        logs_dir = root / "logs"
        return [logs_dir] if logs_dir.is_dir() else []
    found: list[Path] = []
    for path in sorted(root.rglob("logs")):
        if path.is_dir():
            found.append(path)
    return found


def _iter_wrapper_logs(logs_dir: Path, test_type: str, patterns: tuple[str, ...]) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for log_path in sorted(logs_dir.glob(pattern)):
            if not log_path.is_file():
                continue
            resolved = log_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield log_path


def _iter_diag_reports(root: Path, recursive: bool) -> Iterator[Path]:
    if recursive:
        for path in sorted(root.rglob(DIAG_REPORT_NAME)):
            if path.is_file():
                yield path
        return
    direct = root / DIAG_REPORT_NAME
    if direct.is_file():
        yield direct


def discover_leftovers(root: Path, *, recursive: bool = False) -> list[Leftover]:
    root = root.resolve()
    found: list[Leftover] = []
    seen_sources: set[Path] = set()

    for logs_dir in _iter_log_dirs(root, recursive):
        tree_root = logs_dir.parent
        for test_type, patterns in _LOG_SPECS:
            for log_path in _iter_wrapper_logs(logs_dir, test_type, patterns):
                resolved = log_path.resolve()
                if resolved in seen_sources:
                    continue
                leftover = leftover_from_log(test_type, log_path, tree_root)
                if leftover is None:
                    seen_sources.add(resolved)
                    continue
                seen_sources.add(resolved)
                found.append(leftover)

    for diag_path in _iter_diag_reports(root, recursive):
        resolved = diag_path.resolve()
        if resolved in seen_sources:
            continue
        leftover = leftover_from_diag_report(diag_path, root)
        seen_sources.add(resolved)
        if leftover is not None:
            found.append(leftover)

    return found


def filter_leftovers(
    leftovers: list[Leftover],
    *,
    from_date: date | None,
    to_date: date | None,
) -> list[Leftover]:
    return [item for item in leftovers if in_mtime_window(item.mtime, from_date, to_date)]
