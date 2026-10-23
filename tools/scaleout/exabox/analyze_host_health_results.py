#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Map an existing diag_report.json to a host analyzer code (does not run tests).

Sibling to analyze_fabric_results.py / analyze_validation_results.py.
Reads overall_status from the diag suite artifact; does not import diag_runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# diag overall_status → host analyzer_code (report_adapters.host).
_STATUS_TO_CODE = {"PASS": 0, "FAIL": 1, "WARN": 2}

DIAG_REPORT_NAME = "diag_report.json"


@dataclass(frozen=True)
class HostHealthExtract:
    hosts: str
    analyzer_code: int | None
    ts: str | None
    duration_s: float | None
    labels: dict[str, str] = field(default_factory=dict)
    artifact_dir: str = ""
    dry_run: bool = False
    incomplete: bool = False
    incomplete_reason: str = ""


def _format_ts_utc(value: str) -> str:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("ts: must include a UTC offset")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_diag_report_path(json_path: str | None, artifact_dir: str | None) -> Path:
    if json_path:
        path = Path(json_path)
        if not path.is_file():
            raise ValueError(f"json: not a file: {path}")
        return path
    if artifact_dir:
        path = Path(artifact_dir) / DIAG_REPORT_NAME
        if not path.is_file():
            raise ValueError(f"artifact-dir: missing {DIAG_REPORT_NAME} under {artifact_dir}")
        return path
    raise ValueError("json or artifact-dir is required")


def parse_diag_report(
    path: Path,
    *,
    artifact_dir: str | None = None,
) -> HostHealthExtract:
    """Parse one diag_report.json. Raises ValueError if the file is unusable."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")

    host = obj.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError(f"{path}: missing host")
    hosts = host.strip()

    dry_run = bool(obj.get("dry_run"))

    labels: dict[str, str] = {}
    tier = obj.get("tier")
    if isinstance(tier, str) and tier.strip():
        labels["tier"] = tier.strip()
    board_rev = obj.get("detected_board_rev")
    if isinstance(board_rev, str) and board_rev.strip():
        labels["board_rev"] = board_rev.strip()

    analyzer_code: int | None = None
    incomplete = False
    incomplete_reason = ""
    overall = obj.get("overall_status")
    if isinstance(overall, str) and overall in _STATUS_TO_CODE:
        analyzer_code = _STATUS_TO_CODE[overall]
    else:
        incomplete = True
        incomplete_reason = "missing_terminal_outcome"

    ts: str | None = None
    for key in ("ended_utc", "started_utc"):
        raw = obj.get(key)
        if isinstance(raw, str) and raw.strip():
            try:
                ts = _format_ts_utc(raw)
                break
            except ValueError:
                continue

    duration_s: float | None = None
    raw_dur = obj.get("total_duration_s")
    if isinstance(raw_dur, (int, float)) and not isinstance(raw_dur, bool) and raw_dur >= 0:
        duration_s = float(raw_dur)

    resolved_artifact = artifact_dir or str(path.parent.resolve())

    return HostHealthExtract(
        hosts=hosts,
        analyzer_code=analyzer_code,
        ts=ts,
        duration_s=duration_s,
        labels=labels,
        artifact_dir=resolved_artifact,
        dry_run=dry_run,
        incomplete=incomplete,
        incomplete_reason=incomplete_reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze an existing diag_report.json (does not run the diag suite).")
    parser.add_argument("--json", dest="json_path", help=f"Path to {DIAG_REPORT_NAME}")
    parser.add_argument(
        "--artifact-dir",
        dest="artifact_dir",
        help=f"Directory containing {DIAG_REPORT_NAME}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path = resolve_diag_report_path(args.json_path, args.artifact_dir)
        extract = parse_diag_report(path, artifact_dir=args.artifact_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if extract.dry_run:
        print("Error: refusing dry-run diag report", file=sys.stderr)
        return 1
    print(f"host={extract.hosts}")
    if extract.ts:
        print(f"ts={extract.ts}")
    if extract.duration_s is not None:
        print(f"duration_s={extract.duration_s}")
    for key, value in extract.labels.items():
        print(f"{key}={value}")
    code = extract.analyzer_code if extract.analyzer_code is not None else 1
    print(f"Analysis exit code: {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
