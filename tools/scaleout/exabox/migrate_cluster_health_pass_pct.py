#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Add inferred ``pass_pct`` to stored physical cluster-health records.

Dry-run is the default. ``--apply`` requires ``--backup-root`` and atomically
replaces only physical records that were missing ``pass_pct`` and whose
artifact logs still yield a rate. Record IDs, timestamps, status, hosts, and
labels are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster_health_schema import validate_record
from summarize_physical_artifact import summarize_physical_artifact

SKIP_REASONS = ("no_rate",)


@dataclass
class Change:
    root: Path
    path: Path
    original: dict[str, Any]
    updated: dict[str, Any]
    pass_pct: float


def _iter_records(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"record root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*.json")
        if not path.name.startswith(".") and ".tmp." not in path.name and not path.name.endswith(".json.tmp")
    )


def infer_pass_pct(record: dict[str, Any]) -> float | None:
    artifact = record.get("artifact_uri")
    if not isinstance(artifact, str) or not artifact.strip():
        return None
    return summarize_physical_artifact(artifact.strip()).pass_pct


def plan_migration(roots: list[Path]) -> tuple[list[Change], list[dict[str, str]], int]:
    changes: list[Change] = []
    skipped: list[dict[str, str]] = []
    scanned = 0
    for root in roots:
        for path in _iter_records(root):
            scanned += 1
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                validate_record(record, file_written="record_id" in record)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                skipped.append({"path": str(path), "reason": f"invalid record: {exc}"})
                continue
            if record.get("test_type") != "physical":
                continue
            if "pass_pct" in record:
                continue
            rate = infer_pass_pct(record)
            if rate is None:
                skipped.append({"path": str(path), "reason": "no_rate"})
                continue
            updated = dict(record)
            updated["pass_pct"] = rate
            validate_record(updated, file_written="record_id" in updated)
            changes.append(
                Change(root=root, path=path, original=record, updated=updated, pass_pct=rate)
            )
    return changes, skipped, scanned


def apply_changes(changes: list[Change], backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=False)
    for change in changes:
        relative = change.path.relative_to(change.root)
        backup = backup_root / change.root.name / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(change.path, backup)

    for change in changes:
        tmp = change.path.with_name(f".{change.path.name}.{os.getpid()}.tmp")
        try:
            payload = json.dumps(change.updated, separators=(",", ":"), ensure_ascii=False) + "\n"
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            shutil.copystat(change.path, tmp)
            os.replace(tmp, change.path)
        finally:
            if tmp.exists():
                tmp.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include every changed and skipped path in JSON output",
    )
    args = parser.parse_args(argv)
    if args.apply and args.backup_root is None:
        parser.error("--apply requires --backup-root")

    try:
        changes, skipped, scanned = plan_migration(args.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    no_rate = sum(1 for item in skipped if item.get("reason") == "no_rate")
    invalid = len(skipped) - no_rate
    summary: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "roots": [str(root) for root in args.root],
        "scanned": scanned,
        "changed": len(changes),
        "already_had_pass_pct": scanned - len(changes) - len(skipped),
        "no_rate": no_rate,
        "invalid": invalid,
    }
    if args.details:
        summary["changes"] = [
            {"path": str(change.path), "pass_pct": change.pass_pct} for change in changes
        ]
        summary["skipped"] = skipped

    if args.apply:
        try:
            apply_changes(changes, args.backup_root)
        except (OSError, ValueError) as exc:
            print(f"Error applying migration: {exc}", file=sys.stderr)
            return 2
        summary["applied"] = len(changes)
        summary["backup_root"] = str(args.backup_root)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
