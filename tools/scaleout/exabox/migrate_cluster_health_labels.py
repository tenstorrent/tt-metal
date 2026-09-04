#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Reclassify stored cluster-health records using one canonical snapshot.

Dry-run is the default. ``--apply`` requires ``--backup-root`` and validates
every planned record before copying originals and atomically replacing files.
Unresolved or mixed-superpod records abort apply unless explicitly allowed.
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

HIERARCHY_KEYS = ("superpod", "ring", "quad")
MIGRATION_VERSION = "canonical-snapshot-v1"


def _aliases(host: dict[str, Any]) -> set[str]:
    values = {
        str(host.get("fqdn") or ""),
        str(host.get("bmc_hostname") or ""),
        str(host.get("slurm_hostname") or ""),
        str(host.get("short_id") or ""),
    }
    fqdn = str(host.get("fqdn") or "")
    if "." in fqdn:
        values.add(fqdn.split(".", 1)[0])
    return {value.strip().lower() for value in values if value.strip()}


def load_snapshot(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    hosts = snapshot.get("hosts") if isinstance(snapshot, dict) else None
    if not isinstance(hosts, list):
        raise ValueError(f"snapshot {path} is missing top-level hosts")
    index: dict[str, dict[str, Any]] = {}
    for record in hosts:
        if not isinstance(record, dict):
            raise ValueError(f"snapshot {path} contains a non-object host")
        for alias in _aliases(record):
            previous = index.get(alias)
            if previous is not None and previous.get("fqdn") != record.get("fqdn"):
                raise ValueError(f"snapshot alias {alias!r} is ambiguous")
            index[alias] = record
    return index, snapshot


def canonical_labels(hosts: list[Any], index: dict[str, dict[str, Any]]) -> tuple[dict[str, str] | None, str]:
    matched: list[dict[str, Any]] = []
    unknown: list[str] = []
    for raw in hosts:
        name = str(raw).strip()
        record = index.get(name.lower())
        if record is None:
            unknown.append(name)
        else:
            matched.append(record)
    if unknown:
        return None, "unknown hosts: " + ", ".join(unknown)
    if not matched:
        return None, "record has no hosts"

    superpods = {str(record.get("superpod") or "") for record in matched}
    if len(superpods) != 1 or "" in superpods:
        return None, "hosts span multiple superpods: " + ", ".join(sorted(superpods))
    quads = {str(record.get("quad") or "") for record in matched}
    rings = {str(record.get("ring") or "") for record in matched}
    result = {"superpod": next(iter(superpods))}
    if len(rings) == 1 and "" not in rings:
        result["ring"] = next(iter(rings))
    if len(quads) == 1 and "" not in quads:
        result["quad"] = next(iter(quads))
    return result, ""


def _hierarchy(labels: dict[str, Any]) -> dict[str, str]:
    return {key: str(labels.get(key) or "") for key in HIERARCHY_KEYS if str(labels.get(key) or "")}


def _iter_records(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"record root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*.json")
        if not path.name.startswith(".") and ".tmp." not in path.name and not path.name.endswith(".json.tmp")
    )


@dataclass
class Change:
    root: Path
    path: Path
    original: dict[str, Any]
    updated: dict[str, Any]
    before: dict[str, str]
    after: dict[str, str]


def plan_migration(
    roots: list[Path],
    index: dict[str, dict[str, Any]],
    snapshot: dict[str, Any],
) -> tuple[list[Change], list[dict[str, str]], int]:
    changes: list[Change] = []
    unresolved: list[dict[str, str]] = []
    scanned = 0
    snapshot_generated_at = str(snapshot.get("generated_at") or "")
    for root in roots:
        for path in _iter_records(root):
            scanned += 1
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                validate_record(record, file_written="record_id" in record)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                unresolved.append({"path": str(path), "reason": f"invalid record: {exc}"})
                continue
            canonical, reason = canonical_labels(record.get("hosts") or [], index)
            if canonical is None:
                unresolved.append({"path": str(path), "reason": reason})
                continue
            labels = dict(record.get("labels") or {})
            before = _hierarchy(labels)
            if before == canonical:
                continue
            for key in HIERARCHY_KEYS:
                labels.pop(key, None)
            labels.update(canonical)
            labels["label_migration"] = MIGRATION_VERSION
            labels["label_migrated_from"] = json.dumps(before, sort_keys=True, separators=(",", ":"))
            if snapshot_generated_at:
                labels["label_snapshot_generated_at"] = snapshot_generated_at
            updated = dict(record)
            updated["labels"] = labels
            validate_record(updated, file_written="record_id" in updated)
            changes.append(
                Change(
                    root=root,
                    path=path,
                    original=record,
                    updated=updated,
                    before=before,
                    after=canonical,
                )
            )
    return changes, unresolved, scanned


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
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Apply resolvable records even when other records are unresolved",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include every changed and unresolved path in JSON output",
    )
    args = parser.parse_args(argv)
    if args.apply and args.backup_root is None:
        parser.error("--apply requires --backup-root")

    try:
        index, snapshot = load_snapshot(args.snapshot)
        changes, unresolved, scanned = plan_migration(args.root, index, snapshot)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "snapshot": str(args.snapshot),
        "roots": [str(root) for root in args.root],
        "scanned": scanned,
        "changed": len(changes),
        "unchanged": scanned - len(changes) - len(unresolved),
        "unresolved": len(unresolved),
    }
    if args.details:
        summary["changes"] = [
            {
                "path": str(change.path),
                "before": change.before,
                "after": change.after,
            }
            for change in changes
        ]
        summary["unresolved_records"] = unresolved

    if args.apply and unresolved and not args.allow_unresolved:
        summary["applied"] = 0
        summary["error"] = "unresolved records present; rerun with --allow-unresolved"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2
    if args.apply:
        try:
            apply_changes(changes, args.backup_root)
        except (OSError, ValueError) as exc:
            print(f"Error applying migration: {exc}", file=sys.stderr)
            return 2
        summary["applied"] = len(changes)
        summary["backup_root"] = str(args.backup_root)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not unresolved or args.allow_unresolved else 2


if __name__ == "__main__":
    sys.exit(main())
