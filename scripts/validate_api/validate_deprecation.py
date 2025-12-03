#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate C++ deprecation policy: items must age 30+ days before removal.

Example: [[deprecated]] void oldFunc(); must exist 30+ days before deletion.
Exit codes: 0 (success), 1 (violations or errors)
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple

from common import find_cpp_sources, is_cpp_source, partition
import git_utils


class Config:
    """Deprecation policy configuration."""

    MIN_DEPRECATION_DAYS = 30
    SECONDS_PER_DAY = 86400


DEPRECATION_PATTERN = re.compile(r'\[\[deprecated(?:\(["\'].*?["\']\))?\]\]', re.IGNORECASE)


class DeprecatedItem(NamedTuple):
    """A deprecated code element with metadata."""

    file_path: str
    line_number: int
    content: str
    timestamp: Optional[int]  # Unix timestamp when added
    is_removed: bool  # True if removed in current branch

    @property
    def age_days(self) -> Optional[float]:
        """Days since deprecation was added."""
        if self.timestamp is None:
            return None
        return (datetime.now().timestamp() - self.timestamp) / Config.SECONDS_PER_DAY

    @property
    def can_remove(self) -> bool:
        """Check if old enough to remove."""
        return self.age_days is not None and self.age_days >= Config.MIN_DEPRECATION_DAYS


def find_removed_deprecations(directory: str, base_ref: str = "origin/main") -> List[DeprecatedItem]:
    # Get all C++ files that changed (including deleted ones)
    changed_files = git_utils.get_changed_file_paths(base_ref)

    # Only check C++ files within the directory scope (handles deleted files by path prefix)
    directory_path = str(Path(directory).resolve())
    changed_cpp_files = [
        f for f in changed_files if is_cpp_source(f) and Path(f).resolve().as_posix().startswith(directory_path)
    ]

    if not changed_cpp_files:
        return []

    # Get diffs for these files (git diff handles deleted files properly)
    try:
        # Process committed changes (base_ref to HEAD)
        committed_diff = git_utils.get_diff(changed_cpp_files, base_ref)
        committed_items = git_utils.parse_diff_for_removed_lines(committed_diff, DEPRECATION_PATTERN)
        committed_timestamps = git_utils.get_timestamps_for_items(committed_items, base_ref)

        # Process staged changes (HEAD to index)
        staged_diff = git_utils.get_staged_diff(changed_cpp_files)
        staged_items = git_utils.parse_diff_for_removed_lines(staged_diff, DEPRECATION_PATTERN)
        staged_timestamps = git_utils.get_timestamps_for_items(staged_items, "HEAD")
    except RuntimeError as e:
        print(f"Error: Failed to get diff: {e}", file=sys.stderr)
        sys.exit(1)

    # Combine results from both committed and staged changes
    results = []
    for file_path, line_num, content in committed_items:
        timestamp = committed_timestamps.get((file_path, line_num))
        results.append(DeprecatedItem(file_path, line_num, content, timestamp, is_removed=True))
    for file_path, line_num, content in staged_items:
        timestamp = staged_timestamps.get((file_path, line_num))
        results.append(DeprecatedItem(file_path, line_num, content, timestamp, is_removed=True))

    return results


def scan_file_for_deprecations(file_path: str) -> Iterator[Tuple[str, int, str]]:
    try:
        # errors='ignore' handles corrupted UTF-8 or binary files that may have been misidentified
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            yield from (
                (file_path, line_num, line.rstrip("\n"))
                for line_num, line in enumerate(f, 1)
                if DEPRECATION_PATTERN.search(line)
            )
    except IOError as e:
        print(f"Warning: Cannot read {file_path}: {e}", file=sys.stderr)
        return


def find_existing_deprecations(files: List[str]) -> List[DeprecatedItem]:
    items: List[Tuple[str, int, str]] = [item for file_path in files for item in scan_file_for_deprecations(file_path)]

    # HEAD is correct here: we want timestamps from current branch history for existing deprecations
    timestamps = git_utils.get_timestamps_for_items(items, "HEAD")
    return [
        DeprecatedItem(
            file_path,
            line_num,
            content,
            timestamps.get((file_path, line_num)),
            is_removed=False,
        )
        for file_path, line_num, content in items
    ]


def print_report(existing: List[DeprecatedItem], removed: List[DeprecatedItem]) -> int:
    print(f"\n{'='*80}")
    print(f"DEPRECATION REPORT")
    print(f"{'='*80}")

    # Report existing deprecations
    if existing:
        ready = [i for i in existing if i.can_remove]
        recent = [i for i in existing if i.age_days is not None and not i.can_remove]
        unknown = [i for i in existing if i.age_days is None]

        print(f"\n📊 Existing: {len(existing)} deprecations")
        for items, label in [
            (ready, f"ready to remove (≥{Config.MIN_DEPRECATION_DAYS} days)"),
            (recent, "not ready yet"),
            (unknown, "unknown age"),
        ]:
            if items:
                print(f"  • {len(items)} {label}")

        print(f"\n📋 Existing deprecations:")
        for item in sorted(existing, key=lambda x: (x.file_path, x.line_number)):
            age_str = f"{item.age_days:.1f} days" if item.age_days is not None else "unknown"
            status = "✓ removable" if item.can_remove else "⏳ too recent" if item.age_days is not None else "? unknown"
            print(f"  [{status}] {item.file_path}:{item.line_number} (age: {age_str})")

    # Report violations
    violations, valid = partition(lambda i: not i.can_remove, removed)

    if removed:
        print(f"\n📋 Removed deprecations:")
        for item in sorted(removed, key=lambda x: (x.file_path, x.line_number)):
            age_str = f"{item.age_days:.1f} days" if item.age_days is not None else "unknown"
            status = "✅ valid" if item.can_remove else "❌ violation"
            print(f"  [{status}] {item.file_path}:{item.line_number} (age: {age_str})")

    if violations:
        print(f"\n❌ POLICY VIOLATIONS: {len(violations)} items removed too early")
        print(f"   (must wait {Config.MIN_DEPRECATION_DAYS} days before removal)")
        for item in violations:
            if item.age_days is not None:
                wait = Config.MIN_DEPRECATION_DAYS - item.age_days
                print(
                    f"  • {item.file_path}:{item.line_number} - {item.age_days:.1f} days old, wait {wait:.1f} more days"
                )
            else:
                print(f"  • {item.file_path}:{item.line_number} - unknown age")

    # Summary
    print(f"\n{'='*80}")
    if violations:
        print(f"FAILED: {len(violations)} violation{'s' if len(violations) != 1 else ''}")
    else:
        print("PASSED: No violations")

    return 1 if violations else 0


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [base_ref]")
        return 1

    directory = sys.argv[1]
    base_ref = sys.argv[2] if len(sys.argv) > 2 else "origin/main"

    if not Path(directory).exists():
        print(f"Error: Directory '{directory}' does not exist", file=sys.stderr)
        return 1

    source_files = find_cpp_sources(directory)
    existing = find_existing_deprecations(source_files)
    removed = find_removed_deprecations(directory, base_ref)

    return print_report(existing, removed)


if __name__ == "__main__":
    sys.exit(main())
