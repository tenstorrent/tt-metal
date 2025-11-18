#!/usr/bin/env python3
"""Validate deprecated code removal policy.

Enforces that deprecated items (marked with [[deprecated]]) cannot be removed
before 30 days have elapsed since the deprecation was added.

Also supports enumerating all existing deprecations in the codebase and
determining which ones are ready to be removed.
"""

import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple, Optional

from common import find_cpp_sources


MIN_DEPRECATION_DAYS = 30

DEPRECATION_PATTERN = re.compile(r'\[\[deprecated(?:\(["\'].*?["\']\))?\]\]', re.IGNORECASE)


class DiffLine(NamedTuple):
    file_path: str
    old_file_line_number: int
    content: str
    last_modified_timestamp: Optional[int]

    @property
    def age_days(self) -> Optional[float]:
        if self.last_modified_timestamp is None:
            return None
        return (datetime.now().timestamp() - self.last_modified_timestamp) / 86400

    @property
    def contains_deprecation(self) -> bool:
        return bool(DEPRECATION_PATTERN.search(self.content))


def get_removed_deprecated_lines(files: list[str], base_ref: str = "origin/main") -> list[DiffLine]:
    """Get removed lines containing deprecation attributes from specified files in a single git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", base_ref, "HEAD", "--"] + files,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to run git diff")

    removed_lines = []
    current_file = None
    current_old_line = 0

    for line in result.stdout.split("\n"):
        if line.startswith("--- a/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            parts = line.split()
            old_range = parts[1][1:]
            if "," in old_range:
                current_old_line = int(old_range.split(",")[0])
            else:
                current_old_line = int(old_range)
        elif line.startswith("-") and not line.startswith("---"):
            if current_file:
                content = line[1:]
                # Only collect lines that contain deprecation attributes
                if DEPRECATION_PATTERN.search(content):
                    removed_lines.append((current_file, current_old_line, content))
                current_old_line += 1
        elif line.startswith(" "):
            current_old_line += 1

    return _enrich_with_blame(removed_lines, base_ref)


def _enrich_with_blame(removed_lines: list[tuple[str, int, str]], base_ref: str) -> list[DiffLine]:
    """Enrich removed lines with blame information, batching by file."""
    # Group lines by file (declarative transformation)
    file_to_lines = defaultdict(list)
    for file_path, line_num, content in removed_lines:
        file_to_lines[file_path].append((line_num, content))

    # Transform each file's lines into DiffLine objects (functional pipeline)
    def enrich_file_lines(file_path: str, lines: list[tuple[int, str]]) -> list[DiffLine]:
        contiguous_ranges = _compute_ranges(sorted(line_num for line_num, _ in lines))
        line_to_timestamp = _get_blame_for_line_ranges(file_path, contiguous_ranges, base_ref)

        return [
            DiffLine(
                file_path=file_path,
                old_file_line_number=line_num,
                content=content,
                last_modified_timestamp=line_to_timestamp.get(line_num),
            )
            for line_num, content in lines
        ]

    # Flatten results from all files (declarative)
    return [
        diff_line for file_path, lines in file_to_lines.items() for diff_line in enrich_file_lines(file_path, lines)
    ]


def _compute_ranges(line_numbers: list[int]) -> list[tuple[int, int]]:
    """Convert list of line numbers to contiguous ranges."""
    if not line_numbers:
        return []

    ranges = []
    start = line_numbers[0]
    end = line_numbers[0]

    for num in line_numbers[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append((start, end))
            start = end = num

    ranges.append((start, end))
    return ranges


def _parse_blame_line(blame_line: str) -> Optional[tuple[int, int]]:
    """Parse a single git blame line to extract (line_number, timestamp)."""
    if not blame_line.strip():
        return None

    parts = blame_line.split()
    if len(parts) < 4:
        return None

    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 10:
            timestamp = int(part)
            line_num_str = parts[i + 2].rstrip(")")
            if line_num_str.isdigit():
                return (int(line_num_str), timestamp)
    return None


def _get_blame_for_line_range(file_path: str, start: int, end: int, base_ref: str) -> dict[int, int]:
    """Get blame info for a single line range."""
    try:
        result = subprocess.run(
            ["git", "blame", "-t", "-L", f"{start},{end}", base_ref, "--", file_path],
            capture_output=True,
            text=True,
            check=True,
        )

        # Declarative: parse all lines, filter out None, convert to dict
        parsed_lines = (_parse_blame_line(line) for line in result.stdout.split("\n"))
        return dict(entry for entry in parsed_lines if entry is not None)

    except subprocess.CalledProcessError:
        return {}


def _get_blame_for_line_ranges(file_path: str, ranges: list[tuple[int, int]], base_ref: str) -> dict[int, int]:
    """Get blame info for multiple line ranges in a single file."""
    # Declarative: merge all blame data from each range
    timestamp_maps = (_get_blame_for_line_range(file_path, start, end, base_ref) for start, end in ranges)

    result = {}
    for timestamp_map in timestamp_maps:
        result.update(timestamp_map)
    return result


class ExistingDeprecation(NamedTuple):
    file_path: str
    line_number: int
    content: str
    last_modified_timestamp: Optional[int]

    @property
    def age_days(self) -> Optional[float]:
        if self.last_modified_timestamp is None:
            return None
        return (datetime.now().timestamp() - self.last_modified_timestamp) / 86400

    @property
    def is_ready_to_remove(self) -> bool:
        if self.age_days is None:
            return False
        return self.age_days >= MIN_DEPRECATION_DAYS


def find_existing_deprecations(files: list[str]) -> list[ExistingDeprecation]:
    """Find all existing deprecation attributes in the current HEAD."""
    deprecations = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, start=1):
                    if DEPRECATION_PATTERN.search(line):
                        deprecations.append((file_path, line_num, line.rstrip("\n")))
        except (OSError, IOError):
            continue

    return _enrich_existing_with_blame(deprecations)


def _enrich_existing_with_blame(deprecations: list[tuple[str, int, str]]) -> list[ExistingDeprecation]:
    """Enrich existing deprecation lines with blame information."""
    file_to_lines = defaultdict(list)
    for file_path, line_num, content in deprecations:
        file_to_lines[file_path].append((line_num, content))

    def enrich_file_lines(file_path: str, lines: list[tuple[int, str]]) -> list[ExistingDeprecation]:
        contiguous_ranges = _compute_ranges(sorted(line_num for line_num, _ in lines))
        line_to_timestamp = _get_blame_for_line_ranges(file_path, contiguous_ranges, "HEAD")

        return [
            ExistingDeprecation(
                file_path=file_path,
                line_number=line_num,
                content=content,
                last_modified_timestamp=line_to_timestamp.get(line_num),
            )
            for line_num, content in lines
        ]

    return [
        deprecation for file_path, lines in file_to_lines.items() for deprecation in enrich_file_lines(file_path, lines)
    ]


def enumerate_existing_deprecations(directory: str) -> int:
    """Enumerate all existing deprecations and check if they're ready to be removed."""
    source_files = find_cpp_sources(directory)
    existing_deprecations = find_existing_deprecations(source_files)

    if not existing_deprecations:
        print("No deprecations found in the codebase.")
        return 0

    print(f"\nFound {len(existing_deprecations)} existing deprecation(s):\n")

    ready_to_remove = []
    not_ready = []
    unknown_age = []

    for dep in existing_deprecations:
        if dep.age_days is None:
            unknown_age.append(dep)
        elif dep.is_ready_to_remove:
            ready_to_remove.append(dep)
        else:
            not_ready.append(dep)

    # Print ready to remove
    if ready_to_remove:
        print(f"READY TO REMOVE ({len(ready_to_remove)}):")
        print("-" * 80)
        for dep in ready_to_remove:
            added_date = datetime.fromtimestamp(dep.last_modified_timestamp).strftime("%Y-%m-%d")
            print(f"{dep.file_path}:{dep.line_number}")
            print(f"  Deprecated on: {added_date} (age: {dep.age_days:.1f} days)")
            print(f"  {dep.content.strip()}")
            print()

    # Print not ready
    if not_ready:
        print(f"NOT READY TO REMOVE ({len(not_ready)}):")
        print("-" * 80)
        for dep in not_ready:
            added_date = datetime.fromtimestamp(dep.last_modified_timestamp).strftime("%Y-%m-%d")
            days_remaining = MIN_DEPRECATION_DAYS - dep.age_days
            print(f"{dep.file_path}:{dep.line_number}")
            print(f"  Deprecated on: {added_date} (age: {dep.age_days:.1f} days, {days_remaining:.1f} days remaining)")
            print(f"  {dep.content.strip()}")
            print()

    # Print unknown age
    if unknown_age:
        print(f"UNKNOWN AGE ({len(unknown_age)}):")
        print("-" * 80)
        for dep in unknown_age:
            print(f"{dep.file_path}:{dep.line_number}")
            print(f"  {dep.content.strip()}")
            print()

    return 0


def validate_removed_deprecations(directory: str, base_ref: str = "origin/main") -> int:
    """Validate that removed deprecated items meet the minimum age requirement."""
    source_files = find_cpp_sources(directory)
    lines_with_deprecated_attr = get_removed_deprecated_lines(source_files, base_ref)

    if not lines_with_deprecated_attr:
        return 0

    print("\nFound removed deprecated items:")
    violations = []

    for line in lines_with_deprecated_attr:
        if line.age_days is None:
            print(f"{line.file_path}:{line.old_file_line_number}: {line.content.strip()} [age unknown]")
            continue

        can_remove = line.age_days >= MIN_DEPRECATION_DAYS
        status = "OK" if can_remove else f"FAIL (age: {line.age_days:.1f}d, min: {MIN_DEPRECATION_DAYS}d)"

        print(f"{line.file_path}:{line.old_file_line_number}: {line.content.strip()} [{status}]")

        if not can_remove:
            added_date = datetime.fromtimestamp(line.last_modified_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            violations.append(
                {
                    "file": line.file_path,
                    "line": line.old_file_line_number,
                    "content": line.content.strip(),
                    "age_days": line.age_days,
                    "added_date": added_date,
                }
            )

    if violations:
        print(f"\nDEPRECATION POLICY VIOLATION")
        print(f"Deprecated items must exist for at least {MIN_DEPRECATION_DAYS} days before removal.\n")

        for violation in violations:
            print(f"{violation['file']}:{violation['line']}")
            print(f"  Deprecated on: {violation['added_date']}")
            print(f"  Days remaining: {MIN_DEPRECATION_DAYS - violation['age_days']:.1f}\n")

        return 1

    return 0


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} <directory> [--enumerate]")
        print()
        print("Modes:")
        print("  Default: Validate that removed deprecated items meet the minimum age requirement")
        print("  --enumerate: List all existing deprecations and check if they're ready to be removed")
        return 1

    directory = sys.argv[1]

    if len(sys.argv) == 3 and sys.argv[2] == "--enumerate":
        return enumerate_existing_deprecations(directory)
    else:
        return validate_removed_deprecations(directory)


if __name__ == "__main__":
    sys.exit(main())
