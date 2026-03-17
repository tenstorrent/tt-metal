#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook: detect legacy device operation patterns in newly added files.

New operations under ttnn/ should use ProgramDescriptorFactoryConcept (just
create_descriptor). This hook flags newly added .hpp files that use the legacy
ProgramFactoryConcept (cached_program_t / CachedProgram) pattern.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

LEGACY_PATTERNS = [
    re.compile(r"\bcached_program_t\b"),
    re.compile(r"\bCachedProgram\s*<"),
]


def get_newly_added_files():
    """Return list of newly added files in the current git staged changes."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--diff-filter=A", "--name-only"],
        capture_output=True,
        text=True,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def check_file(filepath: str) -> list[str]:
    """Check a single file for legacy patterns. Returns list of violations."""
    try:
        content = Path(filepath).read_text()
    except OSError:
        return []

    if "create_descriptor" in content:
        return []

    violations = []
    for pattern in LEGACY_PATTERNS:
        for match in pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            violations.append(f"  {filepath}:{line_num}: found '{match.group()}'")
    return violations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-new-only",
        action="store_true",
        help="Only check newly added files (git staged, diff-filter=A)",
    )
    parser.add_argument("files", nargs="*", help="Files to check (from pre-commit)")
    args = parser.parse_args()

    if args.check_new_only:
        new_files = set(get_newly_added_files())
        candidates = [f for f in (args.files or []) if f in new_files]
    else:
        candidates = args.files or []

    candidates = [f for f in candidates if f.startswith("ttnn/") and f.endswith(".hpp")]

    all_violations = []
    for filepath in candidates:
        all_violations.extend(check_file(filepath))

    if all_violations:
        print("ERROR: Legacy device operation pattern detected in new files.")
        print("New operations must use ProgramDescriptorFactoryConcept (create_descriptor).")
        print("See docs/source/ttnn/ttnn/adding_new_ttnn_operation.rst for guidance.\n")
        print("Violations:")
        for v in all_violations:
            print(v)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
