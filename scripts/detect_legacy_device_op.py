#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Detect classes that implement the legacy concept OldDeviceOperation.

Uses simple pattern matching (no clang) to find classes with all three non-static methods:
    - validate
    - compute_output_specs
    - create_program

Can check a single file or process a git diff.
"""

import argparse
import re
import subprocess
import sys


def get_added_hpp_files(git_base_ref="origin/main"):
    """
    Returns a list of .hpp files that are newly added or renamed in this PR.

    Only checks files with status "A" (added) or "R" (renamed/moved).
    Does NOT check "M" (modified) files - modifications to existing files are allowed.
    """
    cmd = ["git", "diff", "--name-status", git_base_ref, "HEAD"]
    out = subprocess.check_output(cmd, encoding="utf-8")

    added = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        status, path = parts

        # Only process Added (A) or Renamed (R) files, NOT Modified (M) files
        # This ensures we only block new files or moved files, not changes to existing files
        if not (status.startswith("A") or status.startswith("R")):
            continue

        # Handle rename: status is like "R100" and path might have two paths (old -> new)
        if status.startswith("R"):
            # For renames, take the new path (second one after the tab)
            path_parts = path.split(maxsplit=1)
            if len(path_parts) == 2:
                path = path_parts[1]  # New path after rename
            else:
                path = path_parts[0]

        # Only check .hpp files in ttnn/ directories
        if path.endswith(".hpp") and path.startswith("ttnn/"):
            added.append(path)

    return added


def check_method_is_non_static(filepath, method_name):
    """
    Check if a method exists and is non-static.
    Returns True if a non-static declaration is found.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return False

    # Pattern to match method declarations
    # Look for method name followed by ( or <
    method_pattern = re.compile(rf"\b{re.escape(method_name)}\s*[<(]")

    # Find all positions where the method appears
    for match in method_pattern.finditer(content):
        match_pos = match.start()

        # Find which line this match is on
        line_start = content.rfind("\n", 0, match_pos) + 1
        line_end = content.find("\n", match_pos)
        if line_end == -1:
            line_end = len(content)

        # Get context: up to 3 lines before
        context_start = content.rfind("\n", 0, line_start - 1)
        for _ in range(2):  # Go back up to 2 more newlines
            prev_newline = content.rfind("\n", 0, context_start)
            if prev_newline == -1:
                break
            context_start = prev_newline

        if context_start == -1:
            context_start = 0
        else:
            context_start += 1  # Skip the newline

        context = content[context_start:line_end]

        # Check if "static" appears before the method name in this context
        # Look for pattern: static ... method_name (allowing for whitespace, keywords like inline/const/virtual)
        static_pattern = re.compile(
            rf"static\s+(?:inline\s+)?(?:const\s+)?(?:virtual\s+)?(?:typename\s+)?.*?\b{re.escape(method_name)}\s*[<(]",
            re.MULTILINE | re.DOTALL,
        )

        if static_pattern.search(context):
            continue  # This is a static method, skip it

        # Found a non-static method
        return True

    return False


def check_file_for_legacy_class(filepath):
    """
    Check if a file contains a legacy device operation class.
    Returns True if all three methods are found as non-static.
    """
    has_validate = check_method_is_non_static(filepath, "validate")
    has_compute_output_specs = check_method_is_non_static(filepath, "compute_output_specs")
    has_create_program = check_method_is_non_static(filepath, "create_program")

    return has_validate and has_compute_output_specs and has_create_program


def is_file_newly_added(filepath, base_ref="HEAD"):
    """
    Check if a file is newly added (doesn't exist in base_ref).
    Returns True if file is new, False if it exists in base_ref (modified).
    """
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{base_ref}:{filepath}"],
            stderr=subprocess.DEVNULL,
        )
        return False  # File exists in base_ref, so it's modified
    except subprocess.CalledProcessError:
        return True  # File doesn't exist in base_ref, so it's new


def main():
    parser = argparse.ArgumentParser(description="Detect legacy device operation classes")
    parser.add_argument(
        "files", nargs="*", help="Path(s) to .hpp file(s) to check (if not provided, processes git diff)"
    )
    parser.add_argument(
        "--base-ref", default="origin/main", help="Base git reference for diff mode (default: origin/main)"
    )
    parser.add_argument(
        "--check-new-only",
        action="store_true",
        help="Only check files that are newly added (not in base_ref). Used by pre-commit.",
    )
    args = parser.parse_args()

    if args.files:
        # File mode (single or multiple files, e.g., from pre-commit)
        legacy_files = []
        for filepath in args.files:
            # If check-new-only is set, skip files that exist in base_ref (modified files)
            if args.check_new_only:
                if not is_file_newly_added(filepath, "HEAD"):
                    continue  # Skip modified files, only check newly added ones

            if check_file_for_legacy_class(filepath):
                legacy_files.append(filepath)

        if not legacy_files:
            return 0

        # Output helpful error message for pre-commit
        if args.check_new_only:
            print("❌ ERROR: Detected new classes following legacy concept OldDeviceOperation", file=sys.stderr)
            print(
                "   (class implements non-static member functions: validate, compute_output_specs, create_program)",
                file=sys.stderr,
            )
            print("", file=sys.stderr)
            print("Files with legacy classes:", file=sys.stderr)
            for filepath in legacy_files:
                print(f"  - {filepath}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Please follow the modern device operation pattern instead:", file=sys.stderr)
            print(
                "  - See documentation: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html",
                file=sys.stderr,
            )
        else:
            # Output detected legacy classes to stdout (for CI/scripting, when not in pre-commit mode)
            for filepath in legacy_files:
                print(f"{filepath}")

        return 1
    else:
        # Diff mode
        added_hpp_files = get_added_hpp_files(args.base_ref)
        if not added_hpp_files:
            return 0

        legacy_files = []
        for filepath in added_hpp_files:
            if check_file_for_legacy_class(filepath):
                legacy_files.append(filepath)

        if not legacy_files:
            return 0

        # Output detected legacy classes
        for filepath in legacy_files:
            print(f"{filepath}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
