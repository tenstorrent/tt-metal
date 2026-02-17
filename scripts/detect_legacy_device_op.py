#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Detect classes that implement legacy device operation patterns.

Checks for two legacy patterns:

1. OldDeviceOperation concept: classes with non-static methods validate,
   compute_output_specs, create_program, or create_program_at.

2. ProgramFactoryConcept (CachedProgram pattern): factories that define
   cached_program_t / shared_variables_t instead of using the modern
   ProgramDescriptorFactoryConcept (create_descriptor).

Checks specified .hpp files. Used by pre-commit hooks to detect legacy patterns in newly added files.
"""

import argparse
import os
import re
import subprocess
import sys

# Compile regex patterns once at module level for the methods we check
_METHOD_NAMES = ["validate", "compute_output_specs", "create_program", "create_program_at"]
_METHOD_PATTERNS = {name: re.compile(rf"\b{re.escape(name)}\s*[<(]") for name in _METHOD_NAMES}

# Patterns that indicate the old ProgramFactoryConcept (CachedProgram pattern)
_CACHED_PROGRAM_PATTERN = re.compile(r"\bcached_program_t\b")
_SHARED_VARIABLES_PATTERN = re.compile(r"\bshared_variables_t\b")
# Pattern that indicates the new ProgramDescriptorFactoryConcept
_CREATE_DESCRIPTOR_PATTERN = re.compile(r"\bcreate_descriptor\b")


def check_has_non_static_method(filepath, method_name):
    """
    Check if a method exists and is non-static.
    Returns True if a non-static declaration is found.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return False

    # Get pre-compiled pattern for the known method
    method_pattern = _METHOD_PATTERNS[method_name]

    # Find all positions where the method appears
    for match in method_pattern.finditer(content):
        match_pos = match.start()

        # Look backwards for "static" keyword (within reasonable distance, ~200 chars)
        lookback_start = max(0, match_pos - 200)
        context = content[lookback_start:match_pos]

        # Check if "static" appears before the method name
        if re.search(r"\bstatic\s+", context):
            continue  # This is a static method, skip it

        # Found a non-static method
        return True

    return False


def check_file_for_legacy_class(filepath):
    """
    Check if a file contains a legacy device operation class.
    Returns True if any of the methods (validate, compute_output_specs, create_program, or create_program_at)
    is found as non-static.
    """
    # Check if any of the methods exists as non-static
    for method_name in _METHOD_NAMES:
        if check_has_non_static_method(filepath, method_name):
            return True
    return False


def check_file_for_cached_program_factory(filepath):
    """
    Check if a file uses the legacy ProgramFactoryConcept (CachedProgram pattern)
    without also providing create_descriptor.

    Returns True if the file defines cached_program_t or shared_variables_t
    but does NOT define create_descriptor.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return False

    has_cached_program = bool(_CACHED_PROGRAM_PATTERN.search(content))
    has_shared_variables = bool(_SHARED_VARIABLES_PATTERN.search(content))
    has_create_descriptor = bool(_CREATE_DESCRIPTOR_PATTERN.search(content))

    # Flag if the file uses the old pattern without the new one
    return (has_cached_program or has_shared_variables) and not has_create_descriptor


def _file_exists_in_git(filepath, ref):
    """Check if a file exists in a git ref."""
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{ref}:{filepath}"],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _auto_detect_from_ref(to_ref):
    """Auto-detect from_ref by finding merge base with common branches."""
    for base_branch in ["origin/main", "main", "origin/master", "master"]:
        try:
            subprocess.check_output(
                ["git", "rev-parse", "--verify", base_branch],
                stderr=subprocess.DEVNULL,
            )
            merge_base = subprocess.check_output(
                ["git", "merge-base", to_ref, base_branch],
                stderr=subprocess.DEVNULL,
                encoding="utf-8",
            ).strip()
            if merge_base:
                return merge_base
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # Fallback: try HEAD^ for local pre-commit (before commit)
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", "HEAD^"],
            stderr=subprocess.DEVNULL,
        )
        return "HEAD^"
    except subprocess.CalledProcessError:
        pass

    # Last resort: use HEAD (but this won't work correctly in CI)
    return "HEAD"


def get_base_ref_for_precommit(from_ref=None, to_ref=None):
    """
    Determine base refs when running in pre-commit context.
    In CI, pre-commit uses --from-ref and --to-ref, which we accept as arguments.
    Falls back to auto-detection if not provided.

    Returns tuple (from_ref, to_ref).
    """
    # Check environment variables first (could be set by CI)
    from_ref = from_ref or os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = to_ref or os.environ.get("PRE_COMMIT_TO_REF") or "HEAD"

    # Auto-detect from_ref if not provided
    if not from_ref:
        from_ref = _auto_detect_from_ref(to_ref)

    return from_ref, to_ref


def is_file_newly_added(filepath, from_ref=None, to_ref=None):
    """
    Check if a file is newly added (doesn't exist in from_ref but exists in to_ref).
    Returns True if file is new, False if it exists in from_ref (modified).
    If refs are None, tries to auto-detect them.
    """
    from_ref, to_ref = get_base_ref_for_precommit(from_ref, to_ref)

    # File is newly added if it doesn't exist in from_ref but exists in to_ref
    exists_in_from = _file_exists_in_git(filepath, from_ref)
    exists_in_to = _file_exists_in_git(filepath, to_ref) or os.path.exists(filepath)

    return not exists_in_from and exists_in_to


def main():
    parser = argparse.ArgumentParser(description="Detect legacy device operation classes")
    parser.add_argument("files", nargs="+", help="Path(s) to .hpp file(s) to check")
    parser.add_argument(
        "--check-new-only",
        action="store_true",
        help="Only check files that are newly added (not in base_ref). Used by pre-commit.",
    )
    parser.add_argument(
        "--from-ref",
        help="Base ref to compare against (from-ref in pre-commit context). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--to-ref",
        help="Target ref to compare to (to-ref in pre-commit context). Defaults to HEAD.",
        default="HEAD",
    )
    args = parser.parse_args()

    legacy_files = []
    cached_program_files = []
    for filepath in args.files:
        # If check-new-only is set, skip files that exist in from_ref (modified files)
        if args.check_new_only:
            # Use provided from-ref and to-ref, or auto-detect
            if not is_file_newly_added(filepath, args.from_ref, args.to_ref):
                continue  # Skip modified files, only check newly added ones

        if check_file_for_legacy_class(filepath):
            legacy_files.append(filepath)
        elif check_file_for_cached_program_factory(filepath):
            cached_program_files.append(filepath)

    failures = False

    if legacy_files:
        failures = True
        print("❌ ERROR: Detected new classes following legacy concept OldDeviceOperation", file=sys.stderr)
        print(
            "   (class implements non-static member function(s): validate, compute_output_specs, create_program, or create_program_at)",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("Files with legacy classes:", file=sys.stderr)
        for filepath in legacy_files:
            print(f"  - {filepath}", file=sys.stderr)
        print("", file=sys.stderr)

    if cached_program_files:
        failures = True
        print(
            "❌ ERROR: Detected new program factories using the deprecated CachedProgram pattern (ProgramFactoryConcept)",
            file=sys.stderr,
        )
        print(
            "   (file defines cached_program_t / shared_variables_t without create_descriptor)",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("Files with deprecated CachedProgram factories:", file=sys.stderr)
        for filepath in cached_program_files:
            print(f"  - {filepath}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "   New operations must use ProgramDescriptorFactoryConcept (create_descriptor).",
            file=sys.stderr,
        )
        print(
            "   See migration recipe: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/descriptor_migration_recipe.html",
            file=sys.stderr,
        )
        print("", file=sys.stderr)

    if failures:
        print("Please follow the modern device operation pattern instead:", file=sys.stderr)
        print(
            "  - See documentation: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
