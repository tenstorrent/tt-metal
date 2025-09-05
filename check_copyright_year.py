#!/usr/bin/env python3
"""
Copyright Year Validation Script

This script validates that changed files with copyright headers contain the current year.
It should be run after the standard copyright check to ensure new/modified files have current years.
Use --all-files flag to check all files in the repository.
"""

import os
import re
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def get_current_year():
    """Get the current year."""
    return datetime.now().year


def get_changed_files(root_dir, extensions=None):
    """Get files that have been changed in the current commit/PR."""
    if extensions is None:
        extensions = [".cpp", ".cc", ".h", ".hpp", ".py", ".c", ".cxx"]

    try:
        # Get files changed in the current commit
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=AM", "HEAD~1"], capture_output=True, text=True, cwd=root_dir
        )

        if result.returncode == 0:
            changed_files = result.stdout.strip().split("\n")
            changed_files = [f for f in changed_files if f.strip()]

            # Filter by extensions and check if files exist
            filtered_files = []
            for file_path in changed_files:
                if any(file_path.endswith(ext) for ext in extensions):
                    full_path = Path(root_dir) / file_path
                    if full_path.exists():
                        filtered_files.append(full_path)

            return filtered_files
        else:
            # If HEAD~1 doesn't exist (first commit), get all tracked files
            result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=root_dir)
            if result.returncode == 0:
                all_files = result.stdout.strip().split("\n")
                all_files = [f for f in all_files if f.strip()]

                # Filter by extensions
                filtered_files = []
                for file_path in all_files:
                    if any(file_path.endswith(ext) for ext in extensions):
                        full_path = Path(root_dir) / file_path
                        if full_path.exists():
                            filtered_files.append(full_path)

                return filtered_files

    except subprocess.CalledProcessError:
        # If git commands fail, fall back to checking all files
        print("Warning: Could not determine changed files, checking all files")
        return None

    return []


def find_files_with_copyright(root_dir, extensions=None, exclude_dirs=None):
    """Find files that contain copyright headers."""
    if extensions is None:
        extensions = [".cpp", ".cc", ".h", ".hpp", ".py", ".c", ".cxx"]

    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", "build", "third_party", ".github", "python_env", "env", "venv"]

    files_with_copyright = []

    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = Path(root) / file
                # Skip files in python environments or virtual environments
                if "python_env" in str(file_path) or "env/" in str(file_path) or "venv/" in str(file_path):
                    continue
                if has_copyright_header(file_path):
                    files_with_copyright.append(file_path)

    return files_with_copyright


def has_copyright_header(file_path):
    """Check if file has a copyright header."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read first 10 lines to look for copyright
            lines = []
            for i, line in enumerate(f):
                if i >= 10:  # Only check first 10 lines
                    break
                lines.append(line.strip())

        content = "\n".join(lines)
        return "SPDX-FileCopyrightText" in content or "Copyright" in content
    except:
        return False


def extract_copyright_year(file_path):
    """Extract the copyright year from a file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(1000)  # Read first 1000 chars

        # Look for copyright year patterns
        patterns = [r"SPDX-FileCopyrightText:\s*©\s*(\d{4})", r"Copyright.*©?\s*(\d{4})", r"Copyright.*(\d{4})"]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None
    except:
        return None


def validate_copyright_years(root_dir, current_year, verbose=False):
    """Validate that all copyright headers contain the current year."""
    files_with_old_years = []
    files_with_copyright = find_files_with_copyright(root_dir)

    if verbose:
        print(f"Found {len(files_with_copyright)} files with copyright headers")

    for file_path in files_with_copyright:
        year = extract_copyright_year(file_path)
        if year is None:
            if verbose:
                print(f"Warning: Could not extract year from {file_path}")
            continue

        if year != current_year:
            files_with_old_years.append((file_path, year))
            if verbose:
                print(f"Found old copyright year {year} in {file_path}")

    return files_with_old_years


def validate_changed_files_copyright_years(changed_files, current_year, verbose=False):
    """Validate copyright years for a specific list of changed files."""
    files_with_old_years = []

    for file_path in changed_files:
        if has_copyright_header(file_path):
            year = extract_copyright_year(file_path)
            if year is None:
                if verbose:
                    print(f"Warning: Could not extract year from {file_path}")
                continue

            if year != current_year:
                files_with_old_years.append((file_path, year))
                if verbose:
                    print(f"Found old copyright year {year} in {file_path}")

    return files_with_old_years


def main():
    parser = argparse.ArgumentParser(description="Validate copyright years in source files")
    parser.add_argument("files", nargs="*", help="Files to check (from pre-commit)")
    parser.add_argument("--year", type=int, help="Specific year to validate against (default: current year)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--all-files", action="store_true", help="Check all files instead of just changed files")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".cpp", ".cc", ".h", ".hpp", ".py", ".c", ".cxx"],
        help="File extensions to check",
    )

    args = parser.parse_args()

    current_year = args.year if args.year else get_current_year()

    if args.verbose:
        print(f"Validating copyright years against {current_year}")

    # If files are provided (pre-commit mode), use those
    if args.files:
        if args.verbose:
            print(f"Checking {len(args.files)} files from pre-commit...")
        changed_files = [Path(f) for f in args.files if Path(f).exists()]
        old_year_files = validate_changed_files_copyright_years(changed_files, current_year, args.verbose)
    elif args.all_files:
        # Check all files
        old_year_files = validate_copyright_years(".", current_year, args.verbose)
    else:
        # Check only changed files
        if args.verbose:
            print("Checking only changed files...")
        changed_files = get_changed_files(".", args.extensions)
        if changed_files is None:
            # Fall back to checking all files if git fails
            old_year_files = validate_copyright_years(".", current_year, args.verbose)
        else:
            if args.verbose:
                print(f"Found {len(changed_files)} changed files to check")
            old_year_files = validate_changed_files_copyright_years(changed_files, current_year, args.verbose)

    if old_year_files:
        print(f"\n❌ Found {len(old_year_files)} files with old copyright years:")
        for file_path, year in old_year_files:
            print(f"  {file_path} (has {year}, should be {current_year})")

        print("\nPlease update these files with the current year copyright header:")
        print(f"// SPDX-FileCopyrightText: © {current_year} Tenstorrent AI ULC")
        print("//")
        print("// SPDX-License-Identifier: Apache-2.0")
        return 1
    else:
        print(f"✅ All copyright headers contain the current year ({current_year})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
