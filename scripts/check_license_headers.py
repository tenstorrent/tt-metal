#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import re
import unicodedata
from pathlib import Path
import argparse
import subprocess
from typing import Dict, Optional
import shutil

# License header text as a docstring
LICENSE_HEADER = """
SPDX-FileCopyrightText: © <YEAR> Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

# Supported file extensions and their comment prefixes
COMMENT_STYLES = {
    ".py": "# ",
    ".sh": "# ",
    ".c": "// ",
    ".cpp": "// ",
    ".cc": "// ",
    ".h": "// ",
    ".hpp": "// ",
    ".cuh": "// ",
    ".cu": "// ",
    ".js": "// ",
    ".ts": "// ",
    ".java": "// ",
    ".go": "// ",
}


# Slow but reliable fallback
def get_git_year(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%ad", "--date=format:%Y", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        years = result.stdout.strip().split("\n")
        return years[-1] if years else None
    except subprocess.CalledProcessError:
        return None


def normalize_line(line, normalize_year=True, git_year=None):
    line = unicodedata.normalize("NFKC", line).replace("\ufeff", "").strip()
    if normalize_year:
        # Only normalize if we're not comparing with a specific year
        if git_year is None:
            line = re.sub(r"\b20\d{2}(–20\d{2})?\b", "<YEAR>", line)
    return re.sub(r"\s+", " ", line)


def strip_noise_lines(lines):
    return [line for line in lines if line.strip() and not re.match(r"^\s*(//|#)\s*$", line)]


def get_expected_header(file_ext: str, normalize_year=True, git_year=None):
    prefix = COMMENT_STYLES.get(file_ext)
    if not prefix:
        return None

    # If we have a git year and not ignoring years, use that year
    if git_year and not normalize_year:
        header_lines = LICENSE_HEADER.strip().splitlines()
        header_lines = [line.replace("<YEAR>", str(git_year)) for line in header_lines]
        return [normalize_line(f"{prefix}{line}", False) for line in header_lines]

    # Otherwise use the template with <YEAR>
    return [normalize_line(f"{prefix}{line}", True) for line in LICENSE_HEADER.strip().splitlines()]


def extract_header_block(path: Path, normalize_year=True, git_year=None):
    try:
        ext = path.suffix
        comment_prefix = COMMENT_STYLES.get(ext)
        if not comment_prefix:
            return None

        lines = []
        with open(path, encoding="utf-8") as f:
            # Read up to the first 15 lines in search of SPDX header block
            for _ in range(15):
                line = f.readline()
                if not line:
                    break
                if "SPDX" in line:
                    lines.append(line.rstrip("\n\r"))
                    for _ in range(2):
                        next_line = f.readline()
                        if next_line:
                            lines.append(next_line.rstrip("\n\r"))
                    break
        return [normalize_line(line, normalize_year, git_year) for line in lines]
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}", file=sys.stderr)
        return None


def check_file(path: Path, expected_lines, normalize_year=True, git_year=None):
    actual_lines = extract_header_block(path, normalize_year, git_year)
    if actual_lines is None:
        return False

    actual = strip_noise_lines(actual_lines)
    expected = strip_noise_lines(expected_lines)
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    print(f" Check {actual == expected}")
    print("Expected Line 0 chars:", [ord(c) for c in expected[0]])
    print("Actual Line 0 chars:  ", [ord(c) for c in actual[0]])
    for i, (e, a) in enumerate(zip(expected, actual)):
        print(f"Line {i}:")
        print(f"  expected: {repr(e)}")
        print(f"  actual:   {repr(a)}")
        print(f"  equal:    {e == a}")
    if actual != expected:
        print(f"❌ Mismatch in {path}")
        print("---- Expected ----")
        print("\n".join(expected))
        print("---- Found ----")
        print("\n".join(actual))
        print()
        return False
    return True


def check_git_requirements():
    # Check if git is installed
    if not shutil.which("git"):
        print("❌ Error: git is not installed or not in PATH", file=sys.stderr)
        return False

    # Check if we're in a git repository
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a git repository", file=sys.stderr)
        return False

    # Check if repository has history
    try:
        result = subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True, check=True)
        commit_count = int(result.stdout.strip())
        if commit_count == 0:
            print("❌ Error: Git repository has no commit history", file=sys.stderr)
            return False
    except (subprocess.CalledProcessError, ValueError):
        print("❌ Error: Failed to check git history", file=sys.stderr)
        return False

    return True


def get_file_years() -> dict[Path, int | None]:
    try:
        result = subprocess.Popen(
            ["git", "log", "--name-status", "--format=%cs", "--date=short"],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as ex:
        print(ex)
        sys.exit(1)

    year: int = 0
    files: dict[str, int | Path] = {}
    for line in result.stdout:
        if not (line) or len(line) < 2 or line[0] == " " or line[0] == "M" or line[0] == "D":
            continue

        elif line[0].isdigit():
            year = int(line[0:4])

        elif line[0] == "A":
            files[Path(line.split()[1])] = year

        # this doesn't gracefully handle spaces in paths, so we handle that elsewhere
        elif line[0] == "R":
            split_line = line.split()
            # R<num> <original> <new>
            files[Path(split_line[2])] = Path(split_line[1])
        else:
            continue

    # resolve renames
    for key in files.keys():
        val = files[key]
        while (val is not None) and (type(val) is not int):
            new_val = files.get(val)  # returns None if key doesn't exist
            files[key] = new_val
            val = new_val

    return files


# now you have a master list of all the years for each file


def main():
    parser = argparse.ArgumentParser(description="Check license headers in files.")
    parser.add_argument("--ignore-year", action="store_true", help="Ignore year differences.")
    parser.add_argument("files", nargs="+", help="Files to check")
    args = parser.parse_args()

    # Check git requirements before proceeding
    if not check_git_requirements():
        sys.exit(1)

    file_years = get_file_years()

    failed = False
    for file_arg in args.files:
        path = Path(file_arg)
        ext = path.suffix
        print(f"Checking {path} with ext {ext}", file=sys.stderr)

        git_year = file_years.get(path)

        # the fast lookup table doesn't handle spaces in file names, so when
        # we come across a file that was named incorrectly, use the slower
        # method as a reliable fallback.
        if git_year is None:
            git_year = get_git_year(path)

        print(f"Git year: {git_year}", file=sys.stderr)
        expected = get_expected_header(ext, args.ignore_year, git_year)

        if expected is None:
            continue  # Skip unsupported files

        if not check_file(path, expected, args.ignore_year, git_year):
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
