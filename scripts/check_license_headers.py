#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import re
import unicodedata
from pathlib import Path
import argparse

# License header text as a docstring
LICENSE_HEADER = """
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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


def normalize_line(line, normalize_year=True):
    line = unicodedata.normalize("NFKC", line).replace("\ufeff", "").strip()
    if normalize_year:
        line = re.sub(r"\b20\d{2}(–20\d{2})?\b", "<YEAR>", line)
    return re.sub(r"\s+", " ", line)


def strip_noise_lines(lines):
    return [line for line in lines if line.strip() and not re.match(r"^\s*(//|#)\s*$", line)]


def get_expected_header(file_ext: str, normalize_year=True):
    prefix = COMMENT_STYLES.get(file_ext)
    if not prefix:
        return None
    return [normalize_line(f"{prefix}{line}", normalize_year) for line in LICENSE_HEADER.strip().splitlines()]


def extract_header_block(path: Path, normalize_year=True):
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
        return [normalize_line(line, normalize_year) for line in lines]
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}", file=sys.stderr)
        return None


def check_file(path: Path, expected_lines, normalize_year=True):
    actual_lines = extract_header_block(path, normalize_year)
    if actual_lines is None:
        return False

    actual = strip_noise_lines(actual_lines)
    expected = strip_noise_lines(expected_lines)

    if actual != expected:
        print(f"❌ Mismatch in {path}")
        print("---- Expected ----")
        print("\n".join(expected))
        print("---- Found ----")
        print("\n".join(actual))
        print()
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Check license headers in files.")
    parser.add_argument("--ignore-year", action="store_true", help="Ignore year differences.")
    parser.add_argument("files", nargs="+", help="Files to check")
    args = parser.parse_args()

    failed = False
    for file_arg in args.files:
        path = Path(file_arg)
        ext = path.suffix
        expected = get_expected_header(ext, args.ignore_year)
        if expected is None:
            continue  # Skip unsupported files
        if not check_file(path, expected, args.ignore_year):
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
