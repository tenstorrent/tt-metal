#!/usr/bin/env python3
"""
Generate zero-coverage LCOV entries for all source files in the repository.

This ensures that ALL files appear in the coverage report, even if they
have 0% coverage. Files that already have coverage data will be merged
with this zero-coverage data (the merge script will take the maximum).
"""

import argparse
import os
import sys
from pathlib import Path


def is_source_file(file_path):
    """Check if a file is a C++ source file."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in [".cpp", ".cc", ".cxx", ".c", ".hpp", ".h", ".hxx"]


def should_exclude_path(file_path, source_dir):
    """Check if a path should be excluded from coverage."""
    rel_path = os.path.relpath(file_path, source_dir)

    # Exclude common build and cache directories
    exclude_patterns = [
        "/build",
        "/.cpmcache",
        "/.git",
        "/third_party",
        "/external",
        "/venv",
        "/env",
        "/__pycache__",
        "/.pytest_cache",
        "/generated",
    ]

    # Check if any part of the path matches exclude patterns
    path_parts = rel_path.split(os.sep)
    for part in path_parts:
        if part.startswith(".") and part != ".":
            return True
        for pattern in exclude_patterns:
            if pattern.lstrip("/") in part or rel_path.startswith(pattern.lstrip("/")):
                return True

    return False


def count_executable_lines(file_path):
    """
    Count executable lines in a C++ source file.
    Returns: list of line numbers that are executable
    """
    executable_lines = []

    if not os.path.exists(file_path):
        return executable_lines

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            in_block_comment = False
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()

                # Skip blank lines
                if not stripped:
                    continue

                # Handle block comments
                if "/*" in stripped:
                    in_block_comment = True
                if "*/" in stripped:
                    in_block_comment = False
                    # Check if there's code after the comment ends
                    after_comment = stripped.split("*/", 1)[-1].strip()
                    if after_comment and not after_comment.startswith("//"):
                        executable_lines.append(line_num)
                    continue

                if in_block_comment:
                    continue

                # Skip single-line comments
                if stripped.startswith("//"):
                    continue

                # Include preprocessor directives
                if stripped.startswith("#"):
                    executable_lines.append(line_num)
                    continue

                # All other lines are considered executable
                executable_lines.append(line_num)

    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}", file=sys.stderr)

    return executable_lines


def generate_zero_coverage_lcov(source_dir, output_file, exclude_patterns=None):
    """
    Generate zero-coverage LCOV entries for all source files.
    """
    source_dir = os.path.abspath(source_dir)
    lcov_entries = []
    processed = 0
    skipped = 0

    print(f"Scanning for source files in: {source_dir}", file=sys.stderr)

    # Walk through all files
    for root, dirs, files in os.walk(source_dir):
        # Skip excluded directories
        rel_root = os.path.relpath(root, source_dir)
        if should_exclude_path(root, source_dir):
            # Remove from dirs to prevent descending
            dirs[:] = []
            continue

        for file in files:
            file_path = os.path.join(root, file)

            if not is_source_file(file_path):
                continue

            if should_exclude_path(file_path, source_dir):
                skipped += 1
                continue

            # Count executable lines
            executable_lines = count_executable_lines(file_path)

            if not executable_lines:
                skipped += 1
                continue

            # Generate LCOV entry with zero coverage
            rel_path = os.path.relpath(file_path, source_dir)
            abs_path = os.path.normpath(file_path)

            lines = []
            lines.append("TN:")
            lines.append(f"SF:{abs_path}")

            # Add DA entries with 0 coverage for each executable line
            for line_num in executable_lines:
                lines.append(f"DA:{line_num},0")

            # Add summary (all lines found, none hit)
            total_lines = len(executable_lines)
            lines.append(f"LF:{total_lines}")
            lines.append(f"LH:0")  # Zero hits
            lines.append("end_of_record")

            lcov_entries.append("\n".join(lines))
            processed += 1

            if processed % 100 == 0:
                print(f"  Processed {processed} files...", file=sys.stderr)

    # Write output
    print(f"Writing zero-coverage entries for {processed} files...", file=sys.stderr)
    with open(output_file, "w") as f:
        if lcov_entries:
            f.write("\n".join(lcov_entries))
            f.write("\n")

    print(f"✓ Generated zero-coverage for {processed} files (skipped {skipped})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate zero-coverage LCOV entries for all source files")
    parser.add_argument("--source-dir", required=True, help="Root directory of source code")
    parser.add_argument("--output", required=True, help="Output LCOV file path")

    args = parser.parse_args()

    generate_zero_coverage_lcov(args.source_dir, args.output)


if __name__ == "__main__":
    main()
