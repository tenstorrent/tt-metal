#!/usr/bin/env python3
"""
Generate synthetic LCOV coverage entries for kernel files listed in kernel_names.txt.

This script parses kernel_names.txt and generates LCOV format entries marking
all lines in each kernel file as executed (100% coverage), since we can't get
line-level coverage data from the hardware.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import OrderedDict


def parse_kernel_names_file(kernel_names_file):
    """
    Parse kernel_names.txt and extract unique kernel file paths.

    Format: "ID: path/to/kernel.cpp"
    Returns: OrderedDict mapping file paths to their first occurrence ID
    """
    kernel_files = OrderedDict()

    if not os.path.exists(kernel_names_file):
        print(f"Warning: Kernel names file not found: {kernel_names_file}", file=sys.stderr)
        return kernel_files

    with open(kernel_names_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse format: "ID: path/to/file.cpp"
            if ":" not in line:
                print(f"Warning: Invalid format at line {line_num}: {line}", file=sys.stderr)
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                print(f"Warning: Invalid format at line {line_num}: {line}", file=sys.stderr)
                continue

            kernel_id = parts[0].strip()
            kernel_path = parts[1].strip()

            # Skip "blank" entries
            if kernel_path.lower() == "blank":
                continue

            # Store unique files only (keep first occurrence)
            if kernel_path not in kernel_files:
                kernel_files[kernel_path] = kernel_id

    return kernel_files


def count_executable_lines(file_path):
    """
    Count executable lines in a C++ source file.
    Skips comments, blank lines, and preprocessor directives.

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

                # Skip preprocessor directives (but include #include, #define with values, etc.)
                if stripped.startswith("#"):
                    # Include #include, #define, #pragma, etc. as they're "executed"
                    executable_lines.append(line_num)
                    continue

                # All other lines are considered executable
                executable_lines.append(line_num)

    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}", file=sys.stderr)

    return executable_lines


def generate_lcov_entry(file_path, source_dir, executable_lines):
    """
    Generate LCOV format entry for a single file.

    Format:
    TN:
    SF:<absolute_path>
    DA:<line_number>,1
    ...
    LF:<total>
    LH:<hit>
    end_of_record
    """
    # Convert to absolute path
    if os.path.isabs(file_path):
        abs_path = file_path
    else:
        abs_path = os.path.abspath(os.path.join(source_dir, file_path))

    # Normalize path separators
    abs_path = os.path.normpath(abs_path)

    lines = []
    lines.append("TN:")
    lines.append(f"SF:{abs_path}")

    # Add DA entries for each executable line (marked as executed once)
    for line_num in executable_lines:
        lines.append(f"DA:{line_num},1")

    # Add summary
    total_lines = len(executable_lines)
    hit_lines = total_lines  # All lines are marked as executed

    lines.append(f"LF:{total_lines}")
    lines.append(f"LH:{hit_lines}")
    lines.append("end_of_record")

    return "\n".join(lines)


def generate_kernel_coverage(kernel_names_file, source_dir, output_file):
    """
    Main function to generate kernel coverage LCOV file.
    """
    # Parse kernel names
    kernel_files = parse_kernel_names_file(kernel_names_file)

    if not kernel_files:
        print("No kernel files found in kernel_names.txt", file=sys.stderr)
        # Create empty LCOV file
        with open(output_file, "w") as f:
            pass
        return

    print(f"Found {len(kernel_files)} unique kernel file(s)", file=sys.stderr)

    # Generate LCOV entries
    lcov_entries = []
    processed = 0
    skipped = 0

    for kernel_path, kernel_id in kernel_files.items():
        # Resolve file path
        if os.path.isabs(kernel_path):
            file_path = kernel_path
        else:
            file_path = os.path.join(source_dir, kernel_path)

        file_path = os.path.normpath(file_path)

        if not os.path.exists(file_path):
            print(f"Warning: Kernel file not found: {file_path} (from {kernel_path})", file=sys.stderr)
            skipped += 1
            continue

        # Count executable lines
        executable_lines = count_executable_lines(file_path)

        if not executable_lines:
            print(f"Warning: No executable lines found in: {file_path}", file=sys.stderr)
            skipped += 1
            continue

        # Generate LCOV entry
        entry = generate_lcov_entry(kernel_path, source_dir, executable_lines)
        lcov_entries.append(entry)
        processed += 1

    # Write output
    with open(output_file, "w") as f:
        f.write("\n".join(lcov_entries))
        if lcov_entries:
            f.write("\n")

    print(f"Processed {processed} kernel file(s), skipped {skipped}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LCOV coverage for kernel files")
    parser.add_argument("--kernel-names-file", required=True, help="Path to kernel_names.txt file")
    parser.add_argument("--source-dir", required=True, help="Root directory of source code")
    parser.add_argument("--output", required=True, help="Output LCOV file path")

    args = parser.parse_args()

    generate_kernel_coverage(args.kernel_names_file, args.source_dir, args.output)


if __name__ == "__main__":
    main()
