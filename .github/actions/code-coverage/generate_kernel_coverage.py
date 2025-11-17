#!/usr/bin/env python3
"""
Generate synthetic LCOV coverage entries for ALL kernel files in the repository.

Kernel files are JIT-compiled at runtime, so they're not compiled with coverage
instrumentation. This script:
1. Finds ALL kernel files in the repository (in */kernels/ directories)
2. Generates zero-coverage entries for all of them
3. Marks files listed in kernel_names.txt as 100% covered (since they were executed)
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


def generate_lcov_entry(file_path, source_dir, executable_lines, coverage_count=0):
    """
    Generate LCOV format entry for a single file.

    Args:
        file_path: Path to the file (relative or absolute)
        source_dir: Root directory of source code
        executable_lines: List of executable line numbers
        coverage_count: Execution count for each line (0 = not covered, 1+ = covered)

    Format:
    TN:
    SF:<absolute_path>
    DA:<line_number>,<count>
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

    # Add DA entries for each executable line
    for line_num in executable_lines:
        lines.append(f"DA:{line_num},{coverage_count}")

    # Add summary
    total_lines = len(executable_lines)
    hit_lines = sum(1 for _ in executable_lines) if coverage_count > 0 else 0

    lines.append(f"LF:{total_lines}")
    lines.append(f"LH:{hit_lines}")
    lines.append("end_of_record")

    return "\n".join(lines)


def find_all_kernel_files(source_dir):
    """
    Find all kernel files in the repository.

    Kernel files are typically in directories matching */kernels/ or */kernels_ng/
    """
    kernel_files = []
    source_dir = os.path.abspath(source_dir)

    # Patterns for kernel directories
    kernel_patterns = [
        "kernels",
        "kernels_ng",
    ]

    # File extensions for kernel files
    kernel_extensions = [".cpp", ".cc", ".cxx", ".c"]

    print(f"Scanning for kernel files in: {source_dir}", file=sys.stderr)

    # Walk through all files
    for root, dirs, files in os.walk(source_dir):
        # Skip excluded directories
        rel_root = os.path.relpath(root, source_dir)

        # Skip common build and cache directories
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

        should_exclude = False
        path_parts = rel_root.split(os.sep)
        for part in path_parts:
            if part.startswith(".") and part != ".":
                should_exclude = True
                break
            for pattern in exclude_patterns:
                if pattern.lstrip("/") in part or rel_root.startswith(pattern.lstrip("/")):
                    should_exclude = True
                    break
            if should_exclude:
                break

        if should_exclude:
            dirs[:] = []  # Don't descend into excluded directories
            continue

        # Check if this directory is a kernel directory
        # Look for directories containing "kernels" (e.g., "kernels", "test_kernels", "kernels_ng")
        is_kernel_dir = False
        root_normalized = root.replace("\\", "/")
        basename = os.path.basename(root)
        for pattern in kernel_patterns:
            # Check if pattern appears anywhere in the path (e.g., /kernels/, /test_kernels/, kernels_ng/, etc.)
            # This matches directories like: kernels, test_kernels, kernels_ng, device/kernels, etc.
            if pattern in root_normalized or pattern in basename:
                is_kernel_dir = True
                break

        if not is_kernel_dir:
            continue

        # Collect kernel files in this directory
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            if ext.lower() in kernel_extensions:
                rel_path = os.path.relpath(file_path, source_dir)
                kernel_files.append(rel_path)

    return sorted(set(kernel_files))  # Remove duplicates and sort


def generate_kernel_coverage(kernel_names_file, source_dir, output_file):
    """
    Main function to generate kernel coverage LCOV file.

    This function:
    1. Finds ALL kernel files in the repository
    2. Generates zero-coverage entries for all of them
    3. Marks files from kernel_names.txt as 100% covered (executed)
    """
    # Step 1: Find all kernel files in the repository
    all_kernel_files = find_all_kernel_files(source_dir)
    print(f"Found {len(all_kernel_files)} total kernel file(s) in repository", file=sys.stderr)

    # Step 2: Parse executed kernel files from kernel_names.txt
    executed_kernels = set()
    executed_kernels_abs = set()  # Also store absolute paths for matching
    if os.path.exists(kernel_names_file):
        executed_kernel_dict = parse_kernel_names_file(kernel_names_file)
        for kernel_path in executed_kernel_dict.keys():
            # Normalize path for comparison
            if os.path.isabs(kernel_path):
                norm_path = os.path.relpath(kernel_path, source_dir)
                abs_path = kernel_path
            else:
                norm_path = kernel_path
                abs_path = os.path.abspath(os.path.join(source_dir, kernel_path))

            # Normalize separators
            norm_path = norm_path.replace("\\", "/")
            abs_path = os.path.normpath(abs_path).replace("\\", "/")

            executed_kernels.add(norm_path)
            executed_kernels_abs.add(abs_path)
        print(f"Found {len(executed_kernels)} executed kernel file(s) in kernel_names.txt", file=sys.stderr)
        if len(executed_kernels) > 0:
            print(f"  Example: {list(executed_kernels)[0]}", file=sys.stderr)
    else:
        print(f"Warning: Kernel names file not found: {kernel_names_file}", file=sys.stderr)
        print("  All kernel files will be marked as 0% coverage", file=sys.stderr)

    # Step 3: Generate LCOV entries for all kernel files
    lcov_entries = []
    processed = 0
    skipped = 0
    executed_count = 0

    for kernel_path in all_kernel_files:
        # Resolve file path
        if os.path.isabs(kernel_path):
            file_path = kernel_path
        else:
            file_path = os.path.join(source_dir, kernel_path)

        file_path = os.path.normpath(file_path)
        abs_path = file_path  # Store the absolute path for this specific kernel file

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

        # Determine coverage: 1 if executed, 0 if not
        # Normalize kernel_path for comparison (use forward slashes)
        norm_kernel_path = kernel_path.replace("\\", "/")
        abs_kernel_path = file_path.replace("\\", "/")  # Use file_path, not abs_path variable

        is_executed = False
        # Only use exact matches - kernel files are JIT-compiled, so we need precise matching
        # Try exact match against relative paths first (most common case)
        if norm_kernel_path in executed_kernels:
            is_executed = True
        # Try exact match against absolute paths
        elif abs_kernel_path in executed_kernels_abs:
            is_executed = True

        coverage_count = 1 if is_executed else 0

        # Generate LCOV entry
        entry = generate_lcov_entry(kernel_path, source_dir, executable_lines, coverage_count)
        lcov_entries.append(entry)
        processed += 1
        if is_executed:
            executed_count += 1

        if processed % 100 == 0:
            print(f"  Processed {processed} kernel files...", file=sys.stderr)

    # Write output
    print(f"Writing LCOV entries for {processed} kernel files...", file=sys.stderr)
    with open(output_file, "w") as f:
        if lcov_entries:
            f.write("\n".join(lcov_entries))
            f.write("\n")

    print(f"✓ Processed {processed} kernel file(s):", file=sys.stderr)
    print(f"  - {executed_count} marked as executed (100% coverage)", file=sys.stderr)
    print(f"  - {processed - executed_count} marked as not executed (0% coverage)", file=sys.stderr)
    if skipped > 0:
        print(f"  - {skipped} skipped (file not found or empty)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LCOV coverage for kernel files")
    parser.add_argument("--kernel-names-file", required=True, help="Path to kernel_names.txt file")
    parser.add_argument("--source-dir", required=True, help="Root directory of source code")
    parser.add_argument("--output", required=True, help="Output LCOV file path")

    args = parser.parse_args()

    generate_kernel_coverage(args.kernel_names_file, args.source_dir, args.output)


if __name__ == "__main__":
    main()
