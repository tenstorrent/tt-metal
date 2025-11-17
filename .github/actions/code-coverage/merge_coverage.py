#!/usr/bin/env python3
"""
Merge multiple LCOV format coverage files into a single file.

This script handles:
- Merging coverage data from multiple LCOV files
- Summing execution counts for the same lines across files
- Preserving function and branch coverage data
- Handling duplicate file entries
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


class CoverageData:
    """Store coverage data for a single source file."""

    def __init__(self):
        self.test_name = None
        self.lines = {}  # line_num -> execution_count
        self.functions = {}  # function_name -> (line_num, execution_count)
        self.branches = {}  # (line_num, block, branch) -> taken_count
        self.found_lines = 0
        self.hit_lines = 0
        self.found_functions = 0
        self.hit_functions = 0
        self.found_branches = 0
        self.hit_branches = 0


def parse_lcov_file(lcov_file):
    """
    Parse an LCOV file and return a dictionary mapping file paths to CoverageData.

    Returns: dict[str, CoverageData]
    """
    coverage_data = {}
    current_file = None
    current_data = None

    if not os.path.exists(lcov_file):
        print(f"Warning: LCOV file not found: {lcov_file}", file=sys.stderr)
        return coverage_data

    try:
        with open(lcov_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip()

                # Test name
                if line.startswith("TN:"):
                    test_name = line[3:].strip() or None
                    if current_data:
                        current_data.test_name = test_name

                # Source file
                elif line.startswith("SF:"):
                    file_path = line[3:].strip()
                    if file_path not in coverage_data:
                        coverage_data[file_path] = CoverageData()
                    current_file = file_path
                    current_data = coverage_data[file_path]

                # Line data: DA:<line_number>,<execution_count>
                elif line.startswith("DA:"):
                    if not current_data:
                        continue
                    match = re.match(r"DA:(\d+),(\d+)", line)
                    if match:
                        line_num = int(match.group(1))
                        exec_count = int(match.group(2))
                        # Sum execution counts if line appears multiple times
                        current_data.lines[line_num] = current_data.lines.get(line_num, 0) + exec_count

                # Function definition: FN:<line_number>,<function_name>
                elif line.startswith("FN:"):
                    if not current_data:
                        continue
                    match = re.match(r"FN:(\d+),(.+)", line)
                    if match:
                        line_num = int(match.group(1))
                        func_name = match.group(2)
                        current_data.functions[func_name] = (line_num, 0)

                # Function execution: FNDA:<execution_count>,<function_name>
                elif line.startswith("FNDA:"):
                    if not current_data:
                        continue
                    match = re.match(r"FNDA:(\d+),(.+)", line)
                    if match:
                        exec_count = int(match.group(1))
                        func_name = match.group(2)
                        if func_name in current_data.functions:
                            old_line, old_count = current_data.functions[func_name]
                            current_data.functions[func_name] = (old_line, old_count + exec_count)
                        else:
                            # Function not defined yet, create entry
                            current_data.functions[func_name] = (0, exec_count)

                # Branch data: BRDA:<line>,<block>,<branch>,<taken> or BRDA:<line>,<block>,<branch>,-
                elif line.startswith("BRDA:"):
                    if not current_data:
                        continue
                    match = re.match(r"BRDA:(\d+),(\d+),(\d+),(-?\d+)", line)
                    if match:
                        line_num = int(match.group(1))
                        block = int(match.group(2))
                        branch = int(match.group(3))
                        taken_str = match.group(4)
                        key = (line_num, block, branch)
                        if taken_str == "-":
                            taken = 0
                        else:
                            taken = int(taken_str)
                        current_data.branches[key] = current_data.branches.get(key, 0) + taken

                # Summary lines
                elif line.startswith("LF:"):
                    if current_data:
                        current_data.found_lines = int(line[3:].strip())
                elif line.startswith("LH:"):
                    if current_data:
                        current_data.hit_lines = int(line[3:].strip())
                elif line.startswith("FNF:"):
                    if current_data:
                        current_data.found_functions = int(line[4:].strip())
                elif line.startswith("FNH:"):
                    if current_data:
                        current_data.hit_functions = int(line[4:].strip())
                elif line.startswith("BRF:"):
                    if current_data:
                        current_data.found_branches = int(line[4:].strip())
                elif line.startswith("BRH:"):
                    if current_data:
                        current_data.hit_branches = int(line[4:].strip())

                # End of record
                elif line == "end_of_record":
                    current_file = None
                    current_data = None

    except Exception as e:
        print(f"Error parsing LCOV file {lcov_file}: {e}", file=sys.stderr)
        return coverage_data

    return coverage_data


def merge_coverage_data(all_coverage_data):
    """
    Merge coverage data from multiple LCOV files.

    Args:
        all_coverage_data: list of dict[str, CoverageData] from multiple files

    Returns:
        dict[str, CoverageData] with merged data
    """
    merged = {}

    for coverage_dict in all_coverage_data:
        for file_path, data in coverage_dict.items():
            if file_path not in merged:
                merged[file_path] = CoverageData()

            merged_data = merged[file_path]

            # Merge lines
            for line_num, exec_count in data.lines.items():
                merged_data.lines[line_num] = merged_data.lines.get(line_num, 0) + exec_count

            # Merge functions
            for func_name, (line_num, exec_count) in data.functions.items():
                if func_name in merged_data.functions:
                    old_line, old_count = merged_data.functions[func_name]
                    merged_data.functions[func_name] = (line_num or old_line, old_count + exec_count)
                else:
                    merged_data.functions[func_name] = (line_num, exec_count)

            # Merge branches
            for key, taken_count in data.branches.items():
                merged_data.branches[key] = merged_data.branches.get(key, 0) + taken_count

    # Recalculate summaries
    for merged_data in merged.values():
        merged_data.found_lines = len(merged_data.lines)
        merged_data.hit_lines = sum(1 for count in merged_data.lines.values() if count > 0)
        merged_data.found_functions = len(merged_data.functions)
        merged_data.hit_functions = sum(1 for _, count in merged_data.functions.values() if count > 0)
        merged_data.found_branches = len(merged_data.branches)
        merged_data.hit_branches = sum(1 for count in merged_data.branches.values() if count > 0)

    return merged


def write_lcov_file(coverage_data, output_file):
    """
    Write merged coverage data to LCOV format file.
    """
    with open(output_file, "w") as f:
        for file_path, data in sorted(coverage_data.items()):
            # Test name
            if data.test_name:
                f.write(f"TN:{data.test_name}\n")
            else:
                f.write("TN:\n")

            # Source file
            f.write(f"SF:{file_path}\n")

            # Function definitions
            for func_name, (line_num, _) in sorted(data.functions.items(), key=lambda x: x[1][0]):
                f.write(f"FN:{line_num},{func_name}\n")

            # Function execution counts
            for func_name, (_, exec_count) in sorted(data.functions.items(), key=lambda x: x[1][0]):
                f.write(f"FNDA:{exec_count},{func_name}\n")

            # Function summary
            if data.found_functions > 0:
                f.write(f"FNF:{data.found_functions}\n")
                f.write(f"FNH:{data.hit_functions}\n")

            # Line data
            for line_num in sorted(data.lines.keys()):
                exec_count = data.lines[line_num]
                f.write(f"DA:{line_num},{exec_count}\n")

            # Line summary
            f.write(f"LF:{data.found_lines}\n")
            f.write(f"LH:{data.hit_lines}\n")

            # Branch data
            for (line_num, block, branch), taken_count in sorted(data.branches.items()):
                if taken_count == 0:
                    f.write(f"BRDA:{line_num},{block},{branch},-\n")
                else:
                    f.write(f"BRDA:{line_num},{block},{branch},{taken_count}\n")

            # Branch summary
            if data.found_branches > 0:
                f.write(f"BRF:{data.found_branches}\n")
                f.write(f"BRH:{data.hit_branches}\n")

            # End of record
            f.write("end_of_record\n")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LCOV coverage files")
    parser.add_argument("lcov_files", nargs="+", help="Input LCOV files to merge")
    parser.add_argument("--output", "-o", default="-", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Parse all LCOV files
    all_coverage_data = []
    for lcov_file in args.lcov_files:
        if not os.path.exists(lcov_file):
            print(f"Warning: Skipping non-existent file: {lcov_file}", file=sys.stderr)
            continue
        coverage_data = parse_lcov_file(lcov_file)
        if coverage_data:
            all_coverage_data.append(coverage_data)

    if not all_coverage_data:
        print("Error: No valid LCOV files to merge", file=sys.stderr)
        sys.exit(1)

    # Merge coverage data
    merged = merge_coverage_data(all_coverage_data)

    # Write output
    if args.output == "-":
        write_lcov_file(merged, sys.stdout)
    else:
        write_lcov_file(merged, args.output)


if __name__ == "__main__":
    main()
