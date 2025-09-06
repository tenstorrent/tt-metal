# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Script to insert only the best configuration for each tensor size into the GEMM_FLOPS.md report
Usage: python3 insert_best_configs.py [--restore]
"""

import csv
import os
import sys
import shutil
import re
from pathlib import Path
from collections import defaultdict


def clean_header(header):
    """Clean and format header text for better readability"""
    # Remove [text] patterns
    header = re.sub(r"\[.*?\]", "", header)
    # Replace underscores with spaces
    header = header.replace("_", " ")
    # Capitalize words
    header = header.title()
    return header.strip()


def find_best_configs(csv_file_path):
    """Find the best configuration (highest TFLOPS) for each tensor size"""
    best_configs = {}  # key: (m, k, n), value: best row data

    try:
        with open(csv_file_path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Get tensor dimensions
                m = int(row["m"])
                k = int(row["k"])
                n = int(row["n"])
                tensor_size = (m, k, n)

                # Get TFLOPS value
                try:
                    tflops = float(row["tflops"])
                except (ValueError, KeyError):
                    continue

                # Check if this is the best config for this tensor size
                if tensor_size not in best_configs or tflops > best_configs[tensor_size]["tflops"]:
                    best_configs[tensor_size] = {"tflops": tflops, "row": row}

    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return {}

    return best_configs


def create_best_configs_table(csv_file_path, title):
    """Create markdown table with only the best configurations"""
    best_configs = find_best_configs(csv_file_path)

    if not best_configs:
        return [f"Error: No valid configurations found in {csv_file_path}"]

    lines = []
    lines.append("")
    lines.append("<details>")
    lines.append(f"<summary><strong>{title}</strong> (click to expand)</summary>")
    lines.append("")

    # Get headers from the first row
    first_row = next(iter(best_configs.values()))["row"]
    headers = list(first_row.keys())
    cleaned_headers = [clean_header(h) for h in headers]

    # Create table header
    header_row = "| " + " | ".join(cleaned_headers) + " |"
    lines.append(header_row)

    # Create separator row
    separator_row = "| " + " | ".join(["---"] * len(cleaned_headers)) + " |"
    lines.append(separator_row)

    # Sort by tensor size for better readability
    sorted_configs = sorted(best_configs.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))

    # Add data rows (only best configs)
    for tensor_size, config_data in sorted_configs:
        row = config_data["row"]
        values = [str(row[header]).strip() for header in headers]
        data_row = "| " + " | ".join(values) + " |"
        lines.append(data_row)

    lines.append("")
    lines.append(f"_Best configurations only: {len(best_configs)} unique tensor sizes (highest TFLOPS per size)._")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    return lines


def insert_tables_into_markdown(md_file_path, csv_files_info):
    """Insert best config tables into markdown file under 'All Data' section"""

    # Create backup
    backup_file = md_file_path + ".backup"
    shutil.copy2(md_file_path, backup_file)
    print(f"Created backup: {backup_file}")

    # Read original markdown file
    with open(md_file_path, "r") as f:
        lines = f.readlines()

    # Find "### All Data" section
    all_data_line = None
    for i, line in enumerate(lines):
        if line.strip() == "### All Data":
            all_data_line = i
            break

    if all_data_line is None:
        print("Error: Could not find '### All Data' section in markdown file")
        return False

    # Create new content
    new_lines = lines[: all_data_line + 1]  # Include the "### All Data" line

    # Add introduction
    new_lines.append("\n")
    new_lines.append(
        "This section contains the best performing configurations for each tensor size. For each unique matrix dimension (M×K×N), only the configuration achieving the highest TFLOPS is shown.\n"
    )
    new_lines.append("\n")

    # Add CSV tables with best configs only
    for csv_file, title in csv_files_info:
        if os.path.exists(csv_file):
            print(f"Processing {title}...")
            best_configs = find_best_configs(csv_file)
            print(f"  Found {len(best_configs)} unique tensor sizes")
            table_lines = create_best_configs_table(csv_file, title)
            new_lines.extend(line + "\n" for line in table_lines)
        else:
            print(f"Warning: {csv_file} not found")

    # Add any remaining content after "### All Data" (if any)
    remaining_lines = lines[all_data_line + 1 :]
    if remaining_lines:
        # Skip any existing content until we find a new section or end of file
        skip_until_new_section = True
        for line in remaining_lines:
            if skip_until_new_section and line.strip().startswith("###"):
                skip_until_new_section = False
            if not skip_until_new_section:
                new_lines.append(line)

    # Write updated markdown file
    with open(md_file_path, "w") as f:
        f.writelines(new_lines)

    print(f"Successfully updated {md_file_path} with best configuration tables")
    return True


def restore_backup(md_file_path):
    """Restore markdown file from backup"""
    backup_file = md_file_path + ".backup"
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, md_file_path)
        print(f"Restored from backup: {backup_file}")
        return True
    else:
        print("No backup file found")
        return False


def main():
    script_dir = Path(__file__).parent
    md_file = script_dir / "GEMM_FLOPS.md"

    # Check for restore flag
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_backup(str(md_file))
        return

    # Check if markdown file exists
    if not md_file.exists():
        print(f"Error: {md_file} not found")
        sys.exit(1)

    # Define CSV files and their titles
    csv_files_info = [
        (str(script_dir / "n150-sweep.csv"), "N150 Best Configurations"),
        (str(script_dir / "p150-sweep.csv"), "P150 Best Configurations"),
    ]

    # Insert tables
    success = insert_tables_into_markdown(str(md_file), csv_files_info)

    if success:
        print("Process completed successfully!")
        print(f"To restore the original file, run: python3 {__file__} --restore")
    else:
        print("Process failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
