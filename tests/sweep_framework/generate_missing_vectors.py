#!/usr/bin/env python3
"""
Script to generate vector files for missing modules by executing
sweeps_parameter_generator.py for each missing module.

Usage:
    python3 generate_missing_vectors.py

This script will:
    1. Extract module names from the missing.txt log file (copy log messages from terminal to missing.txt)
    2. Execute sweeps_parameter_generator.py for each missing module
"""

import re
import subprocess
import sys
from pathlib import Path


def extract_module_names(log_file_path):
    """
    Extract module names from the missing.txt log file.

    Args:
        log_file_path (str): Path to the log file

    Returns:
        list: List of module names that are missing vector files
    """
    module_names = []

    with open(log_file_path, "r") as f:
        for line in f:
            # Look for lines containing "No vector file found for module"
            if "No vector file found for module" in line:
                # Extract the module name between single quotes
                match = re.search(r"module '([^']+)'", line)
                if match:
                    module_name = match.group(1)
                    module_names.append(module_name)

    return module_names


def execute_command(module_name):
    """
    Execute the sweeps_parameter_generator.py command for a given module.

    Args:
        module_name (str): The module name to generate vectors for

    Returns:
        bool: True if command executed successfully, False otherwise
    """
    cmd = [
        "python3",
        "tests/sweep_framework/sweeps_parameter_generator.py",
        "--dump-file",
        "--module-name",
        module_name,
    ]

    print(f"Executing: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully generated vectors for module: {module_name}")
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate vectors for module: {module_name}")
        print(f"  Error: {e.stderr.strip()}")
        return False


def main():
    """Main function to process missing modules and generate vector files."""
    log_file_path = "tests/sweep_framework/missing.txt"

    # Check if log file exists
    if not Path(log_file_path).exists():
        print(f"Error: Log file not found at {log_file_path}")
        sys.exit(1)

    # Extract missing module names
    print("Extracting missing module names from log file...")
    missing_modules = extract_module_names(log_file_path)

    if not missing_modules:
        print("No missing modules found in the log file.")
        return

    print(f"Found {len(missing_modules)} missing modules:")
    for i, module in enumerate(missing_modules, 1):
        print(f"  {i}. {module}")

    print(f"\nGenerating vector files for {len(missing_modules)} modules...")
    print("=" * 60)

    # Execute command for each missing module
    successful = 0
    failed = 0

    for module_name in missing_modules:
        if execute_command(module_name):
            successful += 1
        else:
            failed += 1
        print()  # Add spacing between commands

    # Summary
    print("=" * 60)
    print(f"Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(missing_modules)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
