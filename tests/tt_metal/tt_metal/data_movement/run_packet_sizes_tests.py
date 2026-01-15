#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to run all relevant data movement packet sizes tests (tests with metadata) and generate reports.
This script identifies tests that have the 'pattern' metadata field defined and runs them with --report flag.
"""

import yaml
import sys
import os
import subprocess
from pathlib import Path


def convert_test_name_to_gtest_filter(test_name: str) -> str:
    """
    Convert test name to gtest filter format.

    Args:
        test_name: Test name from YAML (e.g., "DRAM Packet Sizes")

    Returns:
        Gtest filter name with exclusion suffix
        (e.g., "TensixDataMovementDRAMPacketSizes:-2_0")

    Examples:
        "DRAM Packet Sizes" -> "TensixDataMovementDRAMPacketSizes:-2_0"
        "One to One Packet Sizes" -> "TensixDataMovementOneToOnePacketSizes:-2_0"
        "One from All Packet Sizes" -> "TensixDataMovementOneFromAllPacketSizes:-2_0"
        "Multi Interleaved 2x2 Sizes" -> "TensixDataMovementMultiInterleaved2x2Sizes:-2_0"
    """
    # Split by spaces and capitalize each word, preserving acronyms
    words = test_name.split()
    capitalized_words = []
    for word in words:
        # If word is all uppercase (acronym like DRAM, I2S), keep it
        if word.isupper():
            capitalized_words.append(word)
        else:
            capitalized_words.append(word.capitalize())

    # Join words and replace periods with underscores (for 2.0 -> 2_0)
    gtest_name = "".join(capitalized_words)
    gtest_name = gtest_name.replace(".", "_")
    # Prepend standard prefix
    gtest_filter = f"TensixDataMovement{gtest_name}"
    # Add suffix to exclude 2.0 API tests (they give same results)
    gtest_filter += ":-2_0"

    return gtest_filter


def load_test_information(yaml_path):
    """Load test information YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_tests_with_metadata(test_info):
    """
    Extract test IDs and names for tests that:
    1. Have "Sizes" in the name (packet sizes tests)
    2. Have metadata defined (pattern field present)

    Returns list of tuples: (test_id, test_name, gtest_filter_name)
    """
    tests_with_metadata = []

    for test_id, test_data in test_info.get("tests", {}).items():
        test_name = test_data.get("name", "")

        # Check if test name contains "Sizes" and has metadata
        if "Sizes" in test_name and "pattern" in test_data:
            gtest_filter = convert_test_name_to_gtest_filter(test_name)
            tests_with_metadata.append((test_id, test_name, gtest_filter))

    return sorted(tests_with_metadata, key=lambda x: x[0])


def run_tests(tests, verbose=False):
    """
    Run the specified tests individually with --report flag.

    Args:
        tests: List of (test_id, test_name, gtest_filter) tuples
        verbose: Whether to enable verbose logging
    """
    if not tests:
        print("No tests with metadata found.")
        return 1

    print("=" * 80)
    print(f"Running {len(tests)} packet sizes tests with --report flag")
    print("=" * 80 + "\n")

    failed_tests = []
    successful_tests = []

    for i, (test_id, test_name, gtest_filter) in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] Running Test ID {test_id}: {test_name}")
        print(f"  gtest filter: {gtest_filter}")

        # Build pytest command for individual test
        cmd = [
            "pytest",
            "tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py",
            f"--gtest-filter={gtest_filter}",
            "--report",
        ]

        if verbose:
            cmd.append("--verbose-log")

        # Run the command
        result = subprocess.run(cmd, cwd=os.environ.get("TT_METAL_HOME", os.getcwd()), capture_output=not verbose)

        if result.returncode == 0:
            print(f"  ✓ PASSED\n")
            successful_tests.append((test_id, test_name))
        else:
            print(f"  ✗ FAILED\n")
            failed_tests.append((test_id, test_name))

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if failed_tests:
        print("\nFailed tests:")
        for test_id, test_name in failed_tests:
            print(f"  [{test_id:3d}] {test_name}")
        return 1

    print("\n✓ All tests passed!")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all packet sizes tests and generate CSV reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all packet sizes tests with report generation
  python tests/tt_metal/tt_metal/data_movement/run_packet_sizes_tests.py

  # Run with verbose logging
  python tests/tt_metal/tt_metal/data_movement/run_packet_sizes_tests.py --verbose

  # List tests without running
  python tests/tt_metal/tt_metal/data_movement/run_packet_sizes_tests.py --list-only
""",
    )

    parser.add_argument("--list-only", action="store_true", help="List tests with metadata without running them")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Get script directory and locate test_information.yaml
    script_dir = Path(__file__).parent
    yaml_path = script_dir / "python" / "test_mappings" / "test_information.yaml"

    if not yaml_path.exists():
        print(f"Error: Could not find test_information.yaml at {yaml_path}")
        return 1

    # Load test information and find tests with metadata
    test_info = load_test_information(yaml_path)
    tests_with_metadata = get_tests_with_metadata(test_info)

    if not tests_with_metadata:
        print("No tests with metadata (pattern field) found.")
        return 1

    print(f"\nFound {len(tests_with_metadata)} packet sizes tests with metadata:\n")
    for test_id, test_name, gtest_filter in tests_with_metadata:
        print(f"  [{test_id:3d}] {test_name} -> {gtest_filter}")

    if args.list_only:
        print("\n(Use --report flag to run and generate CSV files)")
        return 0

    # Run the tests
    print("\n" + "=" * 80)
    print("Running tests with --report flag...")
    print("=" * 80 + "\n")

    return run_tests(tests_with_metadata, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
