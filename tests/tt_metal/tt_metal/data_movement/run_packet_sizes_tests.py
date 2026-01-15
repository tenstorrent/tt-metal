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


# Static mapping of test IDs to exact C++ test names
# This ensures exact matching with gtest filters
TEST_ID_TO_GTEST_NAME = {
    0: "TensixDataMovementDRAMPacketSizes",
    4: "TensixDataMovementOneToOnePacketSizes",
    5: "TensixDataMovementOneFromOnePacketSizes",
    6: "TensixDataMovementOneToAllUnicast2x2PacketSizes",
    7: "TensixDataMovementOneToAllUnicast5x5PacketSizes",
    8: "TensixDataMovementOneToAllUnicastPacketSizes",
    9: "TensixDataMovementOneToAllMulticast2x2PacketSizes",
    10: "TensixDataMovementOneToAllMulticast5x5PacketSizes",
    11: "TensixDataMovementOneToAllMulticastPacketSizes",
    12: "TensixDataMovementOneToAllMulticastLinked2x2PacketSizes",
    13: "TensixDataMovementOneToAllMulticastLinked5x5PacketSizes",
    14: "TensixDataMovementOneToAllMulticastLinkedPacketSizes",
    15: "TensixDataMovementOneFromAllPacketSizes",
    16: "TensixDataMovementLoopbackPacketSizes",
    80: "TensixDataMovementOnePacketReadSizes",
    81: "TensixDataMovementOnePacketWriteSizes",
    113: "TensixDataMovementMultiInterleavedReadSizes",
    115: "TensixDataMovementMultiInterleavedWriteSizes",
    117: "TensixDataMovementMultiInterleaved2x2Sizes",
    119: "TensixDataMovementMultiInterleaved2x2ReadSizes",
    121: "TensixDataMovementMultiInterleaved2x2WriteSizes",
    123: "TensixDataMovementMultiInterleaved6x6Sizes",
    125: "TensixDataMovementMultiInterleaved6x6ReadSizes",
    127: "TensixDataMovementMultiInterleaved6x6WriteSizes",
    146: "TensixDataMovementCoreBidirectionalPacketSizesSameKernel",
    147: "TensixDataMovementCoreBidirectionalPacketSizesDifferentKernels",
    301: "TensixDataMovementAllToAllPacketSizes",
    311: "TensixDataMovementAllFromAllPacketSizes",
}


def load_test_information(yaml_path):
    """Load test information YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_tests_with_metadata(test_info):
    """
    Extract test IDs and names for tests that have metadata defined.
    Returns list of tuples: (test_id, test_name, gtest_filter_name)
    """
    tests_with_metadata = []

    for test_id, test_data in test_info.get("tests", {}).items():
        # Check if test has metadata (pattern field indicates complete metadata)
        if "pattern" in test_data:
            test_name = test_data["name"]
            # Get the exact C++ test name from static mapping
            gtest_name = TEST_ID_TO_GTEST_NAME.get(test_id)

            if gtest_name is None:
                print(f"Warning: Test ID {test_id} has metadata but no gtest mapping. Skipping.")
                continue
            gtest_name += ":-2_0"  # exclude the 2_0 tests, since they are giving the same results
            tests_with_metadata.append((test_id, test_name, gtest_name))

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
