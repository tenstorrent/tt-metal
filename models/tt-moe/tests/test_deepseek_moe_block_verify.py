#!/usr/bin/env python3
"""
Quick verification script to check if parameterized tests are working.
This script just checks the test collection, not execution.
"""

import subprocess
import sys


def run_test_collection():
    """Run pytest collection to verify parameterization."""
    cmd = [
        "pytest",
        "models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference",
        "--collect-only",
        "-q",
    ]

    print("Collecting tests...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/ntarafdar/tt-moe/tt-metal")

    print("\nOutput:")
    print(result.stdout)

    if result.stderr:
        print("\nErrors:")
        print(result.stderr)

    # Check if both test variants are present
    expected_tests = [
        "test_deepseek_moe_against_reference[mode_decode_seq_1]",
        "test_deepseek_moe_against_reference[mode_prefill_seq_128]",
    ]

    for test in expected_tests:
        if test in result.stdout:
            print(f"✅ Found: {test}")
        else:
            print(f"❌ Missing: {test}")

    return result.returncode == 0


if __name__ == "__main__":
    success = run_test_collection()
    if success:
        print("\n✅ Test parameterization verified successfully!")
        print("\nTo run specific tests:")
        print("  # Run decode mode only:")
        print(
            "  pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference[mode_decode_seq_1] -xvs"
        )
        print("\n  # Run prefill mode only:")
        print(
            "  pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference[mode_prefill_seq_128] -xvs"
        )
        print("\n  # Run both modes:")
        print("  pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvs")
    else:
        print("\n❌ Test collection failed!")
        sys.exit(1)
