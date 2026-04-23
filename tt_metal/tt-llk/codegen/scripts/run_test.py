#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run existing Quasar LLK tests.

Usage:
    # Run all reduce tests
    python scripts/run_test.py reduce

    # Run specific test with verbose output
    python scripts/run_test.py reduce -v

    # Run SFPU nonlinear tests
    python scripts/run_test.py sfpu_nonlinear

    # List available tests
    python scripts/run_test.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Test directory relative to this script
TESTS_DIR = Path(__file__).parent.parent.parent / "tests"
PYTHON_TESTS_DIR = TESTS_DIR / "python_tests"

# Map of test names to pytest paths
QUASAR_TESTS = {
    "reduce": "quasar/test_reduce_quasar.py",
    "sfpu_nonlinear": "quasar/test_sfpu_nonlinear_quasar.py",
    # Add more as needed
}


def list_tests():
    """List available Quasar tests."""
    print("Available Quasar tests:")
    print()

    # List known tests
    for name, path in QUASAR_TESTS.items():
        full_path = PYTHON_TESTS_DIR / path
        status = "✓" if full_path.exists() else "✗ (not found)"
        print(f"  {name:20} -> {path} {status}")

    print()
    print("Other Quasar tests in directory:")
    quasar_dir = PYTHON_TESTS_DIR / "quasar"
    if quasar_dir.exists():
        for f in sorted(quasar_dir.glob("test_*.py")):
            name = f.stem.replace("test_", "").replace("_quasar", "")
            if name not in QUASAR_TESTS:
                print(f"  {name:20} -> quasar/{f.name}")


def run_test(test_name: str, verbose: bool = False, extra_args: list = None):
    """Run a specific test."""
    if test_name not in QUASAR_TESTS:
        # Try to find it in quasar directory
        test_path = PYTHON_TESTS_DIR / "quasar" / f"test_{test_name}_quasar.py"
        if not test_path.exists():
            test_path = PYTHON_TESTS_DIR / "quasar" / f"test_{test_name}.py"
        if not test_path.exists():
            print(f"Error: Unknown test '{test_name}'")
            print("Use --list to see available tests")
            return 1
        rel_path = test_path.relative_to(PYTHON_TESTS_DIR)
    else:
        rel_path = QUASAR_TESTS[test_name]

    full_path = PYTHON_TESTS_DIR / rel_path
    if not full_path.exists():
        print(f"Error: Test file not found: {full_path}")
        return 1

    # Build pytest command
    cmd = ["pytest", str(rel_path)]

    if verbose:
        cmd.append("-v")

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {PYTHON_TESTS_DIR}")
    print("-" * 60)

    # Run pytest from the python_tests directory
    result = subprocess.run(
        cmd,
        cwd=PYTHON_TESTS_DIR,
    )

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run Quasar LLK tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run reduce tests
    python scripts/run_test.py reduce

    # Run with verbose output
    python scripts/run_test.py reduce -v

    # Run specific test case
    python scripts/run_test.py reduce -- -k "test_reduce_quasar[Float16_b"

    # List available tests
    python scripts/run_test.py --list
""",
    )
    parser.add_argument("test", nargs="?", help="Test name to run")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available tests"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("extra", nargs="*", help="Extra arguments to pass to pytest")

    args = parser.parse_args()

    if args.list:
        list_tests()
        return 0

    if not args.test:
        parser.print_help()
        print("\nError: Please specify a test name or use --list")
        return 1

    return run_test(args.test, args.verbose, args.extra)


if __name__ == "__main__":
    sys.exit(main())
