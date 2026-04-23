#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run functional tests for LLK kernels.

This script is focused ONLY on running functional tests.
For compilation checking, use check_compile.py.

Usage:
    # Run functional tests for a specific kernel
    python scripts/run_functional_test.py sigmoid

    # Run with specific data format
    python scripts/run_functional_test.py exp --format Float16_b

    # Run specific test cases only (quick smoke test)
    python scripts/run_functional_test.py relu --quick

    # List available tests
    python scripts/run_functional_test.py --list

    # Verbose output
    python scripts/run_functional_test.py tanh -v
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class KernelType(Enum):
    """Types of LLK kernels."""

    SFPU = "sfpu"
    MATH = "math"
    PACK = "pack"
    UNPACK = "unpack"


@dataclass
class FunctionalTestResult:
    """Result of functional testing."""

    kernel: str
    test_file: str
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    output: str
    return_code: int

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Functional Test: {self.kernel}\n"
            f"  Test file: {self.test_file}\n"
            f"  Status: {status}\n"
            f"  Tests: {self.passed_tests}/{self.total_tests} passed"
        )


# Test directory paths
TESTS_DIR = Path(__file__).parent.parent.parent / "tests"
PYTHON_TESTS_DIR = TESTS_DIR / "python_tests"


# Mapping of kernel names to their types
KERNEL_TYPES = {
    # SFPU operations
    "sigmoid": KernelType.SFPU,
    "relu": KernelType.SFPU,
    "exp": KernelType.SFPU,
    "gelu": KernelType.SFPU,
    "tanh": KernelType.SFPU,
    "sqrt": KernelType.SFPU,
    "recip": KernelType.SFPU,
    "reciprocal": KernelType.SFPU,
    "rsqrt": KernelType.SFPU,
    "square": KernelType.SFPU,
    # Math operations
    "reduce": KernelType.MATH,
    "matmul": KernelType.MATH,
    "eltwise_binary": KernelType.MATH,
    "eltwise_unary": KernelType.MATH,
    # Pack operations
    "pack": KernelType.PACK,
    "pack_untilize": KernelType.PACK,
    # Unpack operations
    "unpack_tilize": KernelType.UNPACK,
}


# Mapping of kernel names to pytest test files
KERNEL_TESTS = {
    # SFPU operations that use the nonlinear test
    "exp": {
        "file": "quasar/test_sfpu_nonlinear_quasar.py",
        "filter": "Exp",
    },
    "relu": {
        "file": "quasar/test_sfpu_nonlinear_quasar.py",
        "filter": "Relu",
    },
    "reciprocal": {
        "file": "quasar/test_sfpu_nonlinear_quasar.py",
        "filter": "Reciprocal",
    },
    "recip": {
        "file": "quasar/test_sfpu_nonlinear_quasar.py",
        "filter": "Reciprocal",
    },
    "sqrt": {
        "file": "quasar/test_sfpu_nonlinear_quasar.py",
        "filter": "Sqrt",
    },
    "tanh": {
        "file": "quasar/test_sfpu_nonlinear_quasar.py",
        "filter": "Tanh",
    },
    # Specialized SFPU tests
    "rsqrt": {
        "file": "quasar/test_sfpu_rsqrt_quasar.py",
        "filter": None,
    },
    "square": {
        "file": "quasar/test_sfpu_square_quasar.py",
        "filter": None,
    },
    # Math tests
    "reduce": {
        "file": "quasar/test_reduce_quasar.py",
        "filter": None,
    },
    "matmul": {
        "file": "quasar/test_matmul_quasar.py",
        "filter": None,
    },
    "eltwise_binary": {
        "file": "quasar/test_eltwise_binary_quasar.py",
        "filter": None,
    },
    "eltwise_unary": {
        "file": "quasar/test_eltwise_unary_datacopy_quasar.py",
        "filter": None,
    },
    # Pack tests
    "pack": {
        "file": "quasar/test_pack_quasar.py",
        "filter": None,
    },
    "pack_untilize": {
        "file": "quasar/test_pack_untilize_quasar.py",
        "filter": None,
    },
    # Unpack tests
    "unpack_tilize": {
        "file": "quasar/test_unpack_tilize_quasar.py",
        "filter": None,
    },
}


def get_test_info(kernel_name: str) -> Optional[dict]:
    """Get test file and filter for a kernel."""
    name = kernel_name.lower()

    if name in KERNEL_TESTS:
        return KERNEL_TESTS[name]

    # Try to find a matching test file
    pattern = f"test_*{name}*_quasar.py"
    matches = list((PYTHON_TESTS_DIR / "quasar").glob(pattern))
    if matches:
        return {"file": f"quasar/{matches[0].name}", "filter": None}

    return None


def run_functional_test(
    kernel_name: str,
    data_format: Optional[str] = None,
    quick: bool = False,
    verbose: bool = False,
    extra_args: Optional[list] = None,
) -> FunctionalTestResult:
    """
    Run functional tests for a kernel.

    Args:
        kernel_name: Name of the kernel (e.g., "sigmoid", "exp")
        data_format: Specific data format to test (e.g., "Float16_b")
        quick: Run minimal test cases for quick validation
        verbose: Enable verbose output
        extra_args: Additional pytest arguments

    Returns:
        FunctionalTestResult with test outcomes
    """
    test_info = get_test_info(kernel_name)

    if not test_info:
        return FunctionalTestResult(
            kernel=kernel_name,
            test_file="",
            passed=False,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            output=f"No functional tests found for '{kernel_name}'",
            return_code=-1,
        )

    test_file = test_info["file"]
    test_filter = test_info.get("filter")

    test_path = PYTHON_TESTS_DIR / test_file
    if not test_path.exists():
        return FunctionalTestResult(
            kernel=kernel_name,
            test_file=test_file,
            passed=False,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            output=f"Test file not found: {test_path}",
            return_code=-1,
        )

    # Build pytest command
    cmd = ["pytest", test_file]

    if verbose:
        cmd.append("-v")

    # Build filter expression
    filter_parts = []
    if test_filter:
        filter_parts.append(test_filter)
    if data_format:
        filter_parts.append(data_format)
    if quick:
        # For quick mode, test only one format and small dimensions
        # Can't use "32, 32" directly - commas break pytest -k parser
        filter_parts.append("not 64")  # Exclude 64x64, keep only 32x32

    if filter_parts:
        cmd.extend(["-k", " and ".join(filter_parts)])

    # Quasar always requires simulator
    chip_arch = os.environ.get("CHIP_ARCH", "quasar").lower()
    if chip_arch == "quasar":
        cmd.append("--run-simulator")
        sim_port = os.environ.get("LLK_SIMULATOR_PORT", "5556")
        cmd.extend(["--port", sim_port])

    # Add timeout
    cmd.extend(["--timeout", "600"])

    # Add extra args
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {PYTHON_TESTS_DIR}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            cwd=PYTHON_TESTS_DIR,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        output = result.stdout + result.stderr

        # Parse test counts from pytest output
        total, passed, failed = _parse_pytest_output(output)

        return FunctionalTestResult(
            kernel=kernel_name,
            test_file=test_file,
            passed=result.returncode == 0,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            output=output,
            return_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return FunctionalTestResult(
            kernel=kernel_name,
            test_file=test_file,
            passed=False,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            output="Tests timed out after 10 minutes",
            return_code=-1,
        )
    except Exception as e:
        return FunctionalTestResult(
            kernel=kernel_name,
            test_file=test_file,
            passed=False,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            output=f"Error running tests: {e}",
            return_code=-1,
        )


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest output to extract test counts."""
    import re

    # Look for patterns like "5 passed", "2 failed", "1 error"
    passed = 0
    failed = 0
    errors = 0

    passed_match = re.search(r"(\d+) passed", output)
    if passed_match:
        passed = int(passed_match.group(1))

    failed_match = re.search(r"(\d+) failed", output)
    if failed_match:
        failed = int(failed_match.group(1))

    error_match = re.search(r"(\d+) error", output)
    if error_match:
        errors = int(error_match.group(1))

    total = passed + failed + errors
    return total, passed, failed + errors


def list_available_tests():
    """List all available functional tests."""
    print("Available functional tests:\n")

    # Group by kernel type
    by_type: dict[KernelType, list[str]] = {}
    for kernel, ktype in KERNEL_TYPES.items():
        if ktype not in by_type:
            by_type[ktype] = []
        by_type[ktype].append(kernel)

    for ktype in KernelType:
        kernels = by_type.get(ktype, [])
        if not kernels:
            continue

        print(f"{ktype.value.upper()} kernels:")
        for kernel in sorted(set(kernels)):
            test_info = get_test_info(kernel)
            if test_info:
                test_file = test_info["file"]
                test_path = PYTHON_TESTS_DIR / test_file
                status = "✓" if test_path.exists() else "✗ (not found)"
                print(f"  {kernel:15} -> {test_file} {status}")
            else:
                print(f"  {kernel:15} -> No test defined")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run functional tests for LLK kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests for sigmoid
    python scripts/run_functional_test.py sigmoid

    # Quick smoke test
    python scripts/run_functional_test.py exp --quick

    # Test specific format
    python scripts/run_functional_test.py relu --format Float16_b

    # List available tests
    python scripts/run_functional_test.py --list

    # Verbose output
    python scripts/run_functional_test.py tanh -v
""",
    )

    parser.add_argument("kernel", nargs="?", help="Kernel name to test")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available tests"
    )
    parser.add_argument("--format", "-f", help="Specific data format (e.g., Float16_b)")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick smoke test (minimal cases)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Run an arbitrary test file (for phase tests not in the kernel registry)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument("extra", nargs="*", help="Extra pytest arguments")

    args = parser.parse_args()

    if args.list:
        list_available_tests()
        return 0

    # Handle --test-file: run an arbitrary test file directly (for phase tests)
    if args.test_file:
        test_file_path = Path(args.test_file)
        if not test_file_path.exists():
            print(f"Error: Test file not found: {test_file_path}")
            return 1

        kernel_label = args.kernel or test_file_path.stem

        if args.dry_run:
            print(f"Kernel: {kernel_label}")
            print(f"Test file: {test_file_path} (exists)")
            print(f"Quick mode: {args.quick}")
            print(f"Format: {args.format or 'all'}")
            return 0

        # Build pytest command for the arbitrary test file
        cmd = ["pytest", str(test_file_path)]
        if args.verbose:
            cmd.append("-v")
        filter_parts = []
        if args.format:
            filter_parts.append(args.format)
        if args.quick:
            filter_parts.append("not 64")
        if filter_parts:
            cmd.extend(["-k", " and ".join(filter_parts)])

        chip_arch = os.environ.get("CHIP_ARCH", "quasar").lower()
        if chip_arch == "quasar":
            cmd.append("--run-simulator")
            sim_port = os.environ.get("LLK_SIMULATOR_PORT", "5556")
            cmd.extend(["--port", sim_port])

        cmd.extend(["--timeout", "600"])
        if args.extra:
            cmd.extend(args.extra)

        # Determine working directory: use file's parent if outside PYTHON_TESTS_DIR
        cwd = (
            test_file_path.parent
            if not str(test_file_path).startswith(str(PYTHON_TESTS_DIR))
            else PYTHON_TESTS_DIR
        )

        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {cwd}")
        print("-" * 60)

        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=600
            )
            output = proc.stdout + proc.stderr
            total, passed, failed = _parse_pytest_output(output)
            result = FunctionalTestResult(
                kernel=kernel_label,
                test_file=str(test_file_path),
                passed=proc.returncode == 0,
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                output=output,
                return_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            result = FunctionalTestResult(
                kernel=kernel_label,
                test_file=str(test_file_path),
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                output="Tests timed out after 10 minutes",
                return_code=-1,
            )

        print(f"\n{'='*60}")
        print(result.summary())
        print(f"{'='*60}")
        if not result.passed and args.verbose:
            print("\nFull output:")
            print(result.output)
        return 0 if result.passed else 1

    if not args.kernel:
        parser.print_help()
        print("\nError: Please specify a kernel name or use --list")
        return 1

    if args.dry_run:
        test_info = get_test_info(args.kernel)
        if test_info:
            test_path = PYTHON_TESTS_DIR / test_info["file"]
            exists = "exists" if test_path.exists() else "NOT FOUND"
            print(f"Kernel: {args.kernel}")
            print(f"Test file: {test_info['file']} ({exists})")
            print(f"Filter: {test_info.get('filter', 'None')}")
            print(f"Quick mode: {args.quick}")
            print(f"Format: {args.format or 'all'}")
        else:
            print(f"No test mapping found for '{args.kernel}'")
            # Try runtime discovery
            pattern = f"test_*{args.kernel}*_quasar.py"
            matches = list((PYTHON_TESTS_DIR / "quasar").glob(pattern))
            if matches:
                print(f"Discovered via glob: {[m.name for m in matches]}")
            else:
                print(f"No test files matching '{pattern}' in quasar/")
        return 0

    result = run_functional_test(
        args.kernel,
        data_format=args.format,
        quick=args.quick,
        verbose=args.verbose,
        extra_args=args.extra,
    )

    print(f"\n{'='*60}")
    print(result.summary())
    print(f"{'='*60}")

    if not result.passed and args.verbose:
        print("\nFull output:")
        print(result.output)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
