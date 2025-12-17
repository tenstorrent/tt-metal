#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Run all PCC tests for TTNN PI0 Reference Implementation.

This script runs all PCC (Pearson Correlation Coefficient) tests
to validate that TTNN implementations match PyTorch references.

Usage:
    python run_all_pcc.py [--verbose] [--module MODULE]

Examples:
    python run_all_pcc.py                    # Run all tests
    python run_all_pcc.py --module gemma     # Run only gemma tests
    python run_all_pcc.py --verbose          # Verbose output
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_pcc_common():
    """Run common module PCC tests."""
    from tests.pcc.pcc_common import run_pcc_common_tests
    run_pcc_common_tests()


def run_pcc_attention():
    """Run attention module PCC tests."""
    from tests.pcc.pcc_attention import run_pcc_attention_tests
    run_pcc_attention_tests()


def run_pcc_suffix():
    """Run suffix module PCC tests."""
    from tests.pcc.pcc_suffix import run_pcc_suffix_tests
    run_pcc_suffix_tests()


def run_pcc_prefix():
    """Run prefix module PCC tests."""
    from tests.pcc.pcc_prefix import run_pcc_prefix_tests
    run_pcc_prefix_tests()


def run_pcc_gemma():
    """Run gemma module PCC tests."""
    from tests.pcc.pcc_gemma import run_pcc_gemma_tests
    run_pcc_gemma_tests()


def run_pcc_siglip():
    """Run siglip module PCC tests."""
    from tests.pcc.pcc_siglip import run_pcc_siglip_tests
    run_pcc_siglip_tests()


def run_pcc_paligemma():
    """Run paligemma module PCC tests."""
    from tests.pcc.pcc_paligemma import run_pcc_paligemma_tests
    run_pcc_paligemma_tests()


def run_pcc_denoise():
    """Run denoise module PCC tests."""
    from tests.pcc.pcc_denoise import run_pcc_denoise_tests
    run_pcc_denoise_tests()


def run_pcc_pi0():
    """Run pi0 module PCC tests."""
    from tests.pcc.pcc_pi0 import run_pcc_pi0_tests
    run_pcc_pi0_tests()


# Map module names to test functions
MODULE_TESTS = {
    "common": run_pcc_common,
    "attention": run_pcc_attention,
    "suffix": run_pcc_suffix,
    "prefix": run_pcc_prefix,
    "gemma": run_pcc_gemma,
    "siglip": run_pcc_siglip,
    "paligemma": run_pcc_paligemma,
    "denoise": run_pcc_denoise,
    "pi0": run_pcc_pi0,
}


def run_all_tests():
    """Run all PCC tests."""
    print("=" * 70)
    print("  TTNN PI0 Reference Implementation - PCC Test Suite")
    print("=" * 70)
    print()
    
    start_time = time.time()
    passed = 0
    failed = 0
    
    for module_name, test_fn in MODULE_TESTS.items():
        print(f"\n{'=' * 70}")
        print(f"  Testing: ttnn_{module_name}.py")
        print("=" * 70)
        
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {module_name}")
            print(f"  Error: {e}")
            failed += 1
    
    elapsed = time.time() - start_time
    
    print("\n")
    print("=" * 70)
    print("  PCC Test Summary")
    print("=" * 70)
    print(f"  Total modules: {len(MODULE_TESTS)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.2f}s")
    print("=" * 70)
    
    if failed > 0:
        print("\n✗ Some tests failed!")
        return 1
    else:
        print("\n✓ All PCC tests passed!")
        return 0


def run_single_module(module_name: str):
    """Run PCC tests for a single module."""
    if module_name not in MODULE_TESTS:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(MODULE_TESTS.keys())}")
        return 1
    
    try:
        MODULE_TESTS[module_name]()
        return 0
    except Exception as e:
        print(f"\n✗ FAILED: {module_name}")
        print(f"  Error: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run PCC tests for TTNN PI0 Reference Implementation"
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=list(MODULE_TESTS.keys()),
        help="Run tests for a specific module only",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test modules",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available PCC test modules:")
        for name in MODULE_TESTS.keys():
            print(f"  - {name} (ttnn_{name}.py)")
        return 0
    
    if args.module:
        return run_single_module(args.module)
    else:
        return run_all_tests()


if __name__ == "__main__":
    sys.exit(main())

