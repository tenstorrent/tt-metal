#!/usr/bin/env python3
"""
Galaxy-specific benchmark runner for telemetry performance testing.

This script is designed for safe execution on 32-device Blackhole Galaxy systems,
with built-in safeguards to prevent system-wide corruption.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Import Galaxy configuration
from galaxy_config import (
    GALAXY_PATHS,
    DEVICE_COUNTS_GALAXY,
    POLLING_FREQUENCIES_GALAXY,
    CCL_OPERATIONS_GALAXY,
    TEST_PRESETS,
    get_preset,
    setup_galaxy_environment,
)

# Import test modules
from telemetry_benchmark_utils import save_results_json
import comprehensive_single_device_benchmark as single_device
import comprehensive_multi_device_benchmark as multi_device


class GalaxyBenchmarkRunner:
    """Orchestrates benchmark execution on Galaxy systems."""

    def __init__(self, preset: str = "safe_comprehensive", output_dir: str = "/tmp/galaxy_telemetry"):
        self.preset_name = preset
        self.preset = get_preset(preset)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def run_single_device_tests(self) -> Dict[str, Any]:
        """Run single-device tests with reduced configuration."""
        print("\n" + "=" * 80)
        print("SINGLE-DEVICE TESTS")
        print("=" * 80)

        # Override configurations for single-device tests
        single_device.POLLING_FREQUENCIES_REDUCED = self.preset["polling_frequencies"]
        single_device.N_SAMPLES = self.preset["iterations"]

        # Use reduced tensor sizes for safety
        if self.preset_name == "quick_validation":
            single_device.SINGLE_DEVICE_SHAPES_REDUCED = [self.preset["tensor_size"]]

        print(f"Running single-device tests with {len(self.preset['polling_frequencies'])} frequencies")
        print(f"Iterations per config: {self.preset['iterations']}")

        # Run the benchmark
        try:
            # Import after configuration override
            from comprehensive_single_device_benchmark import main as single_main

            # Capture results
            results_file = self.output_dir / f"single_device_results_{self.timestamp}.json"
            old_argv = sys.argv
            sys.argv = ["comprehensive_single_device_benchmark.py", "reduced"]

            result = single_main()

            sys.argv = old_argv

            return {
                "status": "success",
                "results_file": str(results_file),
            }

        except Exception as e:
            print(f"Single-device test failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def run_multi_device_tests(self) -> Dict[str, Any]:
        """Run multi-device CCL tests with safety limits."""
        print("\n" + "=" * 80)
        print("MULTI-DEVICE CCL TESTS")
        print("=" * 80)

        # Override configurations for multi-device tests
        multi_device.DEVICE_COUNTS_REDUCED = self.preset["device_counts"]
        multi_device.POLLING_FREQUENCIES_REDUCED = self.preset["polling_frequencies"]
        multi_device.CCL_OPERATIONS_REDUCED = self.preset["ccl_operations"]
        multi_device.TENSOR_SIZE = self.preset["tensor_size"]
        multi_device.N_SAMPLES = self.preset["iterations"]

        print(f"Device counts: {self.preset['device_counts']}")
        print(f"CCL operations: {self.preset['ccl_operations']}")
        print(f"Polling frequencies: {len(self.preset['polling_frequencies'])} frequencies")
        print(f"Iterations per config: {self.preset['iterations']}")

        # Safety check for large device counts
        max_devices = max(self.preset["device_counts"])
        if max_devices > 8:
            print(f"\n⚠️  WARNING: Testing with {max_devices} devices")
            print("This may impact other users on the Galaxy system")
            response = input("Continue? (y/n): ")
            if response.lower() != "y":
                return {"status": "skipped", "reason": "User cancelled due to device count"}

        try:
            # Import after configuration override
            from comprehensive_multi_device_benchmark import main as multi_main

            # Capture results
            results_file = self.output_dir / f"multi_device_results_{self.timestamp}.json"
            old_argv = sys.argv
            sys.argv = ["comprehensive_multi_device_benchmark.py", "reduced"]

            result = multi_main()

            sys.argv = old_argv

            return {
                "status": "success",
                "results_file": str(results_file),
            }

        except Exception as e:
            print(f"Multi-device test failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def run_mmio_validation(self) -> Dict[str, Any]:
        """Run MMIO-only validation test."""
        print("\n" + "=" * 80)
        print("MMIO-ONLY VALIDATION TEST")
        print("=" * 80)

        try:
            from validate_mmio_only import main as mmio_main

            results_file = self.output_dir / f"mmio_validation_{self.timestamp}.json"
            old_argv = sys.argv
            sys.argv = ["validate_mmio_only.py"]

            result = mmio_main()

            sys.argv = old_argv

            return {
                "status": "success",
                "results_file": str(results_file),
            }

        except Exception as e:
            print(f"MMIO validation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def generate_summary(self, results: Dict[str, Any]):
        """Generate summary report of all tests."""
        print("\n" + "=" * 80)
        print("GALAXY BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"\nPreset used: {self.preset_name}")
        print(f"Description: {self.preset['description']}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nTest Results:")

        for test_name, test_result in results.items():
            status = test_result.get("status", "unknown")
            symbol = "✅" if status == "success" else "❌" if status == "failed" else "⏭️"
            print(f"  {symbol} {test_name}: {status}")
            if status == "failed":
                print(f"      Error: {test_result.get('error', 'Unknown error')}")
            elif status == "success" and "results_file" in test_result:
                print(f"      Results: {test_result['results_file']}")

        # Save summary
        summary_file = self.output_dir / f"galaxy_benchmark_summary_{self.timestamp}.json"
        save_results_json(
            summary_file,
            {
                "preset": self.preset_name,
                "configuration": self.preset,
                "timestamp": self.timestamp,
                "results": results,
            },
        )

        print(f"\nSummary saved to: {summary_file}")

    def run(self, skip_single: bool = False, skip_multi: bool = False, skip_mmio: bool = False):
        """Run the complete benchmark suite."""
        setup_galaxy_environment()

        print("\n" + "=" * 80)
        print("BLACKHOLE GALAXY TELEMETRY BENCHMARK")
        print("=" * 80)
        print(f"Preset: {self.preset_name}")
        print(f"Description: {self.preset['description']}")
        print("=" * 80)

        results = {}

        # MMIO validation
        if not skip_mmio:
            results["mmio_validation"] = self.run_mmio_validation()
            time.sleep(5)  # Brief pause between tests

        # Single-device tests
        if not skip_single:
            results["single_device"] = self.run_single_device_tests()
            time.sleep(5)

        # Multi-device tests
        if not skip_multi:
            results["multi_device"] = self.run_multi_device_tests()

        # Generate summary
        self.generate_summary(results)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Galaxy-specific telemetry benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  quick_validation    - 15-minute test with 2-4 devices
  safe_comprehensive  - 2-hour test with up to 8 devices (default)
  full_galaxy        - Full 32-device test (use with caution)

Examples:
  # Quick validation
  %(prog)s --preset quick_validation

  # Comprehensive test (default)
  %(prog)s

  # Single-device tests only
  %(prog)s --skip-multi --skip-mmio

  # Multi-device tests only
  %(prog)s --skip-single --skip-mmio
        """,
    )

    parser.add_argument(
        "--preset", choices=list(TEST_PRESETS.keys()), default="safe_comprehensive", help="Test configuration preset"
    )

    parser.add_argument("--output", default="/tmp/galaxy_telemetry", help="Output directory for results")

    parser.add_argument("--skip-single", action="store_true", help="Skip single-device tests")

    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-device tests")

    parser.add_argument("--skip-mmio", action="store_true", help="Skip MMIO validation test")

    args = parser.parse_args()

    runner = GalaxyBenchmarkRunner(preset=args.preset, output_dir=args.output)
    results = runner.run(skip_single=args.skip_single, skip_multi=args.skip_multi, skip_mmio=args.skip_mmio)

    # Return 0 if all non-skipped tests passed
    all_passed = all(r.get("status") in ["success", "skipped"] for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
