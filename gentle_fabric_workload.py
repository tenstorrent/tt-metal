#!/usr/bin/env python3
"""
Gentle fabric workload generator for bandwidth telemetry validation.

This script runs minimal fabric operations in a loop with delays,
allowing telemetry server to safely read metrics without device contention.

Usage:
    python3 gentle_fabric_workload.py [--duration 60] [--interval 2]
"""

import subprocess
import time
import sys
import argparse
from datetime import datetime


def run_single_transfer(test_binary: str, test_filter: str) -> bool:
    """Run a single fabric transfer test"""
    try:
        # Run a single quick fabric test
        result = subprocess.run(
            [test_binary, f"--gtest_filter={test_filter}"],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                "TT_METAL_FABRIC_TELEMETRY": "1",
                "TT_METAL_SLOW_DISPATCH_MODE": "1",
                **dict(subprocess.os.environ),  # Preserve existing env
            },
        )

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("  ⚠️  Transfer timed out", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run gentle fabric workload for telemetry validation")
    parser.add_argument("--duration", type=int, default=60, help="How long to run workload in seconds (default: 60)")
    parser.add_argument("--interval", type=float, default=3.0, help="Delay between transfers in seconds (default: 3.0)")
    parser.add_argument(
        "--test-binary", default="./build/test/tt_metal/tt_fabric/fabric_unit_tests", help="Path to fabric test binary"
    )
    parser.add_argument(
        "--test-filter", default="Fabric2DFixture.TestUnicastConnAPI", help="GTest filter for specific test"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GENTLE FABRIC WORKLOAD GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Duration: {args.duration}s")
    print(f"  Transfer interval: {args.interval}s")
    print(f"  Test: {args.test_filter}")
    print(f"\n⚠️  Make sure tt_telemetry_server is running in another terminal!")
    print(f"\nStarting in 3 seconds...")
    time.sleep(3)

    start_time = time.time()
    transfer_count = 0
    success_count = 0

    try:
        while (time.time() - start_time) < args.duration:
            transfer_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            print(f"\n[{timestamp}] Transfer #{transfer_count}...", end="", flush=True)

            success = run_single_transfer(args.test_binary, args.test_filter)

            if success:
                success_count += 1
                print(" ✓")
            else:
                print(" ✗ (failed)")

            # Wait between transfers to allow telemetry to read
            remaining = args.duration - (time.time() - start_time)
            if remaining > args.interval:
                print(f"  Waiting {args.interval}s before next transfer...")
                time.sleep(args.interval)
            else:
                break

    except KeyboardInterrupt:
        print("\n\nWorkload stopped by user")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Duration: {elapsed:.1f}s")
    print(f"Transfers: {transfer_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {transfer_count - success_count}")
    print(f"Success rate: {100.0 * success_count / transfer_count if transfer_count > 0 else 0:.1f}%")

    if success_count > 0:
        print("\n✓ Workload generated fabric traffic")
        print("  Check telemetry server logs for bandwidth measurements")
        return 0
    else:
        print("\n✗ No successful transfers")
        return 1


if __name__ == "__main__":
    sys.exit(main())
