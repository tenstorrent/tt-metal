#!/usr/bin/env python3
"""
Minimal telemetry test for Blackhole Galaxy - API compatible version.
This script avoids ttnn API issues by using the most basic operations.
"""

import subprocess
import time
import sys
import os
import json
from pathlib import Path
import statistics


def find_telemetry_binary():
    """Find the telemetry binary."""
    telemetry_path = "/localdev/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
    if Path(telemetry_path).exists():
        print(f"Found telemetry binary at: {telemetry_path}")
        return telemetry_path
    else:
        print(f"Telemetry binary not found at: {telemetry_path}")
        return None


def run_device_reset():
    """Reset devices to clean state."""
    print("Resetting devices...")
    subprocess.run(["tt-smi", "-r", "0", "1", "2", "3"], capture_output=True)
    time.sleep(5)
    print("Devices reset")


def check_ttnn_api():
    """Check what's available in ttnn module."""
    test_script = """
import ttnn
import sys

# Try the CreateDevice API found in conftest.py
try:
    device = ttnn.CreateDevice(device_id=0)
    print("SUCCESS: ttnn.CreateDevice(device_id=0) works")
    ttnn.close_device(device)
except Exception as e:
    print(f"ERROR: ttnn.CreateDevice failed: {e}")
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True)
    print("TTNN API Check:")
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0


def run_minimal_test(telemetry_enabled=False, polling_interval_ms=1000):
    """Run a minimal test that works with various ttnn versions."""
    test_script = """
import time
import statistics
import ttnn
import torch

# Use the CreateDevice API from conftest.py
device = ttnn.CreateDevice(device_id=0)
print("Device opened successfully")

# Run simple operations
shape = (1, 1, 32, 32)  # Small tensor to avoid memory issues
times = []

for i in range(5):
    start = time.perf_counter()

    # Create torch tensor
    a = torch.ones(shape, dtype=torch.bfloat16)

    # Transfer to device using the device parameter
    tt_a = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Simple operation
    tt_b = ttnn.add(tt_a, tt_a)

    # Transfer back
    result = ttnn.to_torch(tt_b)

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

# Close device
ttnn.close_device(device)

print(f"RESULT_JSON:{{'mean_ms': {statistics.mean(times)}, 'stdev_ms': {statistics.stdev(times) if len(times) > 1 else 0}}}")
"""

    telemetry_proc = None

    try:
        # Start telemetry if enabled
        if telemetry_enabled:
            telemetry_bin = find_telemetry_binary()
            if not telemetry_bin:
                print("WARNING: Telemetry binary not found, running without telemetry")
                telemetry_enabled = False
            else:
                cmd = [
                    telemetry_bin,
                    "--polling-interval",
                    f"{polling_interval_ms}ms",
                    "--port",
                    "7070",
                    "--fsd",
                    "/localdev/kkfernandez/fsd.textproto",
                ]

                env = os.environ.copy()
                env["TT_METAL_HOME"] = "/localdev/kkfernandez/tt-metal"

                print(f"Starting telemetry with {polling_interval_ms}ms polling...")
                telemetry_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                time.sleep(3)  # Let telemetry initialize

        # Run test
        print(f"Running test (telemetry={'ON' if telemetry_enabled else 'OFF'})...")
        result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=30)

        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        # Parse result
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                json_str = line.replace("RESULT_JSON:", "")
                return eval(json_str)

        return None

    finally:
        if telemetry_proc:
            telemetry_proc.terminate()
            telemetry_proc.wait(timeout=5)


def main():
    """Run minimal telemetry impact test."""
    print("\n" + "=" * 80)
    print("MINIMAL GALAXY TELEMETRY TEST")
    print("=" * 80)

    # First check the API
    print("\nChecking TTNN API compatibility...")
    if not check_ttnn_api():
        print("ERROR: TTNN API check failed. Please check your environment.")
        return 1

    # Reset devices first
    run_device_reset()

    results = {}

    # Test configurations
    polling_intervals = [("1s", 1000), ("100ms", 100), ("10ms", 10)]

    print("\n--- TESTING ---")

    # Baseline (no telemetry)
    print("\nRunning baseline (no telemetry)...")
    baseline = run_minimal_test(telemetry_enabled=False)
    if baseline:
        results["baseline"] = baseline
        print(f"Baseline: {baseline['mean_ms']:.2f} ± {baseline['stdev_ms']:.2f} ms")

    # With telemetry at different frequencies
    for name, interval_ms in polling_intervals:
        print(f"\nRunning with telemetry at {name}...")
        result = run_minimal_test(telemetry_enabled=True, polling_interval_ms=interval_ms)
        if result:
            results[name] = result
            print(f"With telemetry ({name}): {result['mean_ms']:.2f} ± {result['stdev_ms']:.2f} ms")

            if baseline:
                overhead = ((result["mean_ms"] - baseline["mean_ms"]) / baseline["mean_ms"]) * 100
                print(f"Overhead: {overhead:+.2f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if "baseline" in results:
        baseline_ms = results["baseline"]["mean_ms"]
        for freq, data in results.items():
            if freq != "baseline":
                overhead = ((data["mean_ms"] - baseline_ms) / baseline_ms) * 100
                print(f"  {freq:6s}: {overhead:+6.2f}% overhead")

    # Save results
    output_file = f"/tmp/galaxy_minimal_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
