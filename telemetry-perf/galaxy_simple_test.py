#!/usr/bin/env python3
"""
Simplified telemetry test for Blackhole Galaxy that avoids import issues.
This script tests telemetry impact directly without complex dependencies.
"""

import subprocess
import time
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics


def find_telemetry_binary():
    """Find the telemetry binary in various locations."""
    possible_paths = [
        "/localdev/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server",
        "/localdev/kkfernandez/tt-telemetry/build/bin/tt_telemetry_server",
        "/localdev/kkfernandez/tt-telemetry/tt_telemetry_server",
        "/usr/local/bin/tt_telemetry_server",
        "/usr/bin/tt_telemetry_server",
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"Found telemetry binary at: {path}")
            return path

    # Try to build it
    print("Telemetry binary not found. Attempting to build...")
    telemetry_dir = Path("/localdev/kkfernandez/tt-telemetry")
    if telemetry_dir.exists():
        result = subprocess.run(["./build.sh", "Release"], cwd=telemetry_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print("Built telemetry successfully")
            return "/localdev/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
        else:
            print(f"Build failed: {result.stderr}")

    return None


def run_simple_single_device_test(telemetry_enabled=False, polling_interval_ms=1000):
    """Run a simple single-device test using ttnn."""
    test_script = """
import ttnn
import torch
import time

# Open device 0
device = ttnn.open_device(device_id=0)

# Create tensors
shape = (1, 1, 1024, 1024)
a = torch.randn(shape, dtype=torch.bfloat16)

times = []
for i in range(10):
    start = time.perf_counter()

    # Transfer to device
    tt_a = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Simple operation
    tt_b = ttnn.add(tt_a, tt_a)
    tt_c = ttnn.multiply(tt_b, tt_a)

    # Get result
    result = ttnn.to_torch(tt_c)

    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

    # Cleanup
    ttnn.deallocate(tt_a)
    ttnn.deallocate(tt_b)
    ttnn.deallocate(tt_c)

ttnn.close_device(device)

import statistics
print(f"RESULT_JSON:{{'mean_ms': {statistics.mean(times)}, 'stdev_ms': {statistics.stdev(times)}}}")
"""

    telemetry_proc = None

    try:
        # Start telemetry if enabled
        if telemetry_enabled:
            telemetry_bin = find_telemetry_binary()
            if not telemetry_bin:
                print("WARNING: Could not find or build telemetry binary, running without telemetry")
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

                print(f"Starting telemetry with polling interval {polling_interval_ms}ms...")
                telemetry_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                time.sleep(5)  # Let telemetry initialize

        # Run test
        print(f"Running test (telemetry={'ON' if telemetry_enabled else 'OFF'})...")
        result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"Test failed: {result.stderr}")
            return None

        # Parse result
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                json_str = line.replace("RESULT_JSON:", "")
                return eval(json_str)  # Safe here as we control the output

        return None

    finally:
        if telemetry_proc:
            telemetry_proc.terminate()
            telemetry_proc.wait(timeout=5)


def run_simple_multi_device_test(num_devices=2, telemetry_enabled=False, polling_interval_ms=1000):
    """Run a simple multi-device test."""
    test_script = f"""
import ttnn
import torch
import time

# Create mesh device with {num_devices} devices
if {num_devices} == 2:
    mesh_shape = ttnn.MeshShape(1, 2)
elif {num_devices} == 4:
    mesh_shape = ttnn.MeshShape(2, 2)
elif {num_devices} == 8:
    mesh_shape = ttnn.MeshShape(2, 4)
else:
    print(f"Unsupported device count: {num_devices}")
    exit(1)

try:
    mesh = ttnn.open_mesh_device(mesh_shape)

    # Create tensor
    shape = (1, 1, 1024, 1024)

    times = []
    for i in range(5):
        start = time.perf_counter()

        # Create and distribute tensor
        input_tensor = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            layout=ttnn.TILE_LAYOUT
        )

        # CCL operation (AllGather)
        output = ttnn.all_gather(input_tensor, dim=0, num_links=1)

        # Synchronize
        ttnn.synchronize_device(mesh)

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

        # Cleanup
        del output
        del input_tensor

    ttnn.close_mesh_device(mesh)

    import statistics
    print(f"RESULT_JSON:{{'mean_ms': {{statistics.mean(times)}}, 'stdev_ms': {{statistics.stdev(times)}}}}")

except Exception as e:
    print(f"Error: {{e}}")
    # Fall back to single device if mesh fails
    print("Falling back to single device test")
"""

    telemetry_proc = None

    try:
        # Start telemetry if enabled
        if telemetry_enabled:
            telemetry_bin = find_telemetry_binary()
            if not telemetry_bin:
                print("WARNING: Could not find or build telemetry binary, running without telemetry")
                telemetry_enabled = False
            else:
                cmd = [
                    telemetry_bin,
                    "--polling-interval",
                    f"{polling_interval_ms}ms",
                    "--port",
                    "7071",
                    "--fsd",
                    "/localdev/kkfernandez/fsd.textproto",
                ]

                env = os.environ.copy()
                env["TT_METAL_HOME"] = "/localdev/kkfernandez/tt-metal"

                print(f"Starting telemetry with polling interval {polling_interval_ms}ms...")
                telemetry_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                time.sleep(5)  # Let telemetry initialize

        # Run test
        print(f"Running {num_devices}-device test (telemetry={'ON' if telemetry_enabled else 'OFF'})...")
        result = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"Test output: {result.stdout}")
            print(f"Test errors: {result.stderr}")
            return None

        # Parse result
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                json_str = line.replace("RESULT_JSON:", "")
                return eval(json_str)  # Safe here as we control the output

        return None

    finally:
        if telemetry_proc:
            telemetry_proc.terminate()
            telemetry_proc.wait(timeout=5)


def main():
    """Run simplified telemetry impact tests."""
    print("\n" + "=" * 80)
    print("SIMPLIFIED GALAXY TELEMETRY TEST")
    print("=" * 80)

    results = {"single_device": {}, "multi_device": {}}

    # Test configurations
    polling_intervals = [("1s", 1000), ("10ms", 10), ("100us", 0.1)]

    print("\n--- SINGLE DEVICE TESTS ---")

    # Baseline (no telemetry)
    print("\nRunning baseline (no telemetry)...")
    baseline = run_simple_single_device_test(telemetry_enabled=False)
    if baseline:
        results["single_device"]["baseline"] = baseline
        print(f"Baseline: {baseline['mean_ms']:.2f} ± {baseline['stdev_ms']:.2f} ms")

    # With telemetry at different frequencies
    for name, interval_ms in polling_intervals:
        print(f"\nRunning with telemetry at {name}...")
        result = run_simple_single_device_test(telemetry_enabled=True, polling_interval_ms=interval_ms)
        if result:
            results["single_device"][name] = result
            print(f"With telemetry ({name}): {result['mean_ms']:.2f} ± {result['stdev_ms']:.2f} ms")

            if baseline:
                overhead = ((result["mean_ms"] - baseline["mean_ms"]) / baseline["mean_ms"]) * 100
                print(f"Overhead: {overhead:+.2f}%")

    print("\n--- MULTI-DEVICE TESTS (2 devices) ---")

    # Baseline (no telemetry)
    print("\nRunning baseline (no telemetry)...")
    baseline = run_simple_multi_device_test(num_devices=2, telemetry_enabled=False)
    if baseline:
        results["multi_device"]["baseline"] = baseline
        print(f"Baseline: {baseline['mean_ms']:.2f} ± {baseline['stdev_ms']:.2f} ms")

    # With telemetry at different frequencies
    for name, interval_ms in polling_intervals:
        print(f"\nRunning with telemetry at {name}...")
        result = run_simple_multi_device_test(num_devices=2, telemetry_enabled=True, polling_interval_ms=interval_ms)
        if result:
            results["multi_device"][name] = result
            print(f"With telemetry ({name}): {result['mean_ms']:.2f} ± {result['stdev_ms']:.2f} ms")

            if baseline:
                overhead = ((result["mean_ms"] - baseline["mean_ms"]) / baseline["mean_ms"]) * 100
                print(f"Overhead: {overhead:+.2f}%")

    # Save results
    output_file = f"/tmp/galaxy_simple_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- RESULTS SAVED TO: {output_file} ---")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nSingle Device Impact:")
    if "baseline" in results["single_device"]:
        baseline_ms = results["single_device"]["baseline"]["mean_ms"]
        for freq, data in results["single_device"].items():
            if freq != "baseline":
                overhead = ((data["mean_ms"] - baseline_ms) / baseline_ms) * 100
                print(f"  {freq:6s}: {overhead:+6.2f}% overhead")

    print("\nMulti-Device Impact (2 devices):")
    if "baseline" in results["multi_device"]:
        baseline_ms = results["multi_device"]["baseline"]["mean_ms"]
        for freq, data in results["multi_device"].items():
            if freq != "baseline":
                overhead = ((data["mean_ms"] - baseline_ms) / baseline_ms) * 100
                print(f"  {freq:6s}: {overhead:+6.2f}% overhead")


if __name__ == "__main__":
    main()
