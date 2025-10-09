#!/usr/bin/env python3
"""
Benchmark buffer allocation performance with and without tracking enabled.
This helps quantify the overhead of the allocation tracking system.
"""

import os
import sys
import time
import subprocess


def run_benchmark(tracking_enabled: bool, num_iterations: int = 1000):
    """Run a benchmark with tracking enabled or disabled."""
    env = os.environ.copy()
    env["TT_ALLOC_TRACKING_ENABLED"] = "1" if tracking_enabled else "0"

    # Simple test that creates and destroys buffers
    test_code = f"""
import time
import sys
sys.path.insert(0, '/workspace/tt-metal-apv/build/python_env/lib/python3.10/site-packages')

try:
    import tt_metal as ttm

    # Initialize device
    device_id = 0
    device = ttm.CreateDevice(device_id)

    start = time.perf_counter()

    # Create and destroy buffers repeatedly
    for i in range({num_iterations}):
        # Allocate a buffer
        buffer = ttm.Buffer(
            device,
            4096,  # 4KB buffer
            4096,  # page size
            ttm.BufferType.DRAM
        )
        # Let it go out of scope (destructor deallocates)
        del buffer

    elapsed = time.perf_counter() - start

    ttm.CloseDevice(device)

    print(f"{{elapsed:.6f}}")

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(["python3", "-c", test_code], env=env, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"Error running benchmark: {result.stderr}")
            return None

        return float(result.stdout.strip())

    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return None
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None


def main():
    print("=" * 60)
    print("Buffer Allocation Tracking Performance Benchmark")
    print("=" * 60)
    print()

    # Check if server is running
    if os.path.exists("/tmp/tt_allocation_server.sock"):
        print("✓ Allocation server detected")
    else:
        print("⚠ Allocation server not running")
        print("  Start with: ./allocation_server_poc &")
        print()

    num_iterations = 1000
    num_runs = 3

    print(f"Test configuration:")
    print(f"  Iterations per run: {num_iterations}")
    print(f"  Number of runs: {num_runs}")
    print()

    # Warm-up
    print("Warming up...")
    run_benchmark(False, 100)

    # Benchmark without tracking
    print("\n" + "=" * 60)
    print("Testing WITHOUT tracking (baseline)")
    print("=" * 60)

    times_without = []
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
        elapsed = run_benchmark(False, num_iterations)
        if elapsed:
            times_without.append(elapsed)
            print(f"{elapsed:.6f}s ({num_iterations/elapsed:.0f} ops/sec)")
        else:
            print("FAILED")

    # Benchmark with tracking
    print("\n" + "=" * 60)
    print("Testing WITH tracking enabled")
    print("=" * 60)

    times_with = []
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
        elapsed = run_benchmark(True, num_iterations)
        if elapsed:
            times_with.append(elapsed)
            print(f"{elapsed:.6f}s ({num_iterations/elapsed:.0f} ops/sec)")
        else:
            print("FAILED")

    # Calculate results
    if times_without and times_with:
        avg_without = sum(times_without) / len(times_without)
        avg_with = sum(times_with) / len(times_with)
        overhead = ((avg_with - avg_without) / avg_without) * 100

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Average time WITHOUT tracking: {avg_without:.6f}s")
        print(f"Average time WITH tracking:    {avg_with:.6f}s")
        print(f"Overhead:                      {overhead:.2f}%")
        print(f"Per-operation overhead:        {(avg_with - avg_without) / num_iterations * 1e6:.2f} μs")
        print()

        if overhead < 1:
            print("✅ Excellent! Overhead is negligible (< 1%)")
        elif overhead < 2:
            print("✓ Good! Overhead is acceptable (< 2%)")
        elif overhead < 5:
            print("⚠ Moderate overhead (2-5%). Consider optimizations if critical.")
        else:
            print("⚠ High overhead (> 5%). Consider async queue or conditional tracking.")
        print()
    else:
        print("\n❌ Benchmark failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
