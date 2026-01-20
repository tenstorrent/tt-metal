#!/usr/bin/env python3
"""
Fast Single-Device Telemetry Benchmark (No Device Resets)

Optimized for speed and reliability:
- Reuses single device across tests (no close/reopen between tests)
- No thermal cooldown (not needed for single device)
- No topology rediscovery (avoids Ethernet timeout issues)
"""

import sys
import time
from typing import Tuple
from pathlib import Path
import torch

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from telemetry_benchmark_utils import (
    TelemetryManager,
    save_results_json,
    calculate_impact_percentage,
    mann_whitney_u_test,
    cohens_d,
    parse_frequency_to_hz,
    analyze_monotonicity,
    apply_multiple_comparison_correction,
)

import ttnn

# Test configuration
TENSOR_SIZES = [
    (1, 1, 1024, 1024),  # 2MB - small
    (1, 1, 8192, 8192),  # 128MB - medium
    (1, 1, 17408, 17408),  # 578MB - large
]

POLLING_FREQUENCIES = ["60s", "1s", "100ms", "10ms", "1ms", "100us"]

OPERATIONS = ["matmul", "add", "to_memory_config"]

N_SAMPLES = 100
WARMUP_ITERS = 20


def run_matmul(device, shape):
    """Run matrix multiplication."""
    a = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    start = time.perf_counter()
    c = ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    end = time.perf_counter()

    del c, b, a
    return end - start


def run_add(device, shape):
    """Run element-wise addition."""
    a = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    start = time.perf_counter()
    c = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    end = time.perf_counter()

    del c, b, a
    return end - start


def run_to_memory_config(device, shape):
    """Run L1<->DRAM transfer."""
    # Create in L1
    a = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Transfer to DRAM
    start = time.perf_counter()
    b = ttnn.to_memory_config(a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    end = time.perf_counter()

    del b, a
    return end - start


def run_test(device, operation, shape, n_samples, warmup):
    """Run a single test configuration."""

    # Select operation
    if operation == "matmul":
        test_func = lambda: run_matmul(device, shape)
    elif operation == "add":
        test_func = lambda: run_add(device, shape)
    elif operation == "to_memory_config":
        test_func = lambda: run_to_memory_config(device, shape)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Warmup
    print(f"  Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        try:
            test_func()
        except:
            pass

    # Collect samples
    print(f"  Collecting {n_samples} samples...")
    samples = []
    for i in range(n_samples):
        try:
            t = test_func()
            samples.append(t)
            if (i + 1) % 20 == 0:
                print(f"    {i + 1}/{n_samples}")
        except Exception as e:
            print(f"    Sample {i} failed: {e}")

    return samples


def main():
    """Run fast single-device benchmark."""
    print("=" * 80)
    print("FAST SINGLE-DEVICE TELEMETRY BENCHMARK")
    print("=" * 80)
    print(f"\nTest Matrix:")
    print(f"  Operations: {len(OPERATIONS)}")
    print(f"  Tensor sizes: {len(TENSOR_SIZES)}")
    print(f"  Frequencies: {len(POLLING_FREQUENCIES)} + baseline")
    print(f"  Total configs: {len(OPERATIONS) * len(TENSOR_SIZES) * (1 + len(POLLING_FREQUENCIES))}")
    print(f"  Samples per config: {N_SAMPLES}")

    # Open device ONCE (reuse across all tests)
    print("\nOpening device 0...")
    device = ttnn.open_device(device_id=0)
    print("✓ Device opened")

    all_results = []

    config_num = 0
    total_configs = len(OPERATIONS) * len(TENSOR_SIZES) * (1 + len(POLLING_FREQUENCIES))

    # Run all tests
    for operation in OPERATIONS:
        for shape in TENSOR_SIZES:
            shape_str = f"{shape[2]}x{shape[3]}"

            # Baseline test
            config_num += 1
            print(f"\n[{config_num}/{total_configs}] {operation} {shape_str} BASELINE")

            baseline_samples = run_test(device, operation, shape, N_SAMPLES, WARMUP_ITERS)

            if baseline_samples:
                baseline_mean = sum(baseline_samples) / len(baseline_samples)
                print(f"  ✓ Baseline: {baseline_mean*1000:.2f}ms")

                all_results.append(
                    {
                        "operation": operation,
                        "shape": shape,
                        "telemetry": False,
                        "polling_interval": None,
                        "samples": baseline_samples,
                        "mean_ms": baseline_mean * 1000,
                    }
                )

            # Telemetry tests
            for freq in POLLING_FREQUENCIES:
                config_num += 1
                print(f"\n[{config_num}/{total_configs}] {operation} {shape_str} TELEMETRY {freq}")

                # Start telemetry
                try:
                    telemetry_proc = TelemetryManager.start_telemetry_validated(
                        freq, mmio_only=False, stabilization_sec=3.0
                    )
                    print(f"  ✓ Telemetry started")
                except Exception as e:
                    print(f"  ✗ Telemetry failed: {e}")
                    continue

                # Run test
                telemetry_samples = run_test(device, operation, shape, N_SAMPLES, WARMUP_ITERS)

                # Stop telemetry
                TelemetryManager.stop_telemetry(telemetry_proc)

                if telemetry_samples and baseline_samples:
                    telemetry_mean = sum(telemetry_samples) / len(telemetry_samples)
                    impact = calculate_impact_percentage(baseline_mean, telemetry_mean)
                    print(f"  ✓ Telemetry: {telemetry_mean*1000:.2f}ms ({impact:+.1f}%)")

                    all_results.append(
                        {
                            "operation": operation,
                            "shape": shape,
                            "telemetry": True,
                            "polling_interval": freq,
                            "samples": telemetry_samples,
                            "mean_ms": telemetry_mean * 1000,
                            "impact_percent": impact,
                        }
                    )

    # Close device
    print("\nClosing device...")
    ttnn.close_device(device)
    print("✓ Device closed")

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group by operation and shape
    for operation in OPERATIONS:
        for shape in TENSOR_SIZES:
            shape_str = f"{shape[2]}x{shape[3]}"

            # Find baseline
            baseline = next(
                (r for r in all_results if r["operation"] == operation and r["shape"] == shape and not r["telemetry"]),
                None,
            )

            if not baseline:
                continue

            # Find telemetry results
            tel_results = [
                r for r in all_results if r["operation"] == operation and r["shape"] == shape and r["telemetry"]
            ]

            if not tel_results:
                continue

            print(f"\n{operation} {shape_str}:")
            print(f"  Baseline: {baseline['mean_ms']:.2f}ms")

            for tr in sorted(tel_results, key=lambda x: parse_frequency_to_hz(x["polling_interval"])):
                freq = tr["polling_interval"]
                impact = tr["impact_percent"]
                marker = "**" if abs(impact) >= 5.0 else ""
                print(f"    {freq:>6s}: {impact:+6.2f}% {marker}")

    # Save results
    save_results_json(all_results, "/tmp/fast_single_device_results.json")
    print(f"\nResults saved to: /tmp/fast_single_device_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
