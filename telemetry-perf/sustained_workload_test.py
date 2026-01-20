#!/usr/bin/env python3
"""
Sustained Workload Test for Telemetry

Runs CCL operations continuously for extended periods to test:
  - Performance drift over time
  - Thermal throttling effects
  - Telemetry stability under sustained load
  - Accumulation of telemetry overhead

Tests both MMIO-only and full telemetry modes.
"""

import sys
import time
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from statistics import mean, stdev

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from telemetry_benchmark_utils import (
    BenchmarkResult,
    DeviceManager,
    TelemetryManager,
    run_with_clean_state,
    safe_run_with_auto_reset,
    mann_whitney_u_test,
    cohens_d,
    calculate_impact_percentage,
    save_results_json,
)

# Import ttnn
import ttnn
import torch

# Test configuration
TENSOR_SIZE = (1, 1, 8192, 8192)  # 128MB
DEFAULT_DURATION_SEC = 300  # 5 minutes
DEFAULT_ITERATIONS = 1000
POLLING_FREQUENCY = "100ms"


@dataclass
class SustainedTestConfig:
    """Configuration for sustained workload test."""

    operation: str
    num_devices: int
    telemetry_mode: str  # "none", "mmio_only", "full"
    polling_interval: Optional[str]
    duration_sec: Optional[int]
    max_iterations: Optional[int]

    def __str__(self):
        mode_str = self.telemetry_mode if self.telemetry_mode != "none" else "baseline"
        duration_str = f"{self.duration_sec}s" if self.duration_sec else f"{self.max_iterations}iter"
        return f"{self.operation}_{self.num_devices}dev_{mode_str}_{duration_str}"


@dataclass
class SustainedTestResult:
    """Result from sustained workload test."""

    config: Dict[str, Any]
    samples: List[float]
    timestamps: List[float]  # Relative timestamps for each sample
    duration_sec: float
    n_iterations: int
    early_mean: float  # Mean of first 50 samples
    late_mean: float  # Mean of last 50 samples
    drift_percent: float  # Performance drift
    errors: List[str]
    metadata: Dict[str, Any]


def create_mesh_device(num_devices: int) -> ttnn.MeshDevice:
    """Create a mesh device with specified number of devices."""
    print(f"Creating mesh device with {num_devices} devices...")

    if num_devices == 4:
        mesh_shape = (2, 2)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    elif num_devices == 2:
        mesh_shape = (1, 2)
    else:
        raise ValueError(f"Unsupported device count: {num_devices}")

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))

    return mesh_device


def run_allgather_once(mesh_device: ttnn.MeshDevice, shape: Tuple[int, ...]) -> float:
    """Run a single AllGather operation and measure latency."""
    # Create input tensor on each device
    input_tensor = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Measure AllGather
    start = time.perf_counter()
    output_tensor = ttnn.all_gather(input_tensor, dim=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)
    end = time.perf_counter()

    # Cleanup
    del output_tensor
    del input_tensor

    return end - start


def run_sustained_test(config: SustainedTestConfig) -> SustainedTestResult:
    """
    Run sustained workload test.

    Args:
        config: Test configuration

    Returns:
        SustainedTestResult with timing data and drift analysis
    """
    print(f"\n{'='*80}")
    print(f"Running sustained test: {config}")
    print(f"{'='*80}")

    errors = []
    telemetry_proc = None
    samples = []
    timestamps = []

    try:
        # Start telemetry if needed
        if config.telemetry_mode != "none":
            mmio_only = config.telemetry_mode == "mmio_only"
            try:
                telemetry_proc = TelemetryManager.start_telemetry_validated(
                    config.polling_interval, mmio_only=mmio_only, stabilization_sec=5.0
                )
                print(f"Telemetry started in {config.telemetry_mode} mode")
            except Exception as e:
                error_msg = f"Failed to start telemetry: {e}"
                print(f"ERROR: {error_msg}")
                errors.append(error_msg)
                return SustainedTestResult(
                    config=asdict(config),
                    samples=[],
                    timestamps=[],
                    duration_sec=0,
                    n_iterations=0,
                    early_mean=0,
                    late_mean=0,
                    drift_percent=0,
                    errors=errors,
                    metadata={},
                )

        # Create mesh device
        mesh_device = create_mesh_device(config.num_devices)

        # Run sustained workload
        print(f"Starting sustained workload...")
        if config.duration_sec:
            print(f"  Duration: {config.duration_sec}s")
        if config.max_iterations:
            print(f"  Max iterations: {config.max_iterations}")

        start_time = time.perf_counter()
        iteration = 0

        while True:
            # Check stopping conditions
            elapsed = time.perf_counter() - start_time

            if config.duration_sec and elapsed >= config.duration_sec:
                print(f"  Reached duration limit: {elapsed:.1f}s")
                break

            if config.max_iterations and iteration >= config.max_iterations:
                print(f"  Reached iteration limit: {iteration}")
                break

            # Run operation
            try:
                latency = run_allgather_once(mesh_device, TENSOR_SIZE)
                samples.append(latency)
                timestamps.append(elapsed)
                iteration += 1

                # Progress updates
                if iteration % 100 == 0:
                    current_mean = mean(samples[-100:]) if len(samples) >= 100 else mean(samples)
                    print(f"  Iteration {iteration}: mean={current_mean*1000:.2f}ms, elapsed={elapsed:.1f}s")

            except Exception as e:
                error_msg = f"Iteration {iteration} failed: {e}"
                print(f"WARNING: {error_msg}")
                errors.append(error_msg)
                # Continue on individual failures
                continue

        # Close mesh device
        ttnn.close_mesh_device(mesh_device)

        total_duration = time.perf_counter() - start_time
        print(f"Sustained test completed: {iteration} iterations in {total_duration:.1f}s")

        # Analyze drift
        if len(samples) >= 100:
            early_samples = samples[:50]
            late_samples = samples[-50:]

            early_mean = mean(early_samples)
            late_mean = mean(late_samples)

            if early_mean > 0:
                drift_percent = 100 * (late_mean - early_mean) / early_mean
            else:
                drift_percent = 0.0

            print(f"  Early mean: {early_mean*1000:.2f}ms")
            print(f"  Late mean: {late_mean*1000:.2f}ms")
            print(f"  Drift: {drift_percent:+.2f}%")

            if abs(drift_percent) >= 5.0:
                print(f"  WARNING: Significant drift detected!")

        else:
            early_mean = mean(samples) if samples else 0
            late_mean = early_mean
            drift_percent = 0.0
            print(f"  Insufficient samples for drift analysis")

        # Create result
        result = SustainedTestResult(
            config=asdict(config),
            samples=samples,
            timestamps=timestamps,
            duration_sec=total_duration,
            n_iterations=iteration,
            early_mean=early_mean,
            late_mean=late_mean,
            drift_percent=drift_percent,
            errors=errors,
            metadata={"tensor_shape": TENSOR_SIZE, "total_samples": len(samples), "failed_iterations": len(errors)},
        )

        return result

    except Exception as e:
        error_msg = f"Sustained test failed with exception: {e}"
        print(f"ERROR: {error_msg}")
        errors.append(error_msg)

        return SustainedTestResult(
            config=asdict(config),
            samples=samples,
            timestamps=timestamps,
            duration_sec=0,
            n_iterations=len(samples),
            early_mean=0,
            late_mean=0,
            drift_percent=0,
            errors=errors,
            metadata={},
        )

    finally:
        # Stop telemetry
        if telemetry_proc is not None:
            TelemetryManager.stop_telemetry(telemetry_proc)


def analyze_sustained_results(results: List[SustainedTestResult]) -> Dict[str, Any]:
    """
    Analyze sustained workload test results.

    Args:
        results: List of sustained test results

    Returns:
        Analysis dictionary
    """
    print("\n" + "=" * 80)
    print("SUSTAINED WORKLOAD ANALYSIS")
    print("=" * 80)

    analysis = {"summary": {}, "drift_analysis": [], "stability_analysis": []}

    # Find baseline
    baseline = None
    for result in results:
        if result.config["telemetry_mode"] == "none":
            baseline = result
            break

    if not baseline or not baseline.samples:
        print("ERROR: No valid baseline found")
        return analysis

    print(f"\nBaseline ({baseline.n_iterations} iterations, {baseline.duration_sec:.1f}s):")
    print(f"  Early mean: {baseline.early_mean*1000:.2f}ms")
    print(f"  Late mean: {baseline.late_mean*1000:.2f}ms")
    print(f"  Drift: {baseline.drift_percent:+.2f}%")

    # Analyze each telemetry mode
    for result in results:
        if result.config["telemetry_mode"] == "none":
            continue

        if not result.samples:
            print(f"\n{result.config['telemetry_mode']} mode: FAILED")
            print(f"  Errors: {result.errors}")
            continue

        print(
            f"\n{result.config['telemetry_mode']} mode ({result.n_iterations} iterations, {result.duration_sec:.1f}s):"
        )
        print(f"  Early mean: {result.early_mean*1000:.2f}ms")
        print(f"  Late mean: {result.late_mean*1000:.2f}ms")
        print(f"  Drift: {result.drift_percent:+.2f}%")

        # Compare early and late periods to baseline
        early_impact = calculate_impact_percentage(baseline.early_mean, result.early_mean)
        late_impact = calculate_impact_percentage(baseline.late_mean, result.late_mean)
        drift_difference = result.drift_percent - baseline.drift_percent

        print(f"  Early impact vs baseline: {early_impact:+.2f}%")
        print(f"  Late impact vs baseline: {late_impact:+.2f}%")
        print(f"  Additional drift vs baseline: {drift_difference:+.2f}%")

        drift_analysis = {
            "telemetry_mode": result.config["telemetry_mode"],
            "n_iterations": result.n_iterations,
            "duration_sec": result.duration_sec,
            "drift_percent": result.drift_percent,
            "baseline_drift_percent": baseline.drift_percent,
            "drift_difference": drift_difference,
            "early_impact": early_impact,
            "late_impact": late_impact,
            "errors": result.errors,
        }
        analysis["drift_analysis"].append(drift_analysis)

        # Stability analysis: coefficient of variation over time
        if len(result.samples) >= 100:
            # Split into 10 windows
            window_size = len(result.samples) // 10
            window_cvs = []

            for i in range(10):
                window_start = i * window_size
                window_end = window_start + window_size
                window_samples = result.samples[window_start:window_end]

                if len(window_samples) > 1:
                    window_mean = mean(window_samples)
                    window_std = stdev(window_samples)
                    window_cv = window_std / window_mean if window_mean > 0 else 0
                    window_cvs.append(window_cv)

            avg_cv = mean(window_cvs) if window_cvs else 0
            print(f"  Average window CV: {avg_cv*100:.2f}%")

            stability = {
                "telemetry_mode": result.config["telemetry_mode"],
                "window_cvs": [float(cv) for cv in window_cvs],
                "average_cv": float(avg_cv),
            }
            analysis["stability_analysis"].append(stability)

    # Summary
    analysis["summary"] = {
        "baseline_drift": baseline.drift_percent,
        "max_additional_drift": max([d["drift_difference"] for d in analysis["drift_analysis"]], default=0),
        "drift_threshold_exceeded": any(abs(d["drift_difference"]) >= 5.0 for d in analysis["drift_analysis"]),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Baseline drift: {analysis['summary']['baseline_drift']:+.2f}%")
    print(f"  Max additional drift: {analysis['summary']['max_additional_drift']:+.2f}%")
    if analysis["summary"]["drift_threshold_exceeded"]:
        print(f"  WARNING: Drift threshold (5%) exceeded!")
    else:
        print(f"  All drift values within acceptable range (<5%)")
    print("=" * 80)

    return analysis


def main(duration_sec: int = DEFAULT_DURATION_SEC, max_iterations: int = None):
    """
    Run sustained workload tests.

    Args:
        duration_sec: Duration in seconds (if not using iterations)
        max_iterations: Maximum iterations (if not using duration)
    """
    print("=" * 80)
    print("SUSTAINED WORKLOAD TEST")
    print("=" * 80)

    if max_iterations:
        print(f"\nTest Configuration: {max_iterations} iterations")
    else:
        print(f"\nTest Configuration: {duration_sec}s duration")

    print(f"  Operation: AllGather")
    print(f"  Devices: 8")
    print(f"  Tensor size: {TENSOR_SIZE}")
    print(f"  Polling frequency: {POLLING_FREQUENCY}")

    # Test configurations
    configs = [
        SustainedTestConfig(
            operation="AllGather",
            num_devices=8,
            telemetry_mode="none",
            polling_interval=None,
            duration_sec=duration_sec if not max_iterations else None,
            max_iterations=max_iterations,
        ),
        SustainedTestConfig(
            operation="AllGather",
            num_devices=8,
            telemetry_mode="mmio_only",
            polling_interval=POLLING_FREQUENCY,
            duration_sec=duration_sec if not max_iterations else None,
            max_iterations=max_iterations,
        ),
        SustainedTestConfig(
            operation="AllGather",
            num_devices=8,
            telemetry_mode="full",
            polling_interval=POLLING_FREQUENCY,
            duration_sec=duration_sec if not max_iterations else None,
            max_iterations=max_iterations,
        ),
    ]

    all_results = []

    # Run all tests with clean state and auto-reset
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {config}")

        def test_wrapper():
            return run_sustained_test(config)

        result = safe_run_with_auto_reset(run_with_clean_state, test_wrapper, cooldown_sec=30.0, max_retries=2)

        all_results.append(result)

        # Save intermediate results
        save_results_json([asdict(r) for r in all_results], "/tmp/sustained_workload_results_partial.json")

    # Analyze results
    analysis = analyze_sustained_results(all_results)

    # Save final results
    final_output = {
        "test_config": {
            "duration_sec": duration_sec if not max_iterations else None,
            "max_iterations": max_iterations,
            "tensor_size": TENSOR_SIZE,
            "polling_frequency": POLLING_FREQUENCY,
        },
        "results": [asdict(r) for r in all_results],
        "analysis": analysis,
    }

    save_results_json(final_output, "/tmp/sustained_workload_results.json")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print(f"Results saved to: /tmp/sustained_workload_results.json")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    duration_sec = None
    max_iterations = None

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith("s"):
            duration_sec = int(arg[:-1])
        else:
            max_iterations = int(arg)
    else:
        duration_sec = DEFAULT_DURATION_SEC

    sys.exit(main(duration_sec=duration_sec, max_iterations=max_iterations))
