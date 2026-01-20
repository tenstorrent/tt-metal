#!/usr/bin/env python3
"""
Comprehensive Multi-Device Telemetry Performance Benchmark

Tests telemetry impact on multi-device (CCL) workloads across:
  - 3 CCL operations (AllGather, ReduceScatter, AllReduce)
  - 3 device counts (2, 4, 8)
  - 3 telemetry modes (none, mmio_only, full)
  - 12 polling frequencies (60s to 100us)

Validates:
  - MMIO-only prevents ERISC contention
  - Full mode may show impact or failures
  - Monotonic relationship between polling frequency and impact
"""

import sys
import time
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from telemetry_benchmark_utils import (
    BenchmarkResult,
    DeviceManager,
    TelemetryManager,
    run_with_clean_state,
    safe_run_with_auto_reset,
    run_benchmark_with_warmup,
    mann_whitney_u_test,
    cohens_d,
    calculate_impact_percentage,
    apply_multiple_comparison_correction,
    analyze_monotonicity,
    parse_frequency_to_hz,
    check_normality,
    detect_outliers_iqr,
    save_results_json,
)

# Import ttnn
import ttnn
import torch

# Full frequency range
POLLING_FREQUENCIES_FULL = ["60s", "10s", "5s", "1s", "500ms", "100ms", "50ms", "10ms", "5ms", "1ms", "500us", "100us"]

# Reduced frequency range (Phase 1)
POLLING_FREQUENCIES_REDUCED = ["60s", "1s", "100ms", "10ms", "1ms", "100us"]

# Device counts to test
DEVICE_COUNTS_FULL = [2, 4, 8]
DEVICE_COUNTS_REDUCED = [2, 4, 8]

# CCL operations
CCL_OPERATIONS_FULL = ["AllGather", "ReduceScatter", "AllReduce"]
CCL_OPERATIONS_REDUCED = ["AllGather"]  # Only AllGather for reduced phase

# Tensor size
TENSOR_SIZE = (1, 1, 8192, 8192)  # 128MB

# Test parameters
N_SAMPLES = 100
WARMUP_ITERS = 20


@dataclass
class MultiDeviceTestConfig:
    """Configuration for a multi-device test."""

    operation: str
    num_devices: int
    telemetry_mode: str  # "none", "mmio_only", "full"
    polling_interval: Optional[str]

    def __str__(self):
        mode_str = self.telemetry_mode if self.telemetry_mode != "none" else "baseline"
        return f"{self.operation}_{self.num_devices}dev_{mode_str}_{self.polling_interval or 'none'}"


class MultiDeviceOperations:
    """Multi-device CCL operations for benchmarking."""

    def __init__(self, mesh_device: ttnn.MeshDevice):
        self.mesh_device = mesh_device

    def run_allgather(self, shape: Tuple[int, ...]) -> float:
        """Run AllGather operation."""
        # Create input tensor on each device
        input_tensor = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Measure AllGather
        start = time.perf_counter()
        output_tensor = ttnn.all_gather(input_tensor, dim=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(self.mesh_device)
        end = time.perf_counter()

        # Cleanup
        del output_tensor
        del input_tensor

        return end - start

    def run_reduce_scatter(self, shape: Tuple[int, ...]) -> float:
        """Run ReduceScatter operation."""
        # Create input tensor on each device
        input_tensor = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Measure ReduceScatter
        start = time.perf_counter()
        output_tensor = ttnn.reduce_scatter(
            input_tensor, scatter_dim=0, math_op=ttnn.ReduceType.Sum, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.synchronize_device(self.mesh_device)
        end = time.perf_counter()

        # Cleanup
        del output_tensor
        del input_tensor

        return end - start

    def run_all_reduce(self, shape: Tuple[int, ...]) -> float:
        """Run AllReduce operation (highest bandwidth CCL op)."""
        # Create input tensor on each device
        input_tensor = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Measure AllReduce
        start = time.perf_counter()
        output_tensor = ttnn.all_reduce(
            input_tensor, math_op=ttnn.ReduceType.Sum, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.synchronize_device(self.mesh_device)
        end = time.perf_counter()

        # Cleanup
        del output_tensor
        del input_tensor

        return end - start


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


def run_multi_device_test(config: MultiDeviceTestConfig, phase: str = "full") -> BenchmarkResult:
    """
    Run a multi-device test with specified configuration.

    Args:
        config: Test configuration
        phase: "full" or "reduced"

    Returns:
        BenchmarkResult with timing data
    """
    print(f"\n{'='*80}")
    print(f"Running test: {config}")
    print(f"{'='*80}")

    errors = []
    telemetry_proc = None

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
                return BenchmarkResult(
                    operation=str(config),
                    config=asdict(config),
                    samples=[],
                    mean_time=0,
                    std_time=0,
                    median_time=0,
                    min_time=0,
                    max_time=0,
                    cv=0,
                    errors=errors,
                    metadata={},
                )

        # Create mesh device
        mesh_device = create_mesh_device(config.num_devices)
        ops = MultiDeviceOperations(mesh_device)

        # Define test function based on operation
        if config.operation == "AllGather":

            def test_func():
                return ops.run_allgather(TENSOR_SIZE)

        elif config.operation == "ReduceScatter":

            def test_func():
                return ops.run_reduce_scatter(TENSOR_SIZE)

        elif config.operation == "AllReduce":

            def test_func():
                return ops.run_all_reduce(TENSOR_SIZE)

        else:
            raise ValueError(f"Unknown operation: {config.operation}")

        # Run benchmark with warmup
        use_adaptive = phase == "full"
        samples = run_benchmark_with_warmup(
            test_func, n_samples=N_SAMPLES, warmup_iters=WARMUP_ITERS, adaptive=use_adaptive
        )

        # Close mesh device
        ttnn.close_mesh_device(mesh_device)

        # Statistical analysis
        normality = check_normality(samples)
        outliers = detect_outliers_iqr(samples)

        # Create result
        result = BenchmarkResult(
            operation=str(config),
            config=asdict(config),
            samples=samples,
            mean_time=0,
            std_time=0,
            median_time=0,
            min_time=0,
            max_time=0,
            cv=0,
            errors=errors,
            metadata={
                "tensor_shape": TENSOR_SIZE,
                "n_samples": N_SAMPLES,
                "warmup_iters": WARMUP_ITERS,
                "normality": normality,
                "outliers": outliers,
            },
        )

        print(f"Test completed: mean={result.mean_time*1000:.2f}ms, cv={result.cv*100:.1f}%")

        return result

    except Exception as e:
        error_msg = f"Test failed with exception: {e}"
        print(f"ERROR: {error_msg}")
        errors.append(error_msg)

        return BenchmarkResult(
            operation=str(config),
            config=asdict(config),
            samples=[],
            mean_time=0,
            std_time=0,
            median_time=0,
            min_time=0,
            max_time=0,
            cv=0,
            errors=errors,
            metadata={},
        )

    finally:
        # Stop telemetry
        if telemetry_proc is not None:
            TelemetryManager.stop_telemetry(telemetry_proc)


def generate_test_configs(phase: str = "full") -> List[MultiDeviceTestConfig]:
    """
    Generate test configurations.

    Args:
        phase: "full" or "reduced"

    Returns:
        List of test configurations
    """
    if phase == "reduced":
        frequencies = POLLING_FREQUENCIES_REDUCED
        operations = CCL_OPERATIONS_REDUCED
        device_counts = DEVICE_COUNTS_REDUCED
    else:
        frequencies = POLLING_FREQUENCIES_FULL
        operations = CCL_OPERATIONS_FULL
        device_counts = DEVICE_COUNTS_FULL

    configs = []

    for operation in operations:
        for num_devices in device_counts:
            # Baseline test
            configs.append(
                MultiDeviceTestConfig(
                    operation=operation, num_devices=num_devices, telemetry_mode="none", polling_interval=None
                )
            )

            # MMIO-only tests at each frequency
            for freq in frequencies:
                configs.append(
                    MultiDeviceTestConfig(
                        operation=operation, num_devices=num_devices, telemetry_mode="mmio_only", polling_interval=freq
                    )
                )

            # Full mode tests - only at 100ms for reduced, all frequencies for full
            if phase == "reduced":
                test_freqs = ["100ms"]
            else:
                test_freqs = frequencies

            for freq in test_freqs:
                configs.append(
                    MultiDeviceTestConfig(
                        operation=operation, num_devices=num_devices, telemetry_mode="full", polling_interval=freq
                    )
                )

    return configs


def analyze_multi_device_results(results: List[BenchmarkResult], phase: str) -> Dict[str, Any]:
    """
    Analyze multi-device benchmark results.

    Args:
        results: List of benchmark results
        phase: "full" or "reduced"

    Returns:
        Analysis dictionary
    """
    print("\n" + "=" * 80)
    print("MULTI-DEVICE BENCHMARK ANALYSIS")
    print("=" * 80)

    analysis = {
        "phase": phase,
        "summary": {},
        "monotonicity_tests": [],
        "mmio_vs_full_comparisons": [],
        "statistical_comparisons": [],
        "multiple_comparison_correction": {},
    }

    # Group results by operation and device count
    grouped = {}
    for result in results:
        if result.errors or not result.samples:
            # Track errors separately
            if result.config["telemetry_mode"] == "full":
                print(
                    f"ERROR in {result.config['operation']} {result.config['num_devices']}dev full mode: {result.errors}"
                )
            continue

        key = (result.config["operation"], result.config["num_devices"])

        if key not in grouped:
            grouped[key] = {"baseline": None, "mmio_only": [], "full": []}

        mode = result.config["telemetry_mode"]
        if mode == "none":
            grouped[key]["baseline"] = result
        elif mode == "mmio_only":
            grouped[key]["mmio_only"].append(result)
        elif mode == "full":
            grouped[key]["full"].append(result)

    # Analyze each group
    all_p_values = []
    significant_impacts = []

    for (operation, num_devices), data in grouped.items():
        baseline = data["baseline"]
        mmio_results = data["mmio_only"]
        full_results = data["full"]

        if not baseline:
            continue

        print(f"\n{operation} ({num_devices} devices):")
        print(f"  Baseline: {baseline.mean_time*1000:.2f}ms ± {baseline.std_time*1000:.2f}ms")

        # Analyze MMIO-only mode
        if mmio_results:
            print("  MMIO-only mode:")
            mmio_impacts = []
            mmio_frequencies_hz = []

            for result in sorted(mmio_results, key=lambda r: parse_frequency_to_hz(r.config["polling_interval"])):
                freq = result.config["polling_interval"]
                freq_hz = parse_frequency_to_hz(freq)

                impact_pct = calculate_impact_percentage(baseline.mean_time, result.mean_time)
                stat_test = mann_whitney_u_test(baseline.samples, result.samples)
                effect = cohens_d(baseline.samples, result.samples)

                mmio_impacts.append(impact_pct)
                mmio_frequencies_hz.append(freq_hz)
                all_p_values.append(stat_test["p_value"])

                comparison = {
                    "operation": operation,
                    "num_devices": num_devices,
                    "telemetry_mode": "mmio_only",
                    "polling_interval": freq,
                    "frequency_hz": freq_hz,
                    "impact_percent": impact_pct,
                    "p_value": stat_test["p_value"],
                    "cohens_d": effect,
                }
                analysis["statistical_comparisons"].append(comparison)

                if abs(impact_pct) >= 5.0 and stat_test["p_value"] < 0.05:
                    significant_impacts.append(comparison)
                    print(f"    {freq:>6s}: {impact_pct:+6.2f}% (p={stat_test['p_value']:.4f}) **SIGNIFICANT**")
                else:
                    print(f"    {freq:>6s}: {impact_pct:+6.2f}% (p={stat_test['p_value']:.4f})")

            # Monotonicity test for MMIO-only
            if len(mmio_frequencies_hz) >= 3:
                monotonicity = analyze_monotonicity(mmio_frequencies_hz, mmio_impacts)
                monotonicity["operation"] = operation
                monotonicity["num_devices"] = num_devices
                monotonicity["telemetry_mode"] = "mmio_only"
                analysis["monotonicity_tests"].append(monotonicity)

                if monotonicity.get("monotonic"):
                    print(
                        f"  MMIO-only monotonic trend: tau={monotonicity['tau']:.3f}, p={monotonicity['p_value']:.4f}"
                    )

        # Analyze full mode
        if full_results:
            print("  Full mode:")
            full_impacts = []
            full_frequencies_hz = []

            for result in sorted(full_results, key=lambda r: parse_frequency_to_hz(r.config["polling_interval"])):
                freq = result.config["polling_interval"]
                freq_hz = parse_frequency_to_hz(freq)

                impact_pct = calculate_impact_percentage(baseline.mean_time, result.mean_time)
                stat_test = mann_whitney_u_test(baseline.samples, result.samples)
                effect = cohens_d(baseline.samples, result.samples)

                full_impacts.append(impact_pct)
                full_frequencies_hz.append(freq_hz)
                all_p_values.append(stat_test["p_value"])

                comparison = {
                    "operation": operation,
                    "num_devices": num_devices,
                    "telemetry_mode": "full",
                    "polling_interval": freq,
                    "frequency_hz": freq_hz,
                    "impact_percent": impact_pct,
                    "p_value": stat_test["p_value"],
                    "cohens_d": effect,
                }
                analysis["statistical_comparisons"].append(comparison)

                if abs(impact_pct) >= 10.0 and stat_test["p_value"] < 0.05:
                    significant_impacts.append(comparison)
                    print(f"    {freq:>6s}: {impact_pct:+6.2f}% (p={stat_test['p_value']:.4f}) **SIGNIFICANT**")
                else:
                    print(f"    {freq:>6s}: {impact_pct:+6.2f}% (p={stat_test['p_value']:.4f})")

            # Monotonicity test for full mode
            if len(full_frequencies_hz) >= 3:
                monotonicity = analyze_monotonicity(full_frequencies_hz, full_impacts)
                monotonicity["operation"] = operation
                monotonicity["num_devices"] = num_devices
                monotonicity["telemetry_mode"] = "full"
                analysis["monotonicity_tests"].append(monotonicity)

                if monotonicity.get("monotonic"):
                    print(
                        f"  Full mode monotonic trend: tau={monotonicity['tau']:.3f}, p={monotonicity['p_value']:.4f}"
                    )

        # Direct comparison: MMIO-only vs Full at same frequency
        if mmio_results and full_results:
            print("  MMIO-only vs Full comparison:")
            for mmio_result in mmio_results:
                freq = mmio_result.config["polling_interval"]
                # Find matching full mode result
                full_result = next((r for r in full_results if r.config["polling_interval"] == freq), None)

                if full_result:
                    mmio_impact = calculate_impact_percentage(baseline.mean_time, mmio_result.mean_time)
                    full_impact = calculate_impact_percentage(baseline.mean_time, full_result.mean_time)
                    difference = full_impact - mmio_impact

                    stat_test = mann_whitney_u_test(mmio_result.samples, full_result.samples)

                    comparison = {
                        "operation": operation,
                        "num_devices": num_devices,
                        "polling_interval": freq,
                        "mmio_impact_percent": mmio_impact,
                        "full_impact_percent": full_impact,
                        "difference_percent": difference,
                        "p_value": stat_test["p_value"],
                    }
                    analysis["mmio_vs_full_comparisons"].append(comparison)

                    if abs(difference) >= 5.0:
                        print(
                            f"    {freq:>6s}: MMIO {mmio_impact:+.2f}%, Full {full_impact:+.2f}%, Diff {difference:+.2f}% **NOTABLE**"
                        )
                    else:
                        print(
                            f"    {freq:>6s}: MMIO {mmio_impact:+.2f}%, Full {full_impact:+.2f}%, Diff {difference:+.2f}%"
                        )

    # Multiple comparison correction
    if all_p_values:
        correction = apply_multiple_comparison_correction(all_p_values, method="holm")
        analysis["multiple_comparison_correction"] = correction

        print(f"\nMultiple Comparison Correction (Holm method):")
        print(f"  Total tests: {correction['n_tests']}")
        print(f"  Significant (uncorrected α=0.05): {correction['n_significant_uncorrected']}")
        print(f"  Significant (corrected α=0.05): {correction['n_significant_corrected']}")

    # Summary
    analysis["summary"] = {
        "total_tests": len(results),
        "successful_tests": len([r for r in results if not r.errors and r.samples]),
        "failed_tests": len([r for r in results if r.errors or not r.samples]),
        "significant_impacts_uncorrected": len(significant_impacts),
        "monotonic_relationships": len([m for m in analysis["monotonicity_tests"] if m.get("monotonic")]),
        "mmio_vs_full_differences": len(
            [c for c in analysis["mmio_vs_full_comparisons"] if abs(c["difference_percent"]) >= 5.0]
        ),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Total tests: {analysis['summary']['total_tests']}")
    print(f"  Successful: {analysis['summary']['successful_tests']}")
    print(f"  Failed: {analysis['summary']['failed_tests']}")
    print(f"  Significant impacts (uncorrected): {analysis['summary']['significant_impacts_uncorrected']}")
    print(f"  Monotonic relationships: {analysis['summary']['monotonic_relationships']}")
    print(f"  Notable MMIO vs Full differences: {analysis['summary']['mmio_vs_full_differences']}")
    print("=" * 80)

    return analysis


def main(phase: str = "full"):
    """
    Run comprehensive multi-device benchmark.

    Args:
        phase: "full" or "reduced"
    """
    print("=" * 80)
    print(f"COMPREHENSIVE MULTI-DEVICE TELEMETRY BENCHMARK - {phase.upper()} PHASE")
    print("=" * 80)

    # Generate test configurations
    configs = generate_test_configs(phase)

    print(f"\nGenerated {len(configs)} test configurations")
    print(f"  Samples per test: {N_SAMPLES}")
    print(f"  Warmup iterations: {WARMUP_ITERS}")
    if phase == "full":
        print(f"  Adaptive warmup: enabled")
    else:
        print(f"  Adaptive warmup: disabled (reduced phase)")

    all_results = []

    # Run all tests with clean state and auto-reset
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {config}")

        def test_wrapper():
            return run_multi_device_test(config, phase)

        result = safe_run_with_auto_reset(run_with_clean_state, test_wrapper, cooldown_sec=30.0, max_retries=2)

        all_results.append(result)

        # Save intermediate results every 10 tests
        if (i + 1) % 10 == 0:
            save_results_json([asdict(r) for r in all_results], f"/tmp/multi_device_results_{phase}_partial.json")

    # Analyze results
    analysis = analyze_multi_device_results(all_results, phase)

    # Save final results
    final_output = {
        "phase": phase,
        "test_config": {"n_samples": N_SAMPLES, "warmup_iters": WARMUP_ITERS, "n_configs": len(configs)},
        "results": [asdict(r) for r in all_results],
        "analysis": analysis,
    }

    save_results_json(final_output, f"/tmp/multi_device_results_{phase}.json")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print(f"Results saved to: /tmp/multi_device_results_{phase}.json")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    phase = sys.argv[1] if len(sys.argv) > 1 else "reduced"
    if phase not in ["full", "reduced"]:
        print(f"Error: Invalid phase '{phase}'. Must be 'full' or 'reduced'")
        sys.exit(1)

    sys.exit(main(phase))
