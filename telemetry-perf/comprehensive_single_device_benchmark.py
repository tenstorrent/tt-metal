#!/usr/bin/env python3
"""
Comprehensive Single-Device Telemetry Performance Benchmark

Tests telemetry impact on single-device workloads across:
  - 5 tensor sizes (1024² to 17408², reaching ~5% DRAM)
  - 5 operations (compute-bound + memory-bound)
  - 2 memory configs (L1, DRAM)
  - 12 polling frequencies (60s to 100us)

Includes all methodology improvements:
  - Device state reset between tests
  - Validated telemetry startup
  - Adaptive warmup
  - Interleaved baseline measurements
  - Multiple comparison correction
  - Outlier detection and normality checks
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
    interleaved_baseline_measurement,
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

# Full configuration (Phase 2)
SINGLE_DEVICE_SHAPES_FULL = [
    (1, 1, 1024, 1024),  # 2MB (0.02% DRAM) - tiny
    (1, 1, 4096, 4096),  # 32MB (0.26% DRAM) - small
    (1, 1, 8192, 8192),  # 128MB (1.04% DRAM) - medium
    (1, 1, 12288, 12288),  # 288MB (2.34% DRAM) - large
    (1, 1, 17408, 17408),  # 578MB (4.7% DRAM) - very large
]

# Reduced configuration (Phase 1)
SINGLE_DEVICE_SHAPES_REDUCED = [
    (1, 1, 1024, 1024),  # small
    (1, 1, 8192, 8192),  # medium
    (1, 1, 17408, 17408),  # very large
]

# Full frequency range - removed 60s and 10s as too slow to be relevant
POLLING_FREQUENCIES_FULL = ["5s", "1s", "500ms", "100ms", "50ms", "10ms", "5ms", "1ms", "500us", "100us"]

# Reduced frequency range - focus on critical frequencies
POLLING_FREQUENCIES_REDUCED = ["5s", "1s", "100ms", "10ms", "1ms", "100us"]

# Memory configurations - handle different ttnn versions
try:
    # Try the standard attribute names first
    MEMORY_CONFIGS = {
        "DRAM": ttnn.DRAM_MEMORY_CONFIG,
        "L1": ttnn.L1_MEMORY_CONFIG,
    }
except AttributeError:
    # Fallback: try to create memory configs manually
    try:
        # Some versions might use functions or different names
        MEMORY_CONFIGS = {
            "DRAM": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "L1": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        }
    except:
        # Last resort: use None and let ttnn use defaults
        print("Warning: Could not determine memory config format. Using defaults.")
        MEMORY_CONFIGS = {
            "DRAM": None,
            "L1": None,
        }

# Test parameters
N_SAMPLES = 100
WARMUP_ITERS = 20


@dataclass
class SingleDeviceTestConfig:
    """Configuration for a single-device test."""

    operation: str
    tensor_shape: Tuple[int, ...]
    memory_config: str  # "DRAM" or "L1"
    telemetry_enabled: bool
    polling_interval: Optional[str]

    def __str__(self):
        shape_str = f"{self.tensor_shape[2]}x{self.tensor_shape[3]}"
        telemetry_str = f"telemetry_{self.polling_interval}" if self.telemetry_enabled else "baseline"
        return f"{self.operation}_{shape_str}_{self.memory_config}_{telemetry_str}"


class SingleDeviceOperations:
    """Single-device operations for benchmarking."""

    def __init__(self, device: ttnn.Device):
        self.device = device

    def run_matmul(self, shape: Tuple[int, ...], memory_config: ttnn.MemoryConfig) -> float:
        """Run matrix multiplication (compute-bound)."""
        # Create two input tensors
        a = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        b = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        # Measure matmul
        start = time.perf_counter()
        c = ttnn.matmul(a, b, memory_config=memory_config)
        ttnn.synchronize_device(self.device)
        end = time.perf_counter()

        # Cleanup
        del c, b, a

        return end - start

    def run_add(self, shape: Tuple[int, ...], memory_config: ttnn.MemoryConfig) -> float:
        """Run element-wise addition (memory-bound)."""
        # Create two input tensors
        a = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        b = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        # Measure add
        start = time.perf_counter()
        c = ttnn.add(a, b, memory_config=memory_config)
        ttnn.synchronize_device(self.device)
        end = time.perf_counter()

        # Cleanup
        del c, b, a

        return end - start

    def run_concat(self, shape: Tuple[int, ...], memory_config: ttnn.MemoryConfig) -> float:
        """Run concatenation (pure memory copy, memory-bound)."""
        # Create two input tensors
        a = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        b = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        # Measure concat
        start = time.perf_counter()
        c = ttnn.concat([a, b], dim=0, memory_config=memory_config)
        ttnn.synchronize_device(self.device)
        end = time.perf_counter()

        # Cleanup
        del c, b, a

        return end - start

    def run_to_memory_config(self, shape: Tuple[int, ...], memory_config: ttnn.MemoryConfig) -> float:
        """Run explicit L1↔DRAM transfer (memory-bound)."""
        # Create tensor in opposite memory location
        if memory_config == ttnn.DRAM_MEMORY_CONFIG:
            source_config = ttnn.L1_MEMORY_CONFIG
            dest_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            source_config = ttnn.DRAM_MEMORY_CONFIG
            dest_config = ttnn.L1_MEMORY_CONFIG

        a = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=source_config,
        )

        # Measure transfer
        start = time.perf_counter()
        b = ttnn.to_memory_config(a, memory_config=dest_config)
        ttnn.synchronize_device(self.device)
        end = time.perf_counter()

        # Cleanup
        del b, a

        return end - start

    def run_reshape(self, shape: Tuple[int, ...], memory_config: ttnn.MemoryConfig) -> float:
        """Run reshape (layout transformation, memory-bound)."""
        # Create input tensor
        a = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        )

        # Reshape to different shape (preserve total elements)
        new_shape = (shape[0], shape[1], shape[2] * 2, shape[3] // 2)

        # Measure reshape
        start = time.perf_counter()
        b = ttnn.reshape(a, new_shape)
        ttnn.synchronize_device(self.device)
        end = time.perf_counter()

        # Cleanup
        del b, a

        return end - start


def run_single_device_test(config: SingleDeviceTestConfig, phase: str = "full") -> BenchmarkResult:
    """
    Run a single-device test with specified configuration.

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
        if config.telemetry_enabled:
            try:
                telemetry_proc = TelemetryManager.start_telemetry_validated(
                    config.polling_interval,
                    mmio_only=False,  # Single-device doesn't need mmio-only
                    stabilization_sec=5.0,
                )
                print(f"Telemetry started with {config.polling_interval} polling")
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

        # Open device
        device = ttnn.open_device(device_id=0)
        ops = SingleDeviceOperations(device)

        # Get memory config object
        memory_config = MEMORY_CONFIGS[config.memory_config]

        # Define test function based on operation
        if config.operation == "matmul":

            def test_func():
                return ops.run_matmul(config.tensor_shape, memory_config)

        elif config.operation == "add":

            def test_func():
                return ops.run_add(config.tensor_shape, memory_config)

        elif config.operation == "concat":

            def test_func():
                return ops.run_concat(config.tensor_shape, memory_config)

        elif config.operation == "to_memory_config":

            def test_func():
                return ops.run_to_memory_config(config.tensor_shape, memory_config)

        elif config.operation == "reshape":

            def test_func():
                return ops.run_reshape(config.tensor_shape, memory_config)

        else:
            raise ValueError(f"Unknown operation: {config.operation}")

        # Run benchmark with warmup
        use_adaptive = phase == "full"
        samples = run_benchmark_with_warmup(
            test_func, n_samples=N_SAMPLES, warmup_iters=WARMUP_ITERS, adaptive=use_adaptive
        )

        # Close device
        ttnn.close_device(device)

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
                "tensor_shape": config.tensor_shape,
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


def generate_test_configs(phase: str = "full") -> List[SingleDeviceTestConfig]:
    """
    Generate test configurations.

    Args:
        phase: "full" or "reduced"

    Returns:
        List of test configurations
    """
    if phase == "reduced":
        shapes = SINGLE_DEVICE_SHAPES_REDUCED
        frequencies = POLLING_FREQUENCIES_REDUCED
        operations = ["matmul", "add", "to_memory_config"]  # Reduced ops
        memory_configs = ["DRAM"]  # Only DRAM for reduced
    else:
        shapes = SINGLE_DEVICE_SHAPES_FULL
        frequencies = POLLING_FREQUENCIES_FULL
        operations = ["matmul", "add", "concat", "to_memory_config", "reshape"]
        memory_configs = ["DRAM", "L1"]

    configs = []

    for operation in operations:
        for shape in shapes:
            for mem_config in memory_configs:
                # Optimization: only test matmul on DRAM (compute-bound, memory location matters less)
                if operation == "matmul" and mem_config == "L1":
                    continue

                # Baseline test
                configs.append(
                    SingleDeviceTestConfig(
                        operation=operation,
                        tensor_shape=shape,
                        memory_config=mem_config,
                        telemetry_enabled=False,
                        polling_interval=None,
                    )
                )

                # Telemetry tests at each frequency
                for freq in frequencies:
                    configs.append(
                        SingleDeviceTestConfig(
                            operation=operation,
                            tensor_shape=shape,
                            memory_config=mem_config,
                            telemetry_enabled=True,
                            polling_interval=freq,
                        )
                    )

    return configs


def analyze_single_device_results(results: List[BenchmarkResult], phase: str) -> Dict[str, Any]:
    """
    Analyze single-device benchmark results.

    Args:
        results: List of benchmark results
        phase: "full" or "reduced"

    Returns:
        Analysis dictionary
    """
    print("\n" + "=" * 80)
    print("SINGLE-DEVICE BENCHMARK ANALYSIS")
    print("=" * 80)

    analysis = {
        "phase": phase,
        "summary": {},
        "monotonicity_tests": [],
        "statistical_comparisons": [],
        "multiple_comparison_correction": {},
    }

    # Group results by operation, shape, and memory config
    grouped = {}
    for result in results:
        if result.errors or not result.samples:
            continue

        key = (result.config["operation"], tuple(result.config["tensor_shape"]), result.config["memory_config"])

        if key not in grouped:
            grouped[key] = {"baseline": None, "telemetry": []}

        if not result.config["telemetry_enabled"]:
            grouped[key]["baseline"] = result
        else:
            grouped[key]["telemetry"].append(result)

    # Analyze each group
    all_p_values = []
    significant_impacts = []

    for (operation, shape, mem_config), data in grouped.items():
        baseline = data["baseline"]
        telemetry_results = data["telemetry"]

        if not baseline or not telemetry_results:
            continue

        print(f"\n{operation} {shape[2]}x{shape[3]} {mem_config}:")
        print(f"  Baseline: {baseline.mean_time*1000:.2f}ms ± {baseline.std_time*1000:.2f}ms")

        # Analyze each telemetry frequency
        impacts = []
        frequencies_hz = []
        comparisons = []

        for tel_result in sorted(telemetry_results, key=lambda r: parse_frequency_to_hz(r.config["polling_interval"])):
            freq = tel_result.config["polling_interval"]
            freq_hz = parse_frequency_to_hz(freq)

            impact_pct = calculate_impact_percentage(baseline.mean_time, tel_result.mean_time)
            stat_test = mann_whitney_u_test(baseline.samples, tel_result.samples)
            effect = cohens_d(baseline.samples, tel_result.samples)

            impacts.append(impact_pct)
            frequencies_hz.append(freq_hz)
            all_p_values.append(stat_test["p_value"])

            comparison = {
                "operation": operation,
                "shape": shape,
                "memory_config": mem_config,
                "polling_interval": freq,
                "frequency_hz": freq_hz,
                "impact_percent": impact_pct,
                "p_value": stat_test["p_value"],
                "cohens_d": effect,
            }
            comparisons.append(comparison)

            if abs(impact_pct) >= 5.0 and stat_test["p_value"] < 0.05:
                significant_impacts.append(comparison)
                print(f"    {freq:>6s}: {impact_pct:+6.2f}% (p={stat_test['p_value']:.4f}) **SIGNIFICANT**")
            else:
                print(f"    {freq:>6s}: {impact_pct:+6.2f}% (p={stat_test['p_value']:.4f})")

        # Monotonicity test
        if len(frequencies_hz) >= 3:
            monotonicity = analyze_monotonicity(frequencies_hz, impacts)
            monotonicity["operation"] = operation
            monotonicity["shape"] = shape
            monotonicity["memory_config"] = mem_config
            analysis["monotonicity_tests"].append(monotonicity)

            if monotonicity.get("monotonic"):
                print(f"  Monotonic trend detected: tau={monotonicity['tau']:.3f}, p={monotonicity['p_value']:.4f}")

        analysis["statistical_comparisons"].extend(comparisons)

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
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Total tests: {analysis['summary']['total_tests']}")
    print(f"  Successful: {analysis['summary']['successful_tests']}")
    print(f"  Failed: {analysis['summary']['failed_tests']}")
    print(f"  Significant impacts (uncorrected): {analysis['summary']['significant_impacts_uncorrected']}")
    print(f"  Monotonic relationships: {analysis['summary']['monotonic_relationships']}")
    print("=" * 80)

    return analysis


def main(phase: str = "full"):
    """
    Run comprehensive single-device benchmark.

    Args:
        phase: "full" or "reduced"
    """
    print("=" * 80)
    print(f"COMPREHENSIVE SINGLE-DEVICE TELEMETRY BENCHMARK - {phase.upper()} PHASE")
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
            return run_single_device_test(config, phase)

        result = safe_run_with_auto_reset(run_with_clean_state, test_wrapper, cooldown_sec=30.0, max_retries=2)

        all_results.append(result)

        # Save intermediate results every 10 tests
        if (i + 1) % 10 == 0:
            save_results_json([asdict(r) for r in all_results], f"/tmp/single_device_results_{phase}_partial.json")

    # Analyze results
    analysis = analyze_single_device_results(all_results, phase)

    # Save final results
    final_output = {
        "phase": phase,
        "test_config": {"n_samples": N_SAMPLES, "warmup_iters": WARMUP_ITERS, "n_configs": len(configs)},
        "results": [asdict(r) for r in all_results],
        "analysis": analysis,
    }

    save_results_json(final_output, f"/tmp/single_device_results_{phase}.json")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print(f"Results saved to: /tmp/single_device_results_{phase}.json")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    phase = sys.argv[1] if len(sys.argv) > 1 else "reduced"
    if phase not in ["full", "reduced"]:
        print(f"Error: Invalid phase '{phase}'. Must be 'full' or 'reduced'")
        sys.exit(1)

    sys.exit(main(phase))
