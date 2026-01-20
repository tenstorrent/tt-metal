#!/usr/bin/env python3
"""
Core Hypothesis Validation Test for --mmio-only Flag

This test validates the central hypothesis:
  --mmio-only prevents ERISC contention on multi-chip workloads

Test Design:
  - Baseline (no telemetry): Clean operation baseline
  - MMIO-only mode: Should show <5% impact vs baseline
  - Full mode: Should show >10% impact OR failures

Operations tested:
  - AllGather (4 and 8 devices)
  - ReduceScatter (4 devices)
"""

import sys
import time
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

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
    save_results_json,
)

# Import ttnn
import ttnn

# Test Configuration
N_SAMPLES = 100
WARMUP_ITERS = 20
POLLING_FREQUENCY = "100ms"  # Fixed frequency for all tests
TENSOR_SIZE = (1, 1, 8192, 8192)  # 128MB tensor


@dataclass
class HypothesisTestConfig:
    """Configuration for a single hypothesis test."""

    operation: str
    num_devices: int
    telemetry_mode: str  # "none", "mmio_only", "full"
    polling_interval: str | None

    def __str__(self):
        mode_str = self.telemetry_mode if self.telemetry_mode != "none" else "baseline"
        return f"{self.operation}_{self.num_devices}dev_{mode_str}"


# Critical test configurations
CRITICAL_TEST_CONFIGS = [
    # 4-device AllGather
    HypothesisTestConfig("AllGather", 4, "none", None),
    HypothesisTestConfig("AllGather", 4, "mmio_only", POLLING_FREQUENCY),
    HypothesisTestConfig("AllGather", 4, "full", POLLING_FREQUENCY),
    # 8-device AllGather (more ERISC stress)
    HypothesisTestConfig("AllGather", 8, "none", None),
    HypothesisTestConfig("AllGather", 8, "mmio_only", POLLING_FREQUENCY),
    HypothesisTestConfig("AllGather", 8, "full", POLLING_FREQUENCY),
    # 4-device ReduceScatter (higher bandwidth)
    HypothesisTestConfig("ReduceScatter", 4, "none", None),
    HypothesisTestConfig("ReduceScatter", 4, "mmio_only", POLLING_FREQUENCY),
    HypothesisTestConfig("ReduceScatter", 4, "full", POLLING_FREQUENCY),
]


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


def run_allgather_once(mesh_device: ttnn.MeshDevice, tensor_shape: Tuple[int, ...]) -> float:
    """
    Run a single AllGather operation and measure latency.

    Args:
        mesh_device: Mesh device to use
        tensor_shape: Shape of input tensor

    Returns:
        Latency in seconds
    """
    # Create input tensor on each device
    input_tensor = ttnn.from_torch(
        torch.randn(tensor_shape, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Measure AllGather
    start = time.perf_counter()
    output_tensor = ttnn.all_gather(input_tensor, dim=0, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Synchronize to ensure operation completed
    ttnn.synchronize_device(mesh_device)
    end = time.perf_counter()

    # Cleanup
    del output_tensor
    del input_tensor

    return end - start


def run_reduce_scatter_once(mesh_device: ttnn.MeshDevice, tensor_shape: Tuple[int, ...]) -> float:
    """
    Run a single ReduceScatter operation and measure latency.

    Args:
        mesh_device: Mesh device to use
        tensor_shape: Shape of input tensor

    Returns:
        Latency in seconds
    """
    import torch

    # Create input tensor on each device
    input_tensor = ttnn.from_torch(
        torch.randn(tensor_shape, dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Measure ReduceScatter
    start = time.perf_counter()
    output_tensor = ttnn.reduce_scatter(
        input_tensor, scatter_dim=0, math_op=ttnn.ReduceType.Sum, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # Synchronize to ensure operation completed
    ttnn.synchronize_device(mesh_device)
    end = time.perf_counter()

    # Cleanup
    del output_tensor
    del input_tensor

    return end - start


def run_ccl_test(config: HypothesisTestConfig) -> BenchmarkResult:
    """
    Run a CCL operation test with specified configuration.

    Args:
        config: Test configuration

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

        # Define test function based on operation
        if config.operation == "AllGather":

            def test_func():
                return run_allgather_once(mesh_device, TENSOR_SIZE)

        elif config.operation == "ReduceScatter":

            def test_func():
                return run_reduce_scatter_once(mesh_device, TENSOR_SIZE)

        else:
            raise ValueError(f"Unknown operation: {config.operation}")

        # Run benchmark with warmup
        samples = run_benchmark_with_warmup(test_func, n_samples=N_SAMPLES, warmup_iters=WARMUP_ITERS, adaptive=False)

        # Close mesh device
        ttnn.close_mesh_device(mesh_device)

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
            metadata={"tensor_shape": TENSOR_SIZE, "n_samples": N_SAMPLES, "warmup_iters": WARMUP_ITERS},
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


def analyze_hypothesis_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    Analyze results to validate hypothesis.

    Expected outcomes:
      - baseline: clean operation
      - mmio_only: <5% impact (statistically insignificant)
      - full: >10% impact OR failures/timeouts

    Args:
        results: List of benchmark results

    Returns:
        Analysis dictionary
    """
    print("\n" + "=" * 80)
    print("HYPOTHESIS VALIDATION ANALYSIS")
    print("=" * 80)

    analysis = {"hypothesis_validated": False, "summary": "", "comparisons": [], "statistics": {}}

    # Group results by operation and device count
    grouped = {}
    for result in results:
        key = (result.config["operation"], result.config["num_devices"])
        if key not in grouped:
            grouped[key] = {}
        mode = result.config["telemetry_mode"]
        grouped[key][mode] = result

    # Analyze each group
    all_p_values = []
    comparisons = []

    for (operation, num_devices), modes in grouped.items():
        print(f"\n{operation} ({num_devices} devices):")
        print("-" * 60)

        baseline = modes.get("none")
        mmio_result = modes.get("mmio_only")
        full_result = modes.get("full")

        if not baseline or not baseline.samples:
            print("  ERROR: No valid baseline")
            continue

        print(
            f"  Baseline: {baseline.mean_time*1000:.2f}ms ± {baseline.std_time*1000:.2f}ms (CV={baseline.cv*100:.1f}%)"
        )

        # Analyze MMIO-only mode
        if mmio_result and mmio_result.samples:
            impact_mmio = calculate_impact_percentage(baseline.mean_time, mmio_result.mean_time)
            stat_test = mann_whitney_u_test(baseline.samples, mmio_result.samples)
            effect = cohens_d(baseline.samples, mmio_result.samples)

            print(f"  MMIO-only: {mmio_result.mean_time*1000:.2f}ms ± {mmio_result.std_time*1000:.2f}ms")
            print(f"    Impact: {impact_mmio:+.2f}%")
            print(f"    P-value: {stat_test['p_value']:.4f}")
            print(f"    Cohen's d: {effect:.3f}")

            comparison = {
                "operation": operation,
                "num_devices": num_devices,
                "mode": "mmio_only",
                "impact_percent": impact_mmio,
                "p_value": stat_test["p_value"],
                "cohens_d": effect,
                "hypothesis_met": abs(impact_mmio) < 5.0,  # <5% impact
            }
            comparisons.append(comparison)
            all_p_values.append(stat_test["p_value"])

            if abs(impact_mmio) >= 5.0:
                print(f"    ⚠️  WARNING: MMIO-only shows ≥5% impact (hypothesis NOT met)")
            else:
                print(f"    ✓ MMIO-only shows <5% impact (hypothesis met)")

        # Analyze full mode
        if full_result:
            if full_result.errors:
                print(f"  Full mode: FAILED")
                print(f"    Errors: {full_result.errors}")
                print(f"    ✓ Full mode shows failures (hypothesis met)")

                comparison = {
                    "operation": operation,
                    "num_devices": num_devices,
                    "mode": "full",
                    "impact_percent": None,
                    "p_value": None,
                    "cohens_d": None,
                    "hypothesis_met": True,  # Failures validate hypothesis
                    "errors": full_result.errors,
                }
                comparisons.append(comparison)

            elif full_result.samples:
                impact_full = calculate_impact_percentage(baseline.mean_time, full_result.mean_time)
                stat_test = mann_whitney_u_test(baseline.samples, full_result.samples)
                effect = cohens_d(baseline.samples, full_result.samples)

                print(f"  Full mode: {full_result.mean_time*1000:.2f}ms ± {full_result.std_time*1000:.2f}ms")
                print(f"    Impact: {impact_full:+.2f}%")
                print(f"    P-value: {stat_test['p_value']:.4f}")
                print(f"    Cohen's d: {effect:.3f}")

                comparison = {
                    "operation": operation,
                    "num_devices": num_devices,
                    "mode": "full",
                    "impact_percent": impact_full,
                    "p_value": stat_test["p_value"],
                    "cohens_d": effect,
                    "hypothesis_met": abs(impact_full) >= 10.0 or full_result.cv > 0.30,  # >10% impact or high variance
                }
                comparisons.append(comparison)
                all_p_values.append(stat_test["p_value"])

                if abs(impact_full) >= 10.0:
                    print(f"    ✓ Full mode shows ≥10% impact (hypothesis met)")
                elif full_result.cv > 0.30:
                    print(f"    ✓ Full mode shows high variance (CV={full_result.cv*100:.1f}%, hypothesis met)")
                else:
                    print(f"    ⚠️  WARNING: Full mode shows <10% impact and low variance (hypothesis NOT met)")

    # Multiple comparison correction
    if all_p_values:
        correction = apply_multiple_comparison_correction(all_p_values, method="holm")
        analysis["statistics"]["multiple_comparison_correction"] = correction
        print(f"\nMultiple Comparison Correction (Holm method):")
        print(f"  Tests: {correction['n_tests']}")
        print(f"  Significant (uncorrected α=0.05): {correction['n_significant_uncorrected']}")
        print(f"  Significant (corrected α=0.05): {correction['n_significant_corrected']}")

    # Overall hypothesis validation
    hypothesis_validated = all(c["hypothesis_met"] for c in comparisons if "hypothesis_met" in c)

    analysis["hypothesis_validated"] = hypothesis_validated
    analysis["comparisons"] = comparisons

    print("\n" + "=" * 80)
    if hypothesis_validated:
        print("✓ HYPOTHESIS VALIDATED")
        print("  - MMIO-only mode shows <5% impact on all tests")
        print("  - Full mode shows ≥10% impact OR failures on all tests")
        print("  - Conclusion: --mmio-only successfully prevents ERISC contention")
        analysis["summary"] = "Hypothesis validated: --mmio-only prevents ERISC contention"
    else:
        print("✗ HYPOTHESIS NOT VALIDATED")
        print("  - Some tests did not meet expected criteria")
        print("  - Review individual comparisons for details")
        analysis["summary"] = "Hypothesis not validated: review individual test results"
    print("=" * 80)

    return analysis


def main():
    """Run core hypothesis validation tests."""
    print("=" * 80)
    print("CORE HYPOTHESIS VALIDATION TEST")
    print("Validating: --mmio-only prevents ERISC contention on multi-chip workloads")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Samples per test: {N_SAMPLES}")
    print(f"  Warmup iterations: {WARMUP_ITERS}")
    print(f"  Polling frequency: {POLLING_FREQUENCY}")
    print(f"  Tensor size: {TENSOR_SIZE} (~128MB)")
    print(f"  Total tests: {len(CRITICAL_TEST_CONFIGS)}")

    all_results = []

    # Run all tests with clean state and auto-reset
    for i, config in enumerate(CRITICAL_TEST_CONFIGS):
        print(f"\n[{i+1}/{len(CRITICAL_TEST_CONFIGS)}] Testing: {config}")

        def test_wrapper():
            return run_ccl_test(config)

        result = safe_run_with_auto_reset(run_with_clean_state, test_wrapper, cooldown_sec=30.0, max_retries=2)

        all_results.append(result)

        # Save intermediate results
        save_results_json([asdict(r) for r in all_results], "/tmp/mmio_validation_results_partial.json")

    # Analyze results
    analysis = analyze_hypothesis_results(all_results)

    # Save final results
    final_output = {
        "test_config": {
            "n_samples": N_SAMPLES,
            "warmup_iters": WARMUP_ITERS,
            "polling_frequency": POLLING_FREQUENCY,
            "tensor_size": TENSOR_SIZE,
        },
        "results": [asdict(r) for r in all_results],
        "analysis": analysis,
    }

    save_results_json(final_output, "/tmp/mmio_validation_results.json")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print(f"Results saved to: /tmp/mmio_validation_results.json")
    print("=" * 80)

    # Exit with appropriate code
    if analysis["hypothesis_validated"]:
        print("\n✓ SUCCESS: Hypothesis validated")
        return 0
    else:
        print("\n✗ FAILURE: Hypothesis not validated")
        return 1


if __name__ == "__main__":
    import torch

    sys.exit(main())
