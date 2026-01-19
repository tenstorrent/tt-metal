#!/usr/bin/env python3
"""
telemetry_robust_test.py - Robust telemetry impact measurement

Features:
1. Health checks: Verify telemetry server is running before/during/after measurements
2. Multiple workload sizes: Small, Medium, Large tensor operations
3. Multiple repetitions: Run each config multiple times to measure variance
4. Interleaved runs: Alternate baseline/telemetry to control for system drift
5. Statistical robustness: Report confidence intervals and effect sizes
"""

import os
import gc
import sys
import time
import subprocess
import signal
import requests
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

# Configuration
TT_METAL_HOME = "/data/kkfernandez/tt-metal"
TELEMETRY_SERVER = "/data/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
FSD_FILE = "/data/btrzynadlowski/tt-metal/fsd.textproto"

# Test parameters
NUM_SAMPLES_PER_RUN = 100  # Samples per measurement run
NUM_REPETITIONS = 3  # Repetitions per configuration
WARMUP_ITERATIONS = 15
TELEMETRY_PORT = 9999
TELEMETRY_STARTUP_DELAY = 4


@dataclass
class WorkloadConfig:
    name: str
    shape: Tuple[int, int, int, int]
    num_tensors: int
    num_ops: int
    description: str


# Define workload sizes
WORKLOADS = {
    "small": WorkloadConfig(
        name="small",
        shape=(1, 1, 512, 512),
        num_tensors=3,
        num_ops=4,
        description="Small: 512x512 tensors (~1MB), 4 ops",
    ),
    "medium": WorkloadConfig(
        name="medium",
        shape=(1, 1, 2048, 2048),
        num_tensors=5,
        num_ops=10,
        description="Medium: 2048x2048 tensors (~16MB), 10 ops",
    ),
    "large": WorkloadConfig(
        name="large",
        shape=(1, 1, 4096, 4096),
        num_tensors=4,
        num_ops=8,
        description="Large: 4096x4096 tensors (~64MB), 8 ops",
    ),
}

# Polling intervals to test
POLLING_INTERVALS = ["1s", "100ms", "10ms", "1ms"]


@dataclass
class RunResult:
    times_ms: np.ndarray
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    telemetry_healthy: bool = True
    health_checks_passed: int = 0
    health_checks_total: int = 0


@dataclass
class ComparisonResult:
    workload: str
    polling_interval: str
    baseline_mean: float
    baseline_std: float
    telemetry_mean: float
    telemetry_std: float
    mean_impact_pct: float
    p99_baseline: float
    p99_telemetry: float
    p99_impact_pct: float
    p_value: float
    cohen_d: float
    significant: bool
    telemetry_healthy: bool


def setup_environment():
    """Lock down system for consistent measurements"""
    gc.disable()
    try:
        os.sched_setaffinity(0, {0})
        print("  [OK] CPU affinity set to core 0")
    except:
        print("  [WARN] Could not set CPU affinity")
    os.environ["TT_METAL_HOME"] = TT_METAL_HOME


def check_telemetry_health(port: int = TELEMETRY_PORT) -> bool:
    """Check if telemetry server is responding"""
    try:
        response = requests.get(f"http://localhost:{port}/api/status", timeout=2)
        return response.status_code == 200
    except:
        return False


def check_telemetry_process(proc: subprocess.Popen) -> bool:
    """Check if telemetry process is still running"""
    return proc.poll() is None


def start_telemetry_server(polling_interval: str) -> subprocess.Popen:
    """Start telemetry server and verify it's healthy"""
    cmd = [
        TELEMETRY_SERVER,
        "--port",
        str(TELEMETRY_PORT),
        "--polling-interval",
        polling_interval,
        "--fsd",
        FSD_FILE,
        "--mmio-only",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    # Wait for startup
    time.sleep(TELEMETRY_STARTUP_DELAY)

    # Verify process is running
    if not check_telemetry_process(proc):
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Telemetry server died on startup: {stderr.decode()[:500]}")

    # Verify HTTP endpoint is responding
    if not check_telemetry_health():
        stop_telemetry_server(proc)
        raise RuntimeError("Telemetry server started but HTTP endpoint not responding")

    return proc


def stop_telemetry_server(proc: subprocess.Popen):
    """Stop telemetry server gracefully"""
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()


def run_workload_iteration(device, ttnn, torch, config: WorkloadConfig):
    """Run a single workload iteration based on config"""
    shape = config.shape

    # Create tensors
    tensors = []
    for _ in range(config.num_tensors):
        t = ttnn.from_torch(torch.randn(shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tensors.append(t)

    # Run operations
    result = tensors[0]
    intermediate_results = []

    for i in range(config.num_ops):
        idx1 = i % len(tensors)
        idx2 = (i + 1) % len(tensors)

        if i % 2 == 0:
            new_result = ttnn.add(result, tensors[idx1])
        else:
            new_result = ttnn.multiply(result, tensors[idx2])

        # Keep track of intermediate results for cleanup
        if result not in tensors:
            intermediate_results.append(result)
        result = new_result

    # Cleanup
    for t in tensors:
        ttnn.deallocate(t)
    for t in intermediate_results:
        try:
            ttnn.deallocate(t)
        except:
            pass
    try:
        ttnn.deallocate(result)
    except:
        pass


def measure_workload(
    workload: WorkloadConfig, num_samples: int, telemetry_proc: Optional[subprocess.Popen] = None
) -> RunResult:
    """Measure workload with optional telemetry health checks"""
    import torch
    import ttnn

    device = ttnn.CreateDevice(device_id=0)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        run_workload_iteration(device, ttnn, torch, workload)

    # Measurement with periodic health checks
    times = []
    health_checks_passed = 0
    health_checks_total = 0
    check_interval = max(1, num_samples // 5)  # Check 5 times during run

    for i in range(num_samples):
        start = time.perf_counter()
        run_workload_iteration(device, ttnn, torch, workload)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Periodic health check if telemetry is running
        if telemetry_proc and (i + 1) % check_interval == 0:
            health_checks_total += 1
            if check_telemetry_process(telemetry_proc) and check_telemetry_health():
                health_checks_passed += 1
            else:
                print(f"    [WARN] Telemetry health check failed at sample {i+1}")

    # Final health check
    telemetry_healthy = True
    if telemetry_proc:
        health_checks_total += 1
        if check_telemetry_process(telemetry_proc) and check_telemetry_health():
            health_checks_passed += 1
        else:
            telemetry_healthy = False
            print("    [WARN] Telemetry server unhealthy at end of measurement")

    ttnn.CloseDevice(device)

    times_ms = np.array(times) * 1000

    return RunResult(
        times_ms=times_ms,
        mean=np.mean(times_ms),
        std=np.std(times_ms),
        p50=np.percentile(times_ms, 50),
        p95=np.percentile(times_ms, 95),
        p99=np.percentile(times_ms, 99),
        telemetry_healthy=telemetry_healthy,
        health_checks_passed=health_checks_passed,
        health_checks_total=health_checks_total,
    )


def run_interleaved_comparison(
    workload: WorkloadConfig, polling_interval: str, num_repetitions: int = NUM_REPETITIONS
) -> ComparisonResult:
    """Run interleaved baseline/telemetry measurements to control for drift"""

    print(f"\n  Testing {workload.name} workload with {polling_interval} polling")
    print(f"  {workload.description}")
    print(f"  Running {num_repetitions} interleaved repetitions...")

    baseline_results: List[RunResult] = []
    telemetry_results: List[RunResult] = []
    all_telemetry_healthy = True

    for rep in range(num_repetitions):
        # Alternate order to control for drift
        if rep % 2 == 0:
            order = ["baseline", "telemetry"]
        else:
            order = ["telemetry", "baseline"]

        for run_type in order:
            if run_type == "baseline":
                print(f"    Rep {rep+1}/{num_repetitions}: Baseline...", end=" ", flush=True)
                result = measure_workload(workload, NUM_SAMPLES_PER_RUN, None)
                baseline_results.append(result)
                print(f"{result.mean:.2f}ms")

            else:  # telemetry
                print(f"    Rep {rep+1}/{num_repetitions}: Telemetry ({polling_interval})...", end=" ", flush=True)
                try:
                    telem_proc = start_telemetry_server(polling_interval)
                    result = measure_workload(workload, NUM_SAMPLES_PER_RUN, telem_proc)
                    telemetry_results.append(result)

                    if not result.telemetry_healthy:
                        all_telemetry_healthy = False
                        print(f"{result.mean:.2f}ms [UNHEALTHY]")
                    else:
                        health_str = f"{result.health_checks_passed}/{result.health_checks_total}"
                        print(f"{result.mean:.2f}ms [health: {health_str}]")

                    stop_telemetry_server(telem_proc)
                except Exception as e:
                    print(f"FAILED: {e}")
                    all_telemetry_healthy = False

            # Brief pause between runs
            time.sleep(2)

    # Aggregate results
    baseline_times = np.concatenate([r.times_ms for r in baseline_results])
    telemetry_times = np.concatenate([r.times_ms for r in telemetry_results])

    baseline_mean = np.mean(baseline_times)
    baseline_std = np.std(baseline_times)
    telemetry_mean = np.mean(telemetry_times)
    telemetry_std = np.std(telemetry_times)

    baseline_p99 = np.percentile(baseline_times, 99)
    telemetry_p99 = np.percentile(telemetry_times, 99)

    # Statistical analysis
    u_stat, p_value = stats.mannwhitneyu(baseline_times, telemetry_times, alternative="two-sided")

    pooled_std = np.sqrt((baseline_std**2 + telemetry_std**2) / 2)
    cohen_d = (telemetry_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0

    mean_impact = ((telemetry_mean - baseline_mean) / baseline_mean) * 100
    p99_impact = ((telemetry_p99 - baseline_p99) / baseline_p99) * 100

    return ComparisonResult(
        workload=workload.name,
        polling_interval=polling_interval,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        telemetry_mean=telemetry_mean,
        telemetry_std=telemetry_std,
        mean_impact_pct=mean_impact,
        p99_baseline=baseline_p99,
        p99_telemetry=telemetry_p99,
        p99_impact_pct=p99_impact,
        p_value=p_value,
        cohen_d=cohen_d,
        significant=p_value < 0.05,
        telemetry_healthy=all_telemetry_healthy,
    )


def run_full_test_suite():
    """Run complete robust test suite"""

    print("=" * 80)
    print("ROBUST TELEMETRY IMPACT TEST SUITE")
    print("=" * 80)
    print()
    print("Test Parameters:")
    print(f"  Samples per run: {NUM_SAMPLES_PER_RUN}")
    print(f"  Repetitions per config: {NUM_REPETITIONS}")
    print(f"  Total samples per config: {NUM_SAMPLES_PER_RUN * NUM_REPETITIONS}")
    print(f"  Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"  Polling intervals: {POLLING_INTERVALS}")
    print()
    print("Workloads:")
    for name, config in WORKLOADS.items():
        print(f"  {config.description}")
    print()

    setup_environment()

    # Import after setup
    import torch
    import ttnn

    results: List[ComparisonResult] = []

    # Test each workload
    for workload_name in ["small", "medium", "large"]:
        workload = WORKLOADS[workload_name]

        print()
        print("-" * 80)
        print(f"WORKLOAD: {workload_name.upper()}")
        print("-" * 80)

        for polling_interval in POLLING_INTERVALS:
            try:
                result = run_interleaved_comparison(workload, polling_interval)
                results.append(result)

                # Print immediate summary
                health_icon = "[OK]" if result.telemetry_healthy else "[UNHEALTHY]"
                sig_icon = "*" if result.significant else ""
                print(
                    f"    Result: {result.mean_impact_pct:+.2f}% mean, {result.p99_impact_pct:+.2f}% P99 {health_icon}{sig_icon}"
                )

            except Exception as e:
                print(f"    FAILED: {e}")

    # Final summary
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Group by workload
    for workload_name in ["small", "medium", "large"]:
        workload_results = [r for r in results if r.workload == workload_name]
        if not workload_results:
            continue

        print(f"\n{workload_name.upper()} WORKLOAD:")
        baseline_mean = workload_results[0].baseline_mean
        print(f"  Baseline: {baseline_mean:.2f} ms (std: {workload_results[0].baseline_std:.2f})")
        print()
        print(
            f"  {'Polling':<10} | {'Mean Impact':>12} | {'P99 Impact':>12} | {'P-value':>12} | {'Health':>8} | {'Sig?':>5}"
        )
        print("  " + "-" * 70)

        for r in workload_results:
            health = "OK" if r.telemetry_healthy else "BAD"
            sig = "YES" if r.significant else "no"
            print(
                f"  {r.polling_interval:<10} | {r.mean_impact_pct:>+11.2f}% | {r.p99_impact_pct:>+11.2f}% | {r.p_value:>12.2e} | {health:>8} | {sig:>5}"
            )

    # Overall conclusions
    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    healthy_results = [r for r in results if r.telemetry_healthy]
    unhealthy_count = len(results) - len(healthy_results)

    if unhealthy_count > 0:
        print(f"\n[WARN] {unhealthy_count} configurations had telemetry health issues")

    significant_results = [r for r in healthy_results if r.significant]

    if significant_results:
        print(
            f"\nStatistically significant impacts detected in {len(significant_results)}/{len(healthy_results)} healthy configs:"
        )

        # Group by impact magnitude
        high_impact = [r for r in significant_results if abs(r.mean_impact_pct) > 5]
        moderate_impact = [r for r in significant_results if 1 < abs(r.mean_impact_pct) <= 5]
        low_impact = [r for r in significant_results if abs(r.mean_impact_pct) <= 1]

        if high_impact:
            print("\n  HIGH IMPACT (>5% mean):")
            for r in high_impact:
                print(
                    f"    - {r.workload}/{r.polling_interval}: {r.mean_impact_pct:+.2f}% mean, {r.p99_impact_pct:+.2f}% P99"
                )

        if moderate_impact:
            print("\n  MODERATE IMPACT (1-5% mean):")
            for r in moderate_impact:
                print(
                    f"    - {r.workload}/{r.polling_interval}: {r.mean_impact_pct:+.2f}% mean, {r.p99_impact_pct:+.2f}% P99"
                )

        if low_impact:
            print("\n  LOW IMPACT (<1% mean):")
            for r in low_impact:
                print(
                    f"    - {r.workload}/{r.polling_interval}: {r.mean_impact_pct:+.2f}% mean, {r.p99_impact_pct:+.2f}% P99"
                )
    else:
        print("\nNo statistically significant impacts detected in healthy configurations.")

    # Save results
    output = {
        "metadata": {
            "samples_per_run": NUM_SAMPLES_PER_RUN,
            "repetitions": NUM_REPETITIONS,
            "total_samples_per_config": NUM_SAMPLES_PER_RUN * NUM_REPETITIONS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": [
            {
                "workload": r.workload,
                "polling_interval": r.polling_interval,
                "baseline_mean_ms": r.baseline_mean,
                "baseline_std_ms": r.baseline_std,
                "telemetry_mean_ms": r.telemetry_mean,
                "telemetry_std_ms": r.telemetry_std,
                "mean_impact_pct": r.mean_impact_pct,
                "p99_baseline_ms": r.p99_baseline,
                "p99_telemetry_ms": r.p99_telemetry,
                "p99_impact_pct": r.p99_impact_pct,
                "p_value": r.p_value,
                "cohen_d": r.cohen_d,
                "significant": r.significant,
                "telemetry_healthy": r.telemetry_healthy,
            }
            for r in results
        ],
    }

    with open("/tmp/telemetry_robust_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: /tmp/telemetry_robust_results.json")

    gc.enable()
    return results


if __name__ == "__main__":
    results = run_full_test_suite()
