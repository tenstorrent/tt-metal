#!/usr/bin/env python3
"""
telemetry_impact_pipeline.py - Comprehensive telemetry impact measurement pipeline

Tests both single-chip and multi-chip workloads across various telemetry configurations
with rigorous statistical verification.

Features:
1. Single-chip workloads: Various tensor sizes with compute operations
2. Multi-chip workloads: CCL operations (AllGather, ReduceScatter) using fabric
3. Multiple telemetry modes: --mmio-only vs full mode
4. Multiple polling intervals: 1s, 100ms, 10ms, 1ms
5. Statistical rigor: Mann-Whitney U tests, Cohen's d effect size, confidence intervals
6. Interleaved measurements to control for system drift
7. Health monitoring throughout tests
"""

import os
import gc
import sys
import time
import subprocess
import signal
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

# Configuration
TT_METAL_HOME = "/data/kkfernandez/tt-metal"
TELEMETRY_SERVER = "/data/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
FSD_FILE = "/data/kkfernandez/tt-metal/fsd.textproto"
TELEMETRY_PORT = 5555
TELEMETRY_STARTUP_DELAY = 3

# Test parameters (can be overridden via command line)
DEFAULT_SAMPLES_PER_RUN = 100
DEFAULT_REPETITIONS = 3
DEFAULT_WARMUP_ITERATIONS = 15

# Polling intervals to test (granular from 1s down to 1ms)
POLLING_INTERVALS = ["1s", "500ms", "100ms", "50ms", "10ms", "5ms", "1ms"]

# Telemetry modes
TELEMETRY_MODES = ["mmio_only", "full"]


@dataclass
class WorkloadConfig:
    name: str
    workload_type: str  # "single_chip" or "multi_chip"
    description: str
    params: Dict[str, Any] = field(default_factory=dict)


# Define workloads
SINGLE_CHIP_WORKLOADS = {
    "small": WorkloadConfig(
        name="small",
        workload_type="single_chip",
        description="Small: 512x512 tensors, 4 ops (~5ms)",
        params={"shape": (1, 1, 512, 512), "num_tensors": 3, "num_ops": 4},
    ),
    "medium": WorkloadConfig(
        name="medium",
        workload_type="single_chip",
        description="Medium: 2048x2048 tensors, 10 ops (~140ms)",
        params={"shape": (1, 1, 2048, 2048), "num_tensors": 5, "num_ops": 10},
    ),
    "large": WorkloadConfig(
        name="large",
        workload_type="single_chip",
        description="Large: 4096x4096 tensors, 8 ops (~670ms)",
        params={"shape": (1, 1, 4096, 4096), "num_tensors": 4, "num_ops": 8},
    ),
}

MULTI_CHIP_WORKLOADS = {
    "allgather_small": WorkloadConfig(
        name="allgather_small",
        workload_type="multi_chip",
        description="AllGather: 32x128 tensor, dim=3, 4 devices",
        params={
            "op": "all_gather",
            "shape": (1, 1, 32, 128),
            "dim": 3,
            "num_devices": 4,
            "num_iters": 8,
        },
    ),
    "allgather_medium": WorkloadConfig(
        name="allgather_medium",
        workload_type="multi_chip",
        description="AllGather: 32x10240 tensor, dim=3, 8 devices",
        params={
            "op": "all_gather",
            "shape": (1, 1, 32, 10240),
            "dim": 3,
            "num_devices": 8,
            "num_iters": 8,
        },
    ),
    "allgather_large": WorkloadConfig(
        name="allgather_large",
        workload_type="multi_chip",
        description="AllGather: 32x30720 tensor, dim=3, 8 devices",
        params={
            "op": "all_gather",
            "shape": (1, 1, 32, 30720),
            "dim": 3,
            "num_devices": 8,
            "num_iters": 8,
        },
    ),
}


@dataclass
class MeasurementResult:
    times_ms: np.ndarray
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    min_val: float
    max_val: float
    cv: float  # coefficient of variation
    telemetry_healthy: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    workload_name: str
    workload_type: str
    telemetry_mode: str
    polling_interval: str
    baseline: MeasurementResult
    telemetry: MeasurementResult
    mean_impact_pct: float
    p99_impact_pct: float
    p_value: float
    cohen_d: float
    ci_95_lower: float
    ci_95_upper: float
    significant: bool
    practical_significance: str  # "none", "small", "medium", "large"


def check_telemetry_health(port: int = TELEMETRY_PORT, timeout: float = 2.0) -> bool:
    """Check if telemetry server is responding"""
    try:
        import requests

        response = requests.get(f"http://localhost:{port}/api/status", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def start_telemetry_server(polling_interval: str, mode: str = "mmio_only") -> subprocess.Popen:
    """Start telemetry server with specified configuration"""
    cmd = [
        TELEMETRY_SERVER,
        "--port",
        str(TELEMETRY_PORT),
        "--polling-interval",
        polling_interval,
        "--fsd",
        FSD_FILE,
    ]

    if mode == "mmio_only":
        cmd.append("--mmio-only")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    time.sleep(TELEMETRY_STARTUP_DELAY)

    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Telemetry server died on startup: {stderr.decode()[:500]}")

    if not check_telemetry_health():
        stop_telemetry_server(proc)
        raise RuntimeError("Telemetry server started but HTTP endpoint not responding")

    return proc


def stop_telemetry_server(proc: subprocess.Popen):
    """Stop telemetry server gracefully"""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass


def run_single_chip_iteration(device, ttnn, torch, config: WorkloadConfig):
    """Run a single iteration of single-chip workload"""
    params = config.params
    shape = params["shape"]

    tensors = []
    for _ in range(params["num_tensors"]):
        t = ttnn.from_torch(torch.randn(shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tensors.append(t)

    result = tensors[0]
    intermediates = []

    for i in range(params["num_ops"]):
        idx1 = i % len(tensors)
        idx2 = (i + 1) % len(tensors)

        if i % 2 == 0:
            new_result = ttnn.add(result, tensors[idx1])
        else:
            new_result = ttnn.multiply(result, tensors[idx2])

        if result not in tensors:
            intermediates.append(result)
        result = new_result

    # Cleanup
    for t in tensors:
        ttnn.deallocate(t)
    for t in intermediates:
        try:
            ttnn.deallocate(t)
        except:
            pass
    try:
        ttnn.deallocate(result)
    except:
        pass


def run_multi_chip_iteration(mesh_device, ttnn, torch, config: WorkloadConfig):
    """Run a single iteration of multi-chip CCL workload"""
    params = config.params
    op_type = params["op"]
    shape = params["shape"]
    dim = params["dim"]
    num_devices = params["num_devices"]

    # Calculate per-device shape for input
    per_device_shape = list(shape)
    per_device_shape[dim] = shape[dim] // num_devices

    # Create input tensor on mesh
    input_tensor = torch.randn(per_device_shape)

    tt_input = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    if op_type == "all_gather":
        output = ttnn.all_gather(
            tt_input,
            dim=dim,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
    else:
        raise ValueError(f"Unknown op type: {op_type}")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(output)


def measure_single_chip_workload(
    config: WorkloadConfig, num_samples: int, warmup_iters: int, telemetry_proc: Optional[subprocess.Popen] = None
) -> MeasurementResult:
    """Measure single-chip workload performance"""
    import torch
    import ttnn

    errors = []

    device = ttnn.CreateDevice(device_id=0)

    # Warmup
    for _ in range(warmup_iters):
        run_single_chip_iteration(device, ttnn, torch, config)

    # Measurement
    times = []
    for i in range(num_samples):
        start = time.perf_counter()
        run_single_chip_iteration(device, ttnn, torch, config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Periodic health check
        if telemetry_proc and (i + 1) % (num_samples // 4) == 0:
            if not check_telemetry_health():
                errors.append(f"Health check failed at sample {i+1}")

    ttnn.CloseDevice(device)

    times_ms = np.array(times) * 1000
    mean = np.mean(times_ms)
    std = np.std(times_ms)

    telemetry_healthy = True
    if telemetry_proc:
        telemetry_healthy = check_telemetry_health()

    return MeasurementResult(
        times_ms=times_ms,
        mean=mean,
        std=std,
        p50=np.percentile(times_ms, 50),
        p95=np.percentile(times_ms, 95),
        p99=np.percentile(times_ms, 99),
        min_val=np.min(times_ms),
        max_val=np.max(times_ms),
        cv=(std / mean) * 100 if mean > 0 else 0,
        telemetry_healthy=telemetry_healthy,
        errors=errors,
    )


def measure_multi_chip_workload(
    config: WorkloadConfig, num_samples: int, warmup_iters: int, telemetry_proc: Optional[subprocess.Popen] = None
) -> MeasurementResult:
    """Measure multi-chip CCL workload performance"""
    import torch
    import ttnn
    from ttnn import FabricConfig

    errors = []
    params = config.params
    num_devices = params["num_devices"]
    num_iters = params.get("num_iters", 1)

    # Open mesh device with fabric enabled
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, num_devices),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.WORKER,
            ttnn.DispatchCoreAxis.COL,
        ),
        fabric_config=FabricConfig.FABRIC_1D,
    )

    # Warmup
    for _ in range(warmup_iters):
        for _ in range(num_iters):
            run_multi_chip_iteration(mesh_device, ttnn, torch, config)

    # Measurement
    times = []
    for i in range(num_samples):
        start = time.perf_counter()
        for _ in range(num_iters):
            run_multi_chip_iteration(mesh_device, ttnn, torch, config)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Periodic health check
        if telemetry_proc and (i + 1) % max(1, num_samples // 4) == 0:
            if not check_telemetry_health():
                errors.append(f"Health check failed at sample {i+1}")

    ttnn.close_mesh_device(mesh_device)

    times_ms = np.array(times) * 1000
    mean = np.mean(times_ms)
    std = np.std(times_ms)

    telemetry_healthy = True
    if telemetry_proc:
        telemetry_healthy = check_telemetry_health()

    return MeasurementResult(
        times_ms=times_ms,
        mean=mean,
        std=std,
        p50=np.percentile(times_ms, 50),
        p95=np.percentile(times_ms, 95),
        p99=np.percentile(times_ms, 99),
        min_val=np.min(times_ms),
        max_val=np.max(times_ms),
        cv=(std / mean) * 100 if mean > 0 else 0,
        telemetry_healthy=telemetry_healthy,
        errors=errors,
    )


def compute_statistics(
    baseline: MeasurementResult,
    telemetry: MeasurementResult,
    workload_name: str,
    workload_type: str,
    telemetry_mode: str,
    polling_interval: str,
) -> ComparisonResult:
    """Compute statistical comparison between baseline and telemetry runs"""

    baseline_times = baseline.times_ms
    telemetry_times = telemetry.times_ms

    # Mann-Whitney U test (non-parametric, robust to non-normality)
    u_stat, p_value = stats.mannwhitneyu(baseline_times, telemetry_times, alternative="two-sided")

    # Effect size: Cohen's d
    pooled_std = np.sqrt((baseline.std**2 + telemetry.std**2) / 2)
    cohen_d = (telemetry.mean - baseline.mean) / pooled_std if pooled_std > 0 else 0

    # 95% confidence interval for the difference (bootstrap)
    n_bootstrap = 1000
    diffs = []
    for _ in range(n_bootstrap):
        b_sample = np.random.choice(baseline_times, len(baseline_times), replace=True)
        t_sample = np.random.choice(telemetry_times, len(telemetry_times), replace=True)
        diffs.append(np.mean(t_sample) - np.mean(b_sample))
    ci_95_lower = np.percentile(diffs, 2.5)
    ci_95_upper = np.percentile(diffs, 97.5)

    # Impact percentages
    mean_impact = ((telemetry.mean - baseline.mean) / baseline.mean) * 100
    p99_impact = ((telemetry.p99 - baseline.p99) / baseline.p99) * 100

    # Practical significance interpretation
    abs_cohen_d = abs(cohen_d)
    if abs_cohen_d < 0.2:
        practical_significance = "none"
    elif abs_cohen_d < 0.5:
        practical_significance = "small"
    elif abs_cohen_d < 0.8:
        practical_significance = "medium"
    else:
        practical_significance = "large"

    return ComparisonResult(
        workload_name=workload_name,
        workload_type=workload_type,
        telemetry_mode=telemetry_mode,
        polling_interval=polling_interval,
        baseline=baseline,
        telemetry=telemetry,
        mean_impact_pct=mean_impact,
        p99_impact_pct=p99_impact,
        p_value=p_value,
        cohen_d=cohen_d,
        ci_95_lower=ci_95_lower,
        ci_95_upper=ci_95_upper,
        significant=p_value < 0.05,
        practical_significance=practical_significance,
    )


def run_comparison(
    config: WorkloadConfig,
    telemetry_mode: str,
    polling_interval: str,
    num_samples: int,
    num_repetitions: int,
    warmup_iters: int,
) -> ComparisonResult:
    """Run interleaved comparison between baseline and telemetry"""

    print(f"    Testing {config.name} with {telemetry_mode}/{polling_interval}...")

    baseline_results = []
    telemetry_results = []

    measure_func = (
        measure_single_chip_workload if config.workload_type == "single_chip" else measure_multi_chip_workload
    )

    for rep in range(num_repetitions):
        # Alternate order to control for drift
        order = ["baseline", "telemetry"] if rep % 2 == 0 else ["telemetry", "baseline"]

        for run_type in order:
            if run_type == "baseline":
                result = measure_func(config, num_samples, warmup_iters, None)
                baseline_results.append(result)
                print(f"      Rep {rep+1} baseline: {result.mean:.2f}ms (CV: {result.cv:.1f}%)")
            else:
                try:
                    telem_proc = start_telemetry_server(polling_interval, telemetry_mode)
                    result = measure_func(config, num_samples, warmup_iters, telem_proc)
                    telemetry_results.append(result)
                    health_str = "OK" if result.telemetry_healthy else "UNHEALTHY"
                    print(f"      Rep {rep+1} telemetry: {result.mean:.2f}ms [{health_str}]")
                    stop_telemetry_server(telem_proc)
                except Exception as e:
                    print(f"      Rep {rep+1} telemetry: FAILED - {e}")
                    # Create dummy failed result
                    telemetry_results.append(
                        MeasurementResult(
                            times_ms=np.array([0]),
                            mean=0,
                            std=0,
                            p50=0,
                            p95=0,
                            p99=0,
                            min_val=0,
                            max_val=0,
                            cv=0,
                            telemetry_healthy=False,
                            errors=[str(e)],
                        )
                    )

            time.sleep(1)  # Brief pause between runs

    # Aggregate results
    baseline_times = np.concatenate([r.times_ms for r in baseline_results])
    telemetry_times = np.concatenate([r.times_ms for r in telemetry_results if r.telemetry_healthy])

    if len(telemetry_times) == 0:
        print(f"      All telemetry runs failed!")
        telemetry_times = np.array([0])

    baseline_agg = MeasurementResult(
        times_ms=baseline_times,
        mean=np.mean(baseline_times),
        std=np.std(baseline_times),
        p50=np.percentile(baseline_times, 50),
        p95=np.percentile(baseline_times, 95),
        p99=np.percentile(baseline_times, 99),
        min_val=np.min(baseline_times),
        max_val=np.max(baseline_times),
        cv=(np.std(baseline_times) / np.mean(baseline_times)) * 100 if np.mean(baseline_times) > 0 else 0,
        telemetry_healthy=True,
        errors=[],
    )

    telemetry_agg = MeasurementResult(
        times_ms=telemetry_times,
        mean=np.mean(telemetry_times),
        std=np.std(telemetry_times),
        p50=np.percentile(telemetry_times, 50),
        p95=np.percentile(telemetry_times, 95),
        p99=np.percentile(telemetry_times, 99),
        min_val=np.min(telemetry_times),
        max_val=np.max(telemetry_times),
        cv=(np.std(telemetry_times) / np.mean(telemetry_times)) * 100 if np.mean(telemetry_times) > 0 else 0,
        telemetry_healthy=all(r.telemetry_healthy for r in telemetry_results),
        errors=[e for r in telemetry_results for e in r.errors],
    )

    return compute_statistics(
        baseline_agg, telemetry_agg, config.name, config.workload_type, telemetry_mode, polling_interval
    )


def run_pipeline(
    workloads: Dict[str, WorkloadConfig],
    telemetry_modes: List[str],
    polling_intervals: List[str],
    num_samples: int,
    num_repetitions: int,
    warmup_iters: int,
) -> List[ComparisonResult]:
    """Run the full testing pipeline"""

    results = []

    for workload_name, config in workloads.items():
        print(f"\n  === {workload_name.upper()} ({config.workload_type}) ===")
        print(f"  {config.description}")

        for mode in telemetry_modes:
            for interval in polling_intervals:
                try:
                    result = run_comparison(config, mode, interval, num_samples, num_repetitions, warmup_iters)
                    results.append(result)

                    # Print immediate summary
                    sig_str = "*" if result.significant else ""
                    print(
                        f"      -> Impact: {result.mean_impact_pct:+.2f}% mean, "
                        f"{result.p99_impact_pct:+.2f}% P99 "
                        f"(p={result.p_value:.2e}, d={result.cohen_d:.2f}){sig_str}"
                    )
                except Exception as e:
                    print(f"      -> FAILED: {e}")

    return results


def generate_report(results: List[ComparisonResult], output_path: str):
    """Generate comprehensive report from results"""

    report = []
    report.append("=" * 80)
    report.append("TELEMETRY IMPACT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total configurations tested: {len(results)}")

    # Summary statistics
    healthy_results = [r for r in results if r.telemetry.telemetry_healthy]
    significant_results = [r for r in healthy_results if r.significant]

    report.append(f"\nHealthy configurations: {len(healthy_results)}/{len(results)}")
    report.append(f"Statistically significant results: {len(significant_results)}/{len(healthy_results)}")

    # Group by workload type
    for workload_type in ["single_chip", "multi_chip"]:
        type_results = [r for r in healthy_results if r.workload_type == workload_type]
        if not type_results:
            continue

        report.append(f"\n\n{'=' * 80}")
        report.append(f"{workload_type.upper().replace('_', '-')} WORKLOADS")
        report.append("=" * 80)

        # Group by workload
        workload_names = sorted(set(r.workload_name for r in type_results))

        for workload_name in workload_names:
            workload_results = [r for r in type_results if r.workload_name == workload_name]

            report.append(f"\n--- {workload_name} ---")
            report.append(
                f"Baseline mean: {workload_results[0].baseline.mean:.2f}ms "
                f"(std: {workload_results[0].baseline.std:.2f}ms, "
                f"CV: {workload_results[0].baseline.cv:.1f}%)"
            )

            report.append(
                f"\n{'Mode':<12} {'Polling':<8} {'Mean %':>10} {'P99 %':>10} "
                f"{'p-value':>12} {'Cohen d':>10} {'95% CI':>20} {'Sig?':>6}"
            )
            report.append("-" * 90)

            for r in workload_results:
                ci_str = f"[{r.ci_95_lower:+.2f}, {r.ci_95_upper:+.2f}]"
                sig = "YES*" if r.significant else "no"
                report.append(
                    f"{r.telemetry_mode:<12} {r.polling_interval:<8} "
                    f"{r.mean_impact_pct:>+10.2f} {r.p99_impact_pct:>+10.2f} "
                    f"{r.p_value:>12.2e} {r.cohen_d:>+10.2f} {ci_str:>20} {sig:>6}"
                )

    # Conclusions
    report.append(f"\n\n{'=' * 80}")
    report.append("CONCLUSIONS")
    report.append("=" * 80)

    # Analyze by telemetry mode
    for mode in TELEMETRY_MODES:
        mode_results = [r for r in healthy_results if r.telemetry_mode == mode]
        if not mode_results:
            continue

        mode_significant = [r for r in mode_results if r.significant]
        avg_impact = np.mean([r.mean_impact_pct for r in mode_results])
        max_impact = max(r.mean_impact_pct for r in mode_results)

        report.append(f"\n{mode.upper()} MODE:")
        report.append(f"  Significant impacts: {len(mode_significant)}/{len(mode_results)}")
        report.append(f"  Average mean impact: {avg_impact:+.2f}%")
        report.append(f"  Maximum mean impact: {max_impact:+.2f}%")

        # By practical significance
        by_practical = {}
        for r in mode_results:
            by_practical.setdefault(r.practical_significance, []).append(r)

        report.append(f"  By practical significance:")
        for level in ["none", "small", "medium", "large"]:
            if level in by_practical:
                report.append(f"    {level}: {len(by_practical[level])} configurations")

    # Key findings
    report.append(f"\n\nKEY FINDINGS:")

    # Find any high-impact configurations
    high_impact = [r for r in healthy_results if abs(r.mean_impact_pct) > 5]
    if high_impact:
        report.append(f"\n  HIGH IMPACT CONFIGURATIONS (>5% mean impact):")
        for r in high_impact:
            report.append(
                f"    - {r.workload_name}/{r.telemetry_mode}/{r.polling_interval}: " f"{r.mean_impact_pct:+.2f}%"
            )
    else:
        report.append(f"\n  No high-impact (>5%) configurations detected.")

    # mmio-only vs full comparison
    mmio_results = [r for r in healthy_results if r.telemetry_mode == "mmio_only"]
    full_results = [r for r in healthy_results if r.telemetry_mode == "full"]

    if mmio_results and full_results:
        mmio_avg = np.mean([abs(r.mean_impact_pct) for r in mmio_results])
        full_avg = np.mean([abs(r.mean_impact_pct) for r in full_results])

        report.append(f"\n  MODE COMPARISON:")
        report.append(f"    mmio-only average |impact|: {mmio_avg:.2f}%")
        report.append(f"    full mode average |impact|: {full_avg:.2f}%")

        if full_avg > mmio_avg * 1.5:
            report.append(f"    -> Full mode shows notably higher impact than mmio-only")
        else:
            report.append(f"    -> Both modes show similar impact levels")

    # Single vs multi-chip comparison
    single_results = [r for r in healthy_results if r.workload_type == "single_chip"]
    multi_results = [r for r in healthy_results if r.workload_type == "multi_chip"]

    if single_results and multi_results:
        single_avg = np.mean([abs(r.mean_impact_pct) for r in single_results])
        multi_avg = np.mean([abs(r.mean_impact_pct) for r in multi_results])

        report.append(f"\n  WORKLOAD TYPE COMPARISON:")
        report.append(f"    Single-chip average |impact|: {single_avg:.2f}%")
        report.append(f"    Multi-chip average |impact|: {multi_avg:.2f}%")

        if multi_avg > single_avg * 1.5:
            report.append(f"    -> Multi-chip workloads show higher telemetry sensitivity")
        else:
            report.append(f"    -> Both workload types show similar telemetry sensitivity")

    report_text = "\n".join(report)

    with open(output_path, "w") as f:
        f.write(report_text)

    return report_text


def save_results_json(results: List[ComparisonResult], output_path: str):
    """Save detailed results as JSON"""

    def to_python(val):
        """Convert numpy types to native Python types"""
        if isinstance(val, (np.integer, np.floating)):
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
        return val

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "telemetry_server": TELEMETRY_SERVER,
            "fsd_file": FSD_FILE,
        },
        "results": [
            {
                "workload_name": r.workload_name,
                "workload_type": r.workload_type,
                "telemetry_mode": r.telemetry_mode,
                "polling_interval": r.polling_interval,
                "baseline": {
                    "mean_ms": to_python(r.baseline.mean),
                    "std_ms": to_python(r.baseline.std),
                    "p50_ms": to_python(r.baseline.p50),
                    "p95_ms": to_python(r.baseline.p95),
                    "p99_ms": to_python(r.baseline.p99),
                    "min_ms": to_python(r.baseline.min_val),
                    "max_ms": to_python(r.baseline.max_val),
                    "cv_pct": to_python(r.baseline.cv),
                    "n_samples": int(len(r.baseline.times_ms)),
                },
                "telemetry": {
                    "mean_ms": to_python(r.telemetry.mean),
                    "std_ms": to_python(r.telemetry.std),
                    "p50_ms": to_python(r.telemetry.p50),
                    "p95_ms": to_python(r.telemetry.p95),
                    "p99_ms": to_python(r.telemetry.p99),
                    "min_ms": to_python(r.telemetry.min_val),
                    "max_ms": to_python(r.telemetry.max_val),
                    "cv_pct": to_python(r.telemetry.cv),
                    "n_samples": int(len(r.telemetry.times_ms)),
                    "healthy": bool(r.telemetry.telemetry_healthy),
                },
                "analysis": {
                    "mean_impact_pct": to_python(r.mean_impact_pct),
                    "p99_impact_pct": to_python(r.p99_impact_pct),
                    "p_value": to_python(r.p_value),
                    "cohen_d": to_python(r.cohen_d),
                    "ci_95_lower_ms": to_python(r.ci_95_lower),
                    "ci_95_upper_ms": to_python(r.ci_95_upper),
                    "statistically_significant": bool(r.significant),
                    "practical_significance": r.practical_significance,
                },
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Telemetry Impact Testing Pipeline")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_RUN, help="Samples per measurement run")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="Repetitions per configuration")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS, help="Warmup iterations")
    parser.add_argument("--single-chip-only", action="store_true", help="Only test single-chip workloads")
    parser.add_argument("--multi-chip-only", action="store_true", help="Only test multi-chip workloads")
    parser.add_argument("--output-dir", type=str, default="/tmp", help="Output directory for results")
    parser.add_argument(
        "--polling-intervals", type=str, nargs="+", default=POLLING_INTERVALS, help="Polling intervals to test"
    )
    parser.add_argument("--modes", type=str, nargs="+", default=TELEMETRY_MODES, help="Telemetry modes to test")

    args = parser.parse_args()

    print("=" * 80)
    print("TELEMETRY IMPACT TESTING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Samples per run: {args.samples}")
    print(f"  Repetitions: {args.repetitions}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Polling intervals: {args.polling_intervals}")
    print(f"  Telemetry modes: {args.modes}")
    print(f"  Output directory: {args.output_dir}")

    # Select workloads
    workloads = {}
    if not args.multi_chip_only:
        workloads.update(SINGLE_CHIP_WORKLOADS)
    if not args.single_chip_only:
        workloads.update(MULTI_CHIP_WORKLOADS)

    print(f"\nWorkloads to test:")
    for name, config in workloads.items():
        print(f"  - {name}: {config.description}")

    total_configs = len(workloads) * len(args.modes) * len(args.polling_intervals)
    print(f"\nTotal configurations: {total_configs}")
    print()

    # Setup environment
    gc.disable()
    os.environ["TT_METAL_HOME"] = TT_METAL_HOME

    # Run pipeline
    print("\nStarting tests...")
    results = run_pipeline(
        workloads,
        args.modes,
        args.polling_intervals,
        args.samples,
        args.repetitions,
        args.warmup,
    )

    gc.enable()

    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{args.output_dir}/telemetry_impact_report_{timestamp}.txt"
    json_path = f"{args.output_dir}/telemetry_impact_results_{timestamp}.json"

    print("\n\nGenerating report...")
    report_text = generate_report(results, report_path)
    print(report_text)

    save_results_json(results, json_path)

    print(f"\n\nResults saved to:")
    print(f"  Report: {report_path}")
    print(f"  JSON:   {json_path}")

    return results


if __name__ == "__main__":
    main()
