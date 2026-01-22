"""
Shared utilities for telemetry performance benchmarking.

Provides:
- Device state management and cleanup
- Validated telemetry startup
- Adaptive warmup
- Statistical analysis functions
- Multiple comparison correction
"""

import time
import subprocess
import requests
import json
import os
import signal
from statistics import mean, stdev
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.stats import kendalltau, mannwhitneyu, shapiro

# Try to import statsmodels, but make it optional
try:
    from statsmodels.stats.multitest import multipletests

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Multiple comparison correction will be skipped.")

# Configuration
TELEMETRY_PORT = 7070
# Detect tt-metal root (parent of telemetry-perf directory)
import pathlib

_SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
_TT_METAL_ROOT = _SCRIPT_DIR.parent
TELEMETRY_BINARY = str(_TT_METAL_ROOT / "build_Release" / "tt_telemetry" / "tt_telemetry_server")
TT_SMI_BINARY = "tt-smi"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark configuration."""

    operation: str
    config: Dict[str, Any]
    samples: List[float]
    mean_time: float
    std_time: float
    median_time: float
    min_time: float
    max_time: float
    cv: float  # Coefficient of variation
    errors: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Calculate derived statistics."""
        if self.samples and len(self.samples) > 0:
            self.mean_time = mean(self.samples)
            self.std_time = stdev(self.samples) if len(self.samples) > 1 else 0.0
            self.median_time = float(np.median(self.samples))
            self.min_time = min(self.samples)
            self.max_time = max(self.samples)
            self.cv = self.std_time / self.mean_time if self.mean_time > 0 else 0.0


class DeviceManager:
    """Manages device state and cleanup."""

    @staticmethod
    def cleanup_all_devices():
        """Close all TT devices and cleanup."""
        # Note: Individual scripts will close their own devices explicitly
        # This is a no-op placeholder for compatibility
        pass

    @staticmethod
    def reset_devices(device_ids: List[int] = [0, 1, 2, 3]):
        """Reset devices via tt-smi."""
        print(f"Resetting devices: {device_ids}")
        for dev_id in device_ids:
            try:
                result = subprocess.run([TT_SMI_BINARY, "-r", str(dev_id)], timeout=120, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Failed to reset device {dev_id}: {result.stderr}")
                time.sleep(10)  # Wait between resets
            except subprocess.TimeoutExpired:
                print(f"Warning: Timeout resetting device {dev_id}")
            except Exception as e:
                print(f"Warning: Error resetting device {dev_id}: {e}")

    @staticmethod
    def thermal_cooldown(duration_sec: float = 30.0):
        """Wait for thermal equilibration."""
        print(f"Thermal cool-down for {duration_sec}s...")
        time.sleep(duration_sec)


def run_with_clean_state(test_func: Callable, *args, cooldown_sec: float = 30.0, **kwargs) -> Any:
    """
    Run a test function with clean device state.

    Args:
        test_func: Function to run
        *args: Arguments for test_func
        cooldown_sec: Cool-down period duration
        **kwargs: Keyword arguments for test_func

    Returns:
        Result from test_func
    """
    # Cleanup before test
    DeviceManager.cleanup_all_devices()
    DeviceManager.thermal_cooldown(cooldown_sec)

    try:
        # Run test
        result = test_func(*args, **kwargs)
        return result
    finally:
        # Cleanup after test
        DeviceManager.cleanup_all_devices()


def safe_run_with_auto_reset(test_func: Callable, *args, max_retries: int = 2, **kwargs) -> Any:
    """
    Run test with automatic device reset on failure.

    Args:
        test_func: Function to run
        *args: Arguments for test_func
        max_retries: Maximum retry attempts
        **kwargs: Keyword arguments for test_func

    Returns:
        Result from test_func or error result
    """
    for attempt in range(max_retries):
        try:
            result = test_func(*args, **kwargs)

            # Check if result indicates errors
            if hasattr(result, "errors") and result.errors:
                print(f"Test failed with errors, attempting device reset (attempt {attempt + 1}/{max_retries})...")
                DeviceManager.reset_devices()
                DeviceManager.thermal_cooldown(30.0)
                continue

            return result

        except Exception as e:
            print(f"Exception during test: {e}, resetting devices (attempt {attempt + 1}/{max_retries})...")
            DeviceManager.reset_devices()
            DeviceManager.thermal_cooldown(30.0)

            if attempt == max_retries - 1:
                # Return error result on final failure
                return create_error_result(str(e))

    return create_error_result("Max retries exceeded")


def create_error_result(error_msg: str) -> BenchmarkResult:
    """Create a BenchmarkResult indicating an error."""
    return BenchmarkResult(
        operation="error",
        config={},
        samples=[],
        mean_time=0.0,
        std_time=0.0,
        median_time=0.0,
        min_time=0.0,
        max_time=0.0,
        cv=0.0,
        errors=[error_msg],
        metadata={},
    )


class TelemetryManager:
    """Manages telemetry server lifecycle."""

    @staticmethod
    def start_telemetry(polling_interval: str, mmio_only: bool = False) -> Optional[subprocess.Popen]:
        """
        Start telemetry server without validation.

        Args:
            polling_interval: Polling interval (e.g., "100ms", "1s")
            mmio_only: Whether to use --mmio-only flag

        Returns:
            Popen object for telemetry process
        """
        cmd = [TELEMETRY_BINARY, "--logging-interval", polling_interval, "--port", str(TELEMETRY_PORT)]

        if mmio_only:
            cmd.append("--mmio-only")

        print(f"Starting telemetry: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return proc

    @staticmethod
    def check_telemetry_health() -> bool:
        """Check if telemetry server is healthy."""
        try:
            response = requests.get(f"http://localhost:{TELEMETRY_PORT}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def fetch_metrics() -> Optional[Dict]:
        """Fetch current metrics from telemetry server."""
        try:
            response = requests.get(f"http://localhost:{TELEMETRY_PORT}/api/metrics", timeout=2)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    @staticmethod
    def start_telemetry_validated(
        polling_interval: str, mmio_only: bool = False, stabilization_sec: float = 5.0
    ) -> subprocess.Popen:
        """
        Start telemetry server with validation.

        Args:
            polling_interval: Polling interval (e.g., "100ms", "1s")
            mmio_only: Whether to use --mmio-only flag
            stabilization_sec: Additional stabilization period after startup

        Returns:
            Validated Popen object for telemetry process

        Raises:
            RuntimeError: If telemetry fails to start properly
        """
        proc = TelemetryManager.start_telemetry(polling_interval, mmio_only)

        # Wait for health check
        print("Waiting for telemetry server to become healthy...")
        for i in range(20):
            time.sleep(0.5)
            if TelemetryManager.check_telemetry_health():
                # Additional validation: fetch metrics to confirm polling
                metrics = TelemetryManager.fetch_metrics()
                if metrics:
                    print(f"Telemetry server healthy, stabilizing for {stabilization_sec}s...")
                    time.sleep(stabilization_sec)
                    return proc

        # Failed to start
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("Telemetry server failed to start properly")

    @staticmethod
    def stop_telemetry(proc: subprocess.Popen, timeout: float = 10.0):
        """Stop telemetry server gracefully."""
        if proc is None:
            return

        try:
            print("Stopping telemetry server...")
            proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Telemetry did not stop gracefully, killing...")
            proc.kill()
            proc.wait()


def adaptive_warmup(test_func: Callable, target_cv: float = 0.05, min_iters: int = 10, max_iters: int = 50) -> int:
    """
    Run adaptive warmup until coefficient of variation stabilizes.

    Args:
        test_func: Function that returns a single timing measurement
        target_cv: Target coefficient of variation (default 5%)
        min_iters: Minimum warmup iterations
        max_iters: Maximum warmup iterations

    Returns:
        Number of iterations performed
    """
    recent_times = []

    for i in range(max_iters):
        try:
            t = test_func()
            recent_times.append(t)

            # Check CV after minimum iterations
            if len(recent_times) >= min_iters:
                # Use last min_iters samples
                recent_window = recent_times[-min_iters:]
                if len(recent_window) > 1:
                    cv = stdev(recent_window) / mean(recent_window)
                    if cv < target_cv:
                        print(f"Warmup converged after {i + 1} iterations (CV={cv:.4f})")
                        return i + 1
        except Exception as e:
            print(f"Warning: Warmup iteration {i} failed: {e}")
            continue

    print(f"Warmup completed max iterations ({max_iters})")
    return max_iters


def run_benchmark_with_warmup(
    test_func: Callable, n_samples: int = 100, warmup_iters: int = 20, adaptive: bool = True
) -> List[float]:
    """
    Run benchmark with warmup phase.

    Args:
        test_func: Function that returns a single timing measurement
        n_samples: Number of samples to collect
        warmup_iters: Warmup iterations (if not adaptive)
        adaptive: Use adaptive warmup

    Returns:
        List of timing samples
    """
    # Warmup phase
    if adaptive:
        adaptive_warmup(test_func, max_iters=warmup_iters)
    else:
        print(f"Running {warmup_iters} warmup iterations...")
        for _ in range(warmup_iters):
            try:
                test_func()
            except Exception as e:
                print(f"Warning: Warmup iteration failed: {e}")

    # Measurement phase
    print(f"Collecting {n_samples} samples...")
    samples = []
    for i in range(n_samples):
        try:
            t = test_func()
            samples.append(t)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{n_samples} samples")
        except Exception as e:
            print(f"Warning: Sample {i} failed: {e}")
            continue

    return samples


def interleaved_baseline_measurement(
    baseline_func: Callable, telemetry_func: Callable, n_samples: int = 100
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run benchmark with interleaved baseline measurements.

    Measures baseline before and after telemetry test to detect drift.

    Args:
        baseline_func: Function that runs baseline test
        telemetry_func: Function that runs telemetry test
        n_samples: Number of samples per measurement

    Returns:
        Tuple of (baseline_result, telemetry_result)
    """
    print("Running baseline measurement (before)...")
    baseline_before_samples = run_benchmark_with_warmup(baseline_func, n_samples)

    print("Running telemetry measurement...")
    telemetry_samples = run_benchmark_with_warmup(telemetry_func, n_samples)

    print("Running baseline measurement (after)...")
    baseline_after_samples = run_benchmark_with_warmup(baseline_func, n_samples)

    # Combine baseline measurements
    combined_baseline_samples = baseline_before_samples + baseline_after_samples

    # Check for drift
    baseline_before_mean = mean(baseline_before_samples) if baseline_before_samples else 0
    baseline_after_mean = mean(baseline_after_samples) if baseline_after_samples else 0
    if baseline_before_mean > 0:
        drift = abs(baseline_after_mean - baseline_before_mean) / baseline_before_mean
        if drift > 0.10:  # >10% drift
            print(f"WARNING: Baseline drift detected: {drift*100:.1f}%")

    # Create results
    baseline_result = BenchmarkResult(
        operation="baseline",
        config={},
        samples=combined_baseline_samples,
        mean_time=0,
        std_time=0,
        median_time=0,
        min_time=0,
        max_time=0,
        cv=0,
        errors=[],
        metadata={"before_mean": baseline_before_mean, "after_mean": baseline_after_mean},
    )

    telemetry_result = BenchmarkResult(
        operation="telemetry",
        config={},
        samples=telemetry_samples,
        mean_time=0,
        std_time=0,
        median_time=0,
        min_time=0,
        max_time=0,
        cv=0,
        errors=[],
        metadata={},
    )

    return baseline_result, telemetry_result


# Statistical Analysis Functions


def analyze_monotonicity(frequencies_hz: List[float], impacts: List[float]) -> Dict[str, Any]:
    """
    Test for monotonic trend using Kendall tau correlation.

    Args:
        frequencies_hz: Polling frequencies in Hz
        impacts: Performance impacts (as fractions, e.g., 0.05 = 5%)

    Returns:
        Dictionary with tau, p_value, and monotonic flag
    """
    if len(frequencies_hz) < 3:
        return {"error": "Insufficient data for monotonicity test"}

    tau, p_value = kendalltau(frequencies_hz, impacts)

    return {
        "tau": float(tau),
        "p_value": float(p_value),
        "monotonic": p_value < 0.01 and abs(tau) > 0.3,
        "direction": "increasing" if tau > 0 else "decreasing",
    }


def apply_multiple_comparison_correction(
    p_values: List[float], method: str = "holm", alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of uncorrected p-values
        method: Correction method ('holm', 'bonferroni', 'fdr_bh')
        alpha: Significance level

    Returns:
        Dictionary with corrected p-values and rejection decisions
    """
    if not p_values:
        return {"error": "No p-values provided"}

    if not HAS_STATSMODELS:
        # Fallback to simple Bonferroni correction if statsmodels not available
        bonferroni_alpha = alpha / len(p_values)
        return {
            "method": "bonferroni (fallback)",
            "alpha": alpha,
            "n_tests": len(p_values),
            "n_significant_uncorrected": sum(1 for p in p_values if p < alpha),
            "n_significant_corrected": sum(1 for p in p_values if p < bonferroni_alpha),
            "p_corrected": [min(p * len(p_values), 1.0) for p in p_values],
            "reject": [p < bonferroni_alpha for p in p_values],
            "alpha_corrected": bonferroni_alpha,
            "warning": "statsmodels not available, using simple Bonferroni correction",
        }

    reject, p_corrected, alpha_corrected, _ = multipletests(p_values, alpha=alpha, method=method)

    return {
        "method": method,
        "alpha": alpha,
        "n_tests": len(p_values),
        "n_significant_uncorrected": sum(1 for p in p_values if p < alpha),
        "n_significant_corrected": int(sum(reject)),
        "p_corrected": [float(p) for p in p_corrected],
        "reject": [bool(r) for r in reject],
        "alpha_corrected": float(alpha_corrected),
    }


def mann_whitney_u_test(baseline_samples: List[float], telemetry_samples: List[float]) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        baseline_samples: Baseline timing samples
        telemetry_samples: Telemetry timing samples

    Returns:
        Dictionary with test statistics
    """
    if len(baseline_samples) < 3 or len(telemetry_samples) < 3:
        return {"error": "Insufficient samples"}

    statistic, p_value = mannwhitneyu(baseline_samples, telemetry_samples, alternative="two-sided")

    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(baseline_samples), len(telemetry_samples)
    effect_size = 1 - (2 * statistic) / (n1 * n2)

    return {
        "test": "Mann-Whitney U",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def cohens_d(baseline_samples: List[float], telemetry_samples: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        baseline_samples: Baseline timing samples
        telemetry_samples: Telemetry timing samples

    Returns:
        Cohen's d value
    """
    if len(baseline_samples) < 2 or len(telemetry_samples) < 2:
        return 0.0

    mean1 = mean(baseline_samples)
    mean2 = mean(telemetry_samples)
    std1 = stdev(baseline_samples)
    std2 = stdev(telemetry_samples)

    # Pooled standard deviation
    n1, n2 = len(baseline_samples), len(telemetry_samples)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean2 - mean1) / pooled_std


def check_normality(samples: List[float]) -> Dict[str, Any]:
    """
    Check if samples follow normal distribution using Shapiro-Wilk test.

    Args:
        samples: Timing samples

    Returns:
        Dictionary with test results
    """
    if len(samples) < 3:
        return {"error": "Insufficient samples"}

    statistic, p_value = shapiro(samples)

    return {
        "test": "Shapiro-Wilk",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_normal": p_value > 0.05,
    }


def detect_outliers_iqr(samples: List[float], k: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using IQR method.

    Args:
        samples: Timing samples
        k: IQR multiplier (1.5 for standard, 3.0 for extreme outliers)

    Returns:
        Dictionary with outlier information
    """
    if len(samples) < 4:
        return {"error": "Insufficient samples"}

    q1 = np.percentile(samples, 25)
    q3 = np.percentile(samples, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    outliers = [x for x in samples if x < lower_bound or x > upper_bound]

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "n_outliers": len(outliers),
        "outlier_percentage": 100 * len(outliers) / len(samples),
        "outliers": [float(x) for x in outliers],
    }


def parse_frequency_to_hz(freq_str: str) -> float:
    """
    Parse frequency string to Hz.

    Args:
        freq_str: Frequency string (e.g., "100ms", "1s", "500us")

    Returns:
        Frequency in Hz
    """
    freq_str = freq_str.lower()

    if freq_str.endswith("us"):
        interval_sec = float(freq_str[:-2]) / 1e6
    elif freq_str.endswith("ms"):
        interval_sec = float(freq_str[:-2]) / 1e3
    elif freq_str.endswith("s"):
        interval_sec = float(freq_str[:-1])
    else:
        raise ValueError(f"Unknown frequency format: {freq_str}")

    return 1.0 / interval_sec


def calculate_impact_percentage(baseline_mean: float, telemetry_mean: float) -> float:
    """
    Calculate performance impact as percentage.

    Args:
        baseline_mean: Baseline mean time
        telemetry_mean: Telemetry mean time

    Returns:
        Impact as percentage (e.g., 5.2 for 5.2% slowdown)
    """
    if baseline_mean == 0:
        return 0.0

    return 100 * (telemetry_mean - baseline_mean) / baseline_mean


def save_results_json(results: Any, filepath: str):
    """Save results to JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")


def load_results_json(filepath: str) -> Any:
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Telemetry Benchmark Utilities Module")
    print("=====================================")
    print("This module provides shared utilities for telemetry benchmarking.")
    print("\nAvailable classes:")
    print("  - DeviceManager: Device state and cleanup")
    print("  - TelemetryManager: Telemetry server lifecycle")
    print("\nAvailable functions:")
    print("  - run_with_clean_state(): Run test with clean device state")
    print("  - safe_run_with_auto_reset(): Run test with auto-reset on failure")
    print("  - adaptive_warmup(): Adaptive warmup until CV stabilizes")
    print("  - run_benchmark_with_warmup(): Run benchmark with warmup")
    print("  - interleaved_baseline_measurement(): Baseline bracketing")
    print("  - analyze_monotonicity(): Kendall tau monotonicity test")
    print("  - apply_multiple_comparison_correction(): Bonferroni-Holm correction")
    print("  - mann_whitney_u_test(): Non-parametric significance test")
    print("  - cohens_d(): Effect size calculation")
    print("  - check_normality(): Shapiro-Wilk normality test")
    print("  - detect_outliers_iqr(): IQR outlier detection")
