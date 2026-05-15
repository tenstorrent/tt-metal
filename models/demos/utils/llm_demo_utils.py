# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import warnings

from loguru import logger

from models.demos.utils.model_targets import resolve_perf_targets
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler, perf_target_check

TOLERANCE_FAMILY_ALIASES = {
    "decode_t/s": "decode_tolerance",
    "decode_t/s/u": "decode_tolerance",
    "prefill_time_to_token": "prefill_tolerance",
    "prefill_time_to_first_token": "prefill_tolerance",
}

METRIC_NAME_ALIASES = {
    "prefill_time_to_token": "prefill_time_to_first_token",
}


class PerfRegressionWarning(UserWarning):
    """Warning emitted when measured perf drifts from configured thresholds."""


def _normalize_metric_aliases(metrics: dict, source_name: str) -> dict:
    """Map legacy metric aliases to canonical names, rejecting conflicting duplicates."""
    normalized = {}
    source_keys = {}
    for key, value in metrics.items():
        canonical_key = METRIC_NAME_ALIASES.get(key, key)
        if canonical_key in normalized and source_keys[canonical_key] != key:
            raise ValueError(
                f"{source_name} contains both legacy and canonical forms for '{canonical_key}': "
                f"use only one of '{key}' or '{canonical_key}'"
            )
        normalized[canonical_key] = value
        source_keys[canonical_key] = key
    return normalized


def create_benchmark_data(profiler: BenchmarkProfiler, measurements: dict, N_warmup_iter: dict, targets: dict):
    """
    Create a benchmark data object and populate the object with the given measurements.

    Pre-requisites:
    - The measurements dictionary should contain the following keys: "prefill_t/s", "prefill_time_to_first_token", "decode_t/s", "decode_t/s/u"
    - The profiler object should contain the start and end times for the steps "inference_prefill", "inference_decode"

    Optional:
    - "compile_prefill", "compile_decode" - only needed if compilation is tracked separately from warmup
    - "prefill_decode_t/s/u" - if measuring combined prefill+decode perf
    - "token_verification" - if doing token verification

    - The targets dictionary should contain the following keys: "prefill_t/s", "decode_t/s", "decode_t/s/u"
    - The N_warmup_iter dictionary should contain the following keys: "inference_prefill", "inference_decode"
    """

    assert all(
        key in measurements
        for key in [
            "prefill_t/s",
            "prefill_time_to_first_token",
            "decode_t/s",
            "decode_t/s/u",
        ]
    )

    benchmark_data = BenchmarkData()

    # Add optional compile time measurements (only if compilation is tracked separately from warmup)
    if "compile_prefill" in measurements:
        benchmark_data.add_measurement(profiler, 0, "compile_prefill", "time(s)", measurements["compile_prefill"])
    if "compile_decode" in measurements:
        benchmark_data.add_measurement(profiler, 0, "compile_decode", "time(s)", measurements["compile_decode"])
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_prefill",
        "tokens/s",
        measurements["prefill_t/s"],
        step_warm_up_num_iterations=(
            N_warmup_iter["inference_prefill"] if "inference_prefill" in N_warmup_iter else None
        ),
        target=targets["prefill_t/s"] if "prefill_t/s" in targets else None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_prefill",
        "time_to_token",
        measurements["prefill_time_to_first_token"],
        step_warm_up_num_iterations=(
            N_warmup_iter["inference_prefill"] if "inference_prefill" in N_warmup_iter else None
        ),
        target=None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_decode",
        "tokens/s",
        measurements["decode_t/s"],
        step_warm_up_num_iterations=N_warmup_iter["inference_decode"] if "inference_decode" in N_warmup_iter else None,
        target=targets["decode_t/s"] if "decode_t/s" in targets else None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_decode",
        "tokens/s/user",
        measurements["decode_t/s/u"],
        step_warm_up_num_iterations=N_warmup_iter["inference_decode"] if "inference_decode" in N_warmup_iter else None,
        target=targets["decode_t/s/u"] if "decode_t/s/u" in targets else None,
    )

    # Add optional measurement data
    if "prefill_decode_t/s/u" in measurements:
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_prefill_decode",
            "tokens/s/user",
            measurements["prefill_decode_t/s/u"],
            step_warm_up_num_iterations=None,
            target=None,
        )
    if "token_verification" in measurements:
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "token_verification",
            measurements["token_verification"],
        )

    return benchmark_data


def check_tokens_match(generated_text: dict, expected_greedy_output_path: str):
    with open(expected_greedy_output_path, "r") as f:
        expected_output = json.load(f)
    return generated_text == expected_output, expected_output


def verify_perf(
    measurements: dict,
    expected_perf_metrics: dict = None,
    high_tol_percentage=1.15,  # 15% tolerance (approx +-5% CI variance + 5% real increase)
    expected_measurements: dict = None,
    lower_is_better_metrics: set = None,
    model_name: str = None,
    sku: str = None,
    batch_size: int = None,
    seq_len: int = None,
):
    """
    Verify the performance metrics against the expected values.
    The metrics that must be provided are specified in expected_measurements below.
    Args:
        measurements: dict of measured performance values
        expected_perf_metrics: dict of expected performance values. If omitted, the values are
            resolved from centralized YAML using model_name/sku[/batch_size/seq_len].
        high_tol_percentage: tolerance percentage (e.g., 1.15 means 15% tolerance)
        expected_measurements: dict specifying which measurements are required
        lower_is_better_metrics: set of metric names where lower values are better (e.g., TTFT)
    """
    targets_from_centralized_yaml = expected_perf_metrics is None
    if expected_perf_metrics is None:
        if not model_name or not sku:
            raise ValueError("model_name and sku are required when expected_perf_metrics is not provided")
        expected_perf_metrics = resolve_perf_targets(
            model_name=model_name,
            sku=sku,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        if expected_perf_metrics is None:
            raise ValueError(
                f"No centralized perf targets found for model={model_name}, sku={sku}, "
                f"batch_size={batch_size}, seq_len={seq_len}"
            )
    else:
        expected_perf_metrics = _normalize_metric_aliases(expected_perf_metrics, "expected_perf_metrics")

    measurements = _normalize_metric_aliases(measurements, "measurements")

    expected_measurements_default = {
        "compile_prefill": False,
        "compile_decode": False,
        "prefill_time_to_first_token": False,
        "prefill_decode_t/s/u": False,
        "prefill_t/s": True,
        "decode_t/s": True,
        "decode_t/s/u": True,
    }
    expected_measurements = expected_measurements_default if expected_measurements is None else expected_measurements
    expected_measurements = _normalize_metric_aliases(expected_measurements, "expected_measurements")

    def normalized_expected_value(metric_name: str, expected_value: float) -> float:
        """
        Normalize expected metric units before comparing with raw measurements.

        TTFT measurements are emitted in seconds; centralized YAML stores TTFT in
        milliseconds. Explicit expected_perf_metrics callers are expected to pass
        TTFT targets in seconds.
        """
        if metric_name != "prefill_time_to_first_token":
            return expected_value
        expected_ttft = float(expected_value)
        if targets_from_centralized_yaml:
            return expected_ttft / 1000.0
        return expected_ttft

    def metric_tolerance(metric_name: str) -> float:
        explicit_keys = [
            f"{metric_name}_tolerance",
            f"{metric_name.replace('/', '_')}_tolerance",
        ]
        family_alias = TOLERANCE_FAMILY_ALIASES.get(metric_name)
        if family_alias:
            explicit_keys.append(family_alias)
        for key in explicit_keys:
            tolerance = expected_perf_metrics.get(key)
            if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
                return float(tolerance)
        generic_tolerance = expected_perf_metrics.get("tolerance")
        if isinstance(generic_tolerance, (int, float)) and not isinstance(generic_tolerance, bool):
            return float(generic_tolerance)
        return float(high_tol_percentage)

    # Default metrics where lower is better
    lower_is_better_metrics_default = {
        "prefill_time_to_first_token",
        "compile_prefill",
        "compile_decode",
    }
    if lower_is_better_metrics:
        lower_is_better_metrics = {
            METRIC_NAME_ALIASES.get(metric_name, metric_name) for metric_name in lower_is_better_metrics
        }
    lower_is_better_metrics = (
        lower_is_better_metrics_default.union(lower_is_better_metrics)
        if lower_is_better_metrics
        else lower_is_better_metrics_default
    )

    does_pass = True
    for key in expected_measurements:
        if not expected_measurements[key]:
            continue
        if not perf_target_check(
            key in measurements and key in expected_perf_metrics and expected_perf_metrics[key] is not None,
            f"Metric {key} not found in measurements or expected_perf_metrics",
        ):
            does_pass = False
            continue

        if key in lower_is_better_metrics:
            tolerance = metric_tolerance(key)
            expected_value = normalized_expected_value(key, expected_perf_metrics[key])
            # For metrics where lower is better (e.g., TTFT)
            if measurements[key] > expected_value:  # Higher than expected is bad
                does_pass = False
                logger.warning(f"{key} ({measurements[key]}) is higher than expected {expected_value}")
            elif measurements[key] < expected_value * (2 - tolerance):  # Much lower than expected
                does_pass = False
                logger.warning(
                    f"{key} ({measurements[key]}) is much lower than expected {expected_value}. Please update the expected perf."
                )
        else:
            tolerance = metric_tolerance(key)
            expected_value = normalized_expected_value(key, expected_perf_metrics[key])
            # For metrics where higher is better (e.g., throughput)
            if measurements[key] < expected_value:  # Lower than expected is bad
                does_pass = False
                logger.warning(f"{key} ({measurements[key]}) is lower than expected {expected_value}")
            elif measurements[key] > expected_value * tolerance:  # Much higher than expected
                does_pass = False
                logger.warning(
                    f"{key} ({measurements[key]}) is much higher than expected {expected_value}. Please update the expected perf."
                )

    if does_pass:
        logger.info("Perf Check Passed!")
    else:
        logger.warning("Perf Check Failed!")
        warnings.warn(
            f"Perf drift detected against expected metrics {expected_perf_metrics}. "
            "See warnings above for metric-level details.",
            PerfRegressionWarning,
            stacklevel=2,
        )
        perf_target_check(
            does_pass,
            "Performance regression detected. Failing fast to avoid silently accepting "
            "degraded model performance while centralized targets rollout is in progress.",
        )
