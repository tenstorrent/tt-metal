# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import warnings

from loguru import logger

from models.demos.utils.model_targets import (
    DEFAULT_PERF_TOLERANCE,
    is_tolerance_key,
    resolve_accuracy_targets,
    resolve_metric_tolerance,
    resolve_perf_targets,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

PREFILL_TIME_TO_TOKEN_KEY = "prefill_time_to_token"
PREFILL_TIME_TO_FIRST_TOKEN_KEY = "prefill_time_to_first_token"


class PerfRegressionWarning(UserWarning):
    """Warning emitted when measured perf drifts from configured thresholds."""


class AccuracyRegressionWarning(UserWarning):
    """Warning emitted when measured accuracy drifts from configured thresholds."""


def _normalize_ttft_metrics(metrics: dict, source_name: str, *, ttft_in_ms: bool) -> dict:
    """
    Collapse the two time-to-first-token aliases to a single seconds value.

    `prefill_time_to_first_token` and `prefill_time_to_token` both map to the same
    measured quantity (TTFT); `prefill_time_to_first_token` takes precedence when
    both are present. The resulting seconds value is written back under both keys.

    Units are explicit, not inferred from which key is present, so a seconds value
    is never silently divided by 1000:
      - `ttft_in_ms=True`  for centralized targets, where `prefill_time_to_first_token`
        is stored in milliseconds (converted to seconds here).
      - `ttft_in_ms=False` for measured values, which are already in seconds.
    `prefill_time_to_token` is always treated as seconds regardless of the flag.
    """
    normalized_metrics = dict(metrics)
    has_ttft_first = metrics.get(PREFILL_TIME_TO_FIRST_TOKEN_KEY) is not None
    has_ttft_to = metrics.get(PREFILL_TIME_TO_TOKEN_KEY) is not None

    if has_ttft_first and has_ttft_to:
        logger.warning(
            f"Both {PREFILL_TIME_TO_FIRST_TOKEN_KEY} and {PREFILL_TIME_TO_TOKEN_KEY} are set in {source_name}; "
            f"using {PREFILL_TIME_TO_FIRST_TOKEN_KEY} and ignoring {PREFILL_TIME_TO_TOKEN_KEY}."
        )

    ttft_value_seconds = None
    if has_ttft_first:
        value = float(metrics[PREFILL_TIME_TO_FIRST_TOKEN_KEY])
        ttft_value_seconds = value / 1000.0 if ttft_in_ms else value
    elif has_ttft_to:
        ttft_value_seconds = float(metrics[PREFILL_TIME_TO_TOKEN_KEY])

    if ttft_value_seconds is not None:
        normalized_metrics[PREFILL_TIME_TO_FIRST_TOKEN_KEY] = ttft_value_seconds
        normalized_metrics[PREFILL_TIME_TO_TOKEN_KEY] = ttft_value_seconds

    return normalized_metrics


def create_benchmark_data(profiler: BenchmarkProfiler, measurements: dict, N_warmup_iter: dict, targets: dict):
    """
    Create a benchmark data object and populate the object with the given measurements.

    Pre-requisites:
    - The measurements dictionary should contain the following keys: "prefill_t/s", "prefill_time_to_token", "decode_t/s", "decode_t/s/u"
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
            "prefill_time_to_token",
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
        measurements["prefill_time_to_token"],
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
    expected_measurements: dict = None,
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
        expected_measurements: dict specifying which measurements are required
    """
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
            # Missing coverage is a warning, not a hard failure: it must not abort the
            # test before benchmark data is written. Enforcement of presence is handled
            # by the centralized target validation step in CI.
            logger.warning(
                f"No centralized perf targets found for model={model_name}, sku={sku}, "
                f"batch_size={batch_size}, seq_len={seq_len}; skipping perf check."
            )
            return

    # Measured values are already in seconds; centralized targets store TTFT in milliseconds.
    measurements = _normalize_ttft_metrics(measurements, "measurements", ttft_in_ms=False)
    expected_perf_metrics = _normalize_ttft_metrics(expected_perf_metrics, "expected_perf_metrics", ttft_in_ms=True)
    tolerance_config = {k: v for k, v in expected_perf_metrics.items() if is_tolerance_key(k)}
    expected_perf_metrics = {k: v for k, v in expected_perf_metrics.items() if not is_tolerance_key(k)}

    expected_measurements_default = {
        "compile_prefill": False,
        "compile_decode": False,
        "prefill_time_to_token": False,
        "prefill_time_to_first_token": False,
        "prefill_decode_t/s/u": False,
        "prefill_t/s": True,
        "decode_t/s": True,
        "decode_t/s/u": True,
    }
    expected_measurements = expected_measurements_default if expected_measurements is None else expected_measurements
    expected_measurements = dict(expected_measurements)

    if expected_measurements.get(PREFILL_TIME_TO_FIRST_TOKEN_KEY) and expected_measurements.get(
        PREFILL_TIME_TO_TOKEN_KEY
    ):
        logger.warning(
            f"Both {PREFILL_TIME_TO_FIRST_TOKEN_KEY} and {PREFILL_TIME_TO_TOKEN_KEY} are enabled in expected_measurements; "
            f"using {PREFILL_TIME_TO_FIRST_TOKEN_KEY} and ignoring {PREFILL_TIME_TO_TOKEN_KEY}."
        )
        expected_measurements[PREFILL_TIME_TO_TOKEN_KEY] = False

    # Default metrics where lower is better
    lower_is_better_metrics = {
        "prefill_time_to_token",
        "prefill_time_to_first_token",
        "compile_prefill",
        "compile_decode",
    }

    does_pass = True
    for key in expected_measurements:
        if not expected_measurements[key]:
            continue
        is_key_found = key in measurements and key in expected_perf_metrics and expected_perf_metrics[key] is not None
        if not is_key_found:
            logger.warning(f"Metric {key} not found in measurements or expected_perf_metrics")
            does_pass = False
            continue
        metric_tolerance = resolve_metric_tolerance(
            metric_name=key,
            thresholds=tolerance_config,
            default_tolerance=DEFAULT_PERF_TOLERANCE,
        )
        # Symmetric +/- tolerance band, applied the same way for lower-is-better
        # (e.g. TTFT) and higher-is-better (e.g. throughput) metrics: a value
        # within [expected*(1-tol), expected*(1+tol)] passes. Outside the band,
        # the "worse" side is a regression and the "better" side means the target
        # is stale; lower_is_better only decides which side is which.
        expected = expected_perf_metrics[key]
        measured = measurements[key]
        lower_bound = expected * (1 - metric_tolerance)
        upper_bound = expected * (1 + metric_tolerance)
        if lower_bound <= measured <= upper_bound:
            continue
        does_pass = False
        lower_is_better = key in lower_is_better_metrics
        if measured > upper_bound:
            if lower_is_better:
                logger.warning(f"{key} ({measured}) is higher than expected {expected} (tolerance {metric_tolerance})")
            else:
                logger.warning(
                    f"{key} ({measured}) is much higher than expected {expected}. Please update the expected perf."
                )
        else:  # measured < lower_bound
            if lower_is_better:
                logger.warning(
                    f"{key} ({measured}) is much lower than expected {expected}. Please update the expected perf."
                )
            else:
                logger.warning(f"{key} ({measured}) is lower than expected {expected} (tolerance {metric_tolerance})")

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


def verify_accuracy(
    measurements: dict,
    expected_accuracy_metrics: dict = None,
    model_name: str = None,
    sku: str = None,
    batch_size: int = None,
    seq_len: int = None,
):
    """
    Verify top-k token accuracy against centralized targets in warning-only mode.
    """
    if expected_accuracy_metrics is None:
        if not model_name or not sku:
            raise ValueError("model_name and sku are required when expected_accuracy_metrics is not provided")
        expected_accuracy_metrics = resolve_accuracy_targets(
            model_name=model_name,
            sku=sku,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        if expected_accuracy_metrics is None:
            # Missing coverage is a warning, not a hard failure: it must not abort the
            # test before benchmark data is written. Enforcement of presence is handled
            # by the centralized target validation step in CI.
            logger.warning(
                f"No centralized accuracy targets found for model={model_name}, sku={sku}, "
                f"batch_size={batch_size}, seq_len={seq_len}; skipping accuracy check."
            )
            return

    measured_top1 = measurements.get("top1_token_accuracy", measurements.get("top1"))
    measured_top5 = measurements.get("top5_token_accuracy", measurements.get("top5"))
    target_top1 = expected_accuracy_metrics.get("top1")
    target_top5 = expected_accuracy_metrics.get("top5")

    does_pass = True
    if target_top1 is not None:
        if measured_top1 is None:
            logger.warning("Metric top1_token_accuracy not found in measurements")
            does_pass = False
        elif float(measured_top1) < float(target_top1):
            logger.warning(f"top1_token_accuracy ({measured_top1}) is lower than expected {target_top1}")
            does_pass = False
    if target_top5 is not None:
        if measured_top5 is None:
            logger.warning("Metric top5_token_accuracy not found in measurements")
            does_pass = False
        elif float(measured_top5) < float(target_top5):
            logger.warning(f"top5_token_accuracy ({measured_top5}) is lower than expected {target_top5}")
            does_pass = False

    if does_pass:
        logger.info("Accuracy Check Passed!")
    else:
        logger.warning("Accuracy Check Failed!")
        warnings.warn(
            f"Accuracy drift detected against expected metrics {expected_accuracy_metrics}. "
            "See warnings above for metric-level details.",
            AccuracyRegressionWarning,
            stacklevel=2,
        )
