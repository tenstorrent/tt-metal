# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler, BenchmarkData


def create_benchmark_data(profiler: BenchmarkProfiler, measurements: dict, N_warmup_iter: dict, targets: dict):
    """
    Create a benchmark data object and populate the object with the given measurements.

    Pre-requisites:
    - The measurements dictionary should contain the following keys: "compile_prefill", "compile_decode", "prefill_t/s", "prefill_time_to_token", "decode_t/s", "decode_t/s/u"
    - The profiler object should contain the start and end times for the steps "compile_prefill", "compile_decode", "inference_prefill", "inference_decode"

    Optional (should be provided if measuring perf, not required for token generation):
    - The measurements dictionary should contain the following keys: "prefill_decode_t/s/u"
    - The targets dictionary should contain the following keys: "prefill_t/s", "decode_t/s", "decode_t/s/u"
    - The N_warmup_iter dictionary should contain the following keys: "inference_prefill", "inference_decode"

    Optional (should be provided if doing token verification, not required for perf):
    - The measurements dictionary should contain the key "token_verification"
    """

    assert all(
        key in measurements
        for key in [
            "compile_prefill",
            "compile_decode",
            "prefill_t/s",
            "prefill_time_to_token",
            "decode_t/s",
            "decode_t/s/u",
        ]
    )

    benchmark_data = BenchmarkData()

    # Add required measurement data
    benchmark_data.add_measurement(profiler, 0, "compile_prefill", "time(s)", measurements["compile_prefill"])
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


def verify_perf(measurements: dict, expected_perf_metrics: dict):
    """
    Verify the performance metrics against the expected values.
    The metrics that must be provided are specified in expected_measurements below.
    """

    expected_measurements = {
        "compile_prefill": False,
        "compile_decode": False,
        "prefill_time_to_token": False,
        "prefill_decode_t/s/u": False,
        "prefill_t/s": True,
        "decode_t/s": True,
        "decode_t/s/u": True,
    }

    does_pass = True
    for key in expected_measurements:
        if not expected_measurements[key]:
            continue
        assert (
            key in measurements and key in expected_perf_metrics
        ), f"Metric {key} not found in measurements or expected_perf_metrics"
        if measurements[key] < expected_perf_metrics[key]:
            does_pass = False
            logger.warning(f"{key} ({measurements[key]}) is lower than expected {expected_perf_metrics[key]}")

    if does_pass:
        logger.info("Perf Check Passed!")
    else:
        logger.warning("Perf Check Failed!")
        assert (
            does_pass
        ), f"Prefill or decode perf is lower than {expected_perf_metrics}. See earlier warnings for more details."
