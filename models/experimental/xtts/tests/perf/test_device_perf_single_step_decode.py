# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

TILE = 32
TEXT_LEN = 96
MAX_NEW_TOKENS = 240
PROMPT_LEN = 32 + TEXT_LEN
MAX_SEQ = -(-(PROMPT_LEN + MAX_NEW_TOKENS + 1) // TILE) * TILE


def _inner_command() -> str:
    """Build the pytest command that profiles a single decode step."""
    profile_test = (
        "models/experimental/xtts/tests/perf/test_profile_single_step_decode.py::test_profile_single_step_decode"
    )
    return f"pytest --timeout=1800 {profile_test} -sv"


def main() -> int:
    """Run Tracy device profiling for a single GPT decode step and write a report."""
    from loguru import logger
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_samples_per_s, post_process_ops_log, run_device_profiler

    from models.perf.device_perf_utils import prep_device_perf_report

    num_layers = os.environ.get("XTTS_PERF_NUM_LAYERS", "30")
    model_name = f"xtts_gpt_decode1_L{num_layers}_maxseq{MAX_SEQ}"
    subdir = "xtts_gpt_single_step_decode"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    op_support_count = 50000

    batch_size = 1
    duration_cols = [col + " DURATION [ns]" for col in cols]
    samples_cols = [col + " SAMPLES/S" for col in cols]

    clear_profiler_runtime_artifacts()
    run_device_profiler(
        _inner_command(),
        subdir,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=op_support_count,
    )

    raw = post_process_ops_log(subdir, duration_cols, has_signposts=True)
    post_processed_results = {}
    for s_col, d_col in zip(samples_cols, duration_cols):
        ns = raw[d_col]
        post_processed_results[f"AVG {s_col}"] = get_samples_per_s(ns, batch_size)
        post_processed_results[f"MIN {s_col}"] = get_samples_per_s(ns, batch_size)
        post_processed_results[f"MAX {s_col}"] = get_samples_per_s(ns, batch_size)
        post_processed_results[f"AVG {d_col}"] = ns
        post_processed_results[f"MIN {d_col}"] = ns
        post_processed_results[f"MAX {d_col}"] = ns

    logger.info(f"Device perf results for {model_name}:\n{post_processed_results}")

    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments=f"single_decode_prompt{PROMPT_LEN}_maxseq{MAX_SEQ}_tracedwrite",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
