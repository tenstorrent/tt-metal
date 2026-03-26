# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device perf: Tracy profile of ``test_lingbot_va`` PCC smoke (nested pytest invocation)."""

import os
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

import models
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


def _run_device_profiler_op_support_count(*args, **kwargs):
    # Default cap is low; Lingbot-VA graphs exceed it without this override.
    kwargs.setdefault("op_support_count", 10000)
    return run_device_profiler(*args, **kwargs)


models.perf.device_perf_utils.run_device_profiler = _run_device_profiler_op_support_count


def _prepare_tt_perf_report_csv(subdir: str) -> str:
    source_csv = Path(get_latest_ops_log_filename(subdir))
    df = pd.read_csv(source_csv)
    dtype_cols = [column for column in df.columns if column.endswith("_DATATYPE")]
    if not dtype_cols:
        logger.info("No datatype columns in ops CSV, using {}", source_csv)
        return str(source_csv)
    float32_mask = df[dtype_cols].eq("FLOAT32").any(axis=1)
    float32_rows = int(float32_mask.sum())
    if float32_rows == 0:
        logger.info("No FLOAT32 rows, using {}", source_csv)
        return str(source_csv)
    sanitized_csv = source_csv.with_name(f"{source_csv.stem}_tt_perf_report.csv")
    df.loc[~float32_mask].to_csv(sanitized_csv, index=False)
    logger.info(
        "tt-perf-report CSV: {} (dropped {} FLOAT32 rows)",
        sanitized_csv,
        float32_rows,
    )
    return str(sanitized_csv)


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "ttnn_lingbot_va", float(os.environ.get("LINGBOT_VA_EXPECTED_PERF", "1.5"))),
    ],
)
@pytest.mark.timeout(0)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_lingbot_va(batch_size, model_name, expected_perf):
    subdir = model_name
    num_iterations = 1
    margin = float(os.environ.get("LINGBOT_VA_PERF_MARGIN", "0.5"))

    command = "pytest models/experimental/lingbot_va/tests/pcc/test_lingbot_va.py::test_lingbot_va"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_functional_{model_name}_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

    _prepare_tt_perf_report_csv(subdir)
