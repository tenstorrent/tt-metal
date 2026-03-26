# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pandas as pd
import pytest
import models
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


def _run_device_profiler_op_support_count(*args, **kwargs):
    if "op_support_count" not in kwargs:
        kwargs["op_support_count"] = 10000
    return run_device_profiler(*args, **kwargs)


models.perf.device_perf_utils.run_device_profiler = _run_device_profiler_op_support_count


def _prepare_tt_perf_report_csv(subdir: str) -> str:
    source_csv = Path(get_latest_ops_log_filename(subdir))
    df = pd.read_csv(source_csv)
    dtype_cols = [column for column in df.columns if column.endswith("_DATATYPE")]
    if not dtype_cols:
        logger.info(f"No datatype columns found in ops CSV, using original file for report: {source_csv}")
        return str(source_csv)
    float32_mask = df[dtype_cols].eq("FLOAT32").any(axis=1)
    float32_rows = int(float32_mask.sum())
    if float32_rows == 0:
        logger.info(f"No FLOAT32 rows detected, using original file for report: {source_csv}")
        return str(source_csv)
    sanitized_csv = source_csv.with_name(f"{source_csv.stem}_tt_perf_report.csv")
    df.loc[~float32_mask].to_csv(sanitized_csv, index=False)
    logger.info(
        f"Generated tt-perf-report compatible CSV: {sanitized_csv} "
        f"(removed {float32_rows} rows with FLOAT32 datatypes)."
    )
    logger.info(f"Run report with: tt-perf-report {sanitized_csv}")
    return str(sanitized_csv)


@pytest.mark.parametrize(
    "batch_size, model_name, expected_perf",
    [
        (1, "ttnn_lingbot_va", float(os.environ.get("LINGBOT_VA_EXPECTED_PERF", "1.5"))),
    ],
)
@pytest.mark.timeout(0)  # Tracy + nested pytest can exceed global pytest-timeout (e.g. 300s)
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
