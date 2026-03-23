# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm Performance Test
Measures device kernel duration and memory bandwidth for single-core RMSNorm.
"""

import pandas as pd
import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

# (width, use_fp32) -> expected max duration in microseconds
SHAPE2TIME = {
    (7168, True): 100,
    (7168, False): 100,
    (1536, True): 30,
    (1536, False): 30,
    (512, True): 20,
    (512, False): 20,
}


@pytest.mark.parametrize(
    "width, use_fp32",
    SHAPE2TIME.keys(),
)
def test_rmsnorm_performance(width, use_fp32):
    # Pytest generates IDs bottom-up from stacked parametrize decorators:
    # use_fp32 (innermost) -> epsilon -> width (outermost)
    command = (
        f"pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_rmsnorm.py"
        f"::test_rmsnorm[{use_fp32}-1e-06-{width}]"
    )
    run_device_profiler(command, "rmsnorm_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("rmsnorm_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].sum() / 1000.0
    logger.warning(f"Total Duration (width={width}, fp32={use_fp32}): {duration_us:.3f} us")

    # Input + gamma + output, each width * sizeof(bfloat16)
    bytes_per_element = 2  # bfloat16
    total_bytes = 3 * width * bytes_per_element
    bandwidth_gb_s = total_bytes / (duration_us * 1000) if duration_us > 0 else 0
    logger.warning(f"Realized Bandwidth: {bandwidth_gb_s:.1f} GB/s")

    expected_us = SHAPE2TIME[(width, use_fp32)]
    assert (
        duration_us < expected_us
    ), f"Performance {duration_us:.3f} us exceeds expected {expected_us} us (width={width}, fp32={use_fp32})"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]):
    filename = get_latest_ops_log_filename(output_logs_subdir)
    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
