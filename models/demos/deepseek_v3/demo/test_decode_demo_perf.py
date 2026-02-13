# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3 decode demo device performance test.

Runs the profile_decode test case under the device profiler, filters and
processes the resulting CSV to compute 2-Layer Model E2E Time, then asserts
against known-good performance thresholds.
"""

from pathlib import Path

import pytest
from loguru import logger
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

from models.demos.deepseek_v3.utils.device_perf_utils import compute_e2e_time, filter_profile_csv, process_profile_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_within_margin(metric_name: str, measured: float, expected: float, margin: float):
    """Assert that *measured* falls within *expected* ± *margin* (fraction)."""
    lower = expected * (1 - margin)
    upper = expected * (1 + margin)
    logger.info(f"{metric_name}: measured={measured:.2f} us, expected=[{lower:.2f}, {upper:.2f}] us")
    assert lower <= measured <= upper, (
        f"{metric_name} {measured:.2f} us is outside the expected range "
        f"[{lower:.2f}, {upper:.2f}] us (expected {expected:.2f} us ± {margin*100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "expected_kernel_duration_us, expected_op_to_op_latency_us, expected_e2e_time_us, margin",
    [
        pytest.param(10386.34, 191.52, 10577.85, 0.03, id="decode_e2e_perf"),
    ],
)
def test_decode_demo_perf(expected_kernel_duration_us, expected_op_to_op_latency_us, expected_e2e_time_us, margin):
    """
    End-to-end device-performance test for the DeepSeek V3 2-layer decode demo.
    1st layer is dense decoder block and 2nd layer is MoE Decoder Block.

    Steps
    -----
    1. Run the ``profile_decode`` test case under the device profiler.
    2. Filter the raw ops CSV.
    3. Compute per-op statistics and derive 2-Layer Model E2E Time.
    4. Assert against thresholds.
    """
    subdir = "deepseek_v3_decode_perf"
    command = "pytest models/demos/deepseek_v3/demo/test_demo.py -k profile_decode"

    # ------------------------------------------------------------------
    # Step 1 – Run the device profiler
    # ------------------------------------------------------------------
    logger.info("Step 1: Running device profiler …")
    clear_profiler_runtime_artifacts()
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"], op_support_count=5000)

    # ------------------------------------------------------------------
    # Step 2 – Locate the raw CSV produced by the profiler
    # ------------------------------------------------------------------
    raw_csv_path = Path(get_latest_ops_log_filename(subdir))
    logger.info(f"Step 2: Raw profiler CSV → {raw_csv_path}")
    assert raw_csv_path.exists(), f"Raw CSV not found: {raw_csv_path}"

    # ------------------------------------------------------------------
    # Step 3 – Filter the CSV
    # ------------------------------------------------------------------
    logger.info("Step 3: Filtering CSV …")
    filtered_csv_path = raw_csv_path.parent / "decode_demo_profile_filtered.csv"
    filter_profile_csv(raw_csv_path, filtered_csv_path)

    # ------------------------------------------------------------------
    # Step 4 – Process stats & write merged CSV
    # ------------------------------------------------------------------
    logger.info("Step 4: Processing stats …")
    merged_csv_path = raw_csv_path.parent / "decode_demo_profile_merged.csv"
    results = process_profile_stats(filtered_csv_path, merged_csv_path)

    # ------------------------------------------------------------------
    # Step 5 – Compute & log 2-Layer Model E2E Time
    # ------------------------------------------------------------------
    e2e = compute_e2e_time(results)

    total_kernel_us = e2e["total_kernel_duration_us"]
    total_latency_us = e2e["total_op_to_op_latency_us"]
    e2e_time_us = e2e["e2e_time_us"]

    logger.info(
        f"\n{'=' * 50}"
        f"\n2-Layer Model E2E Time"
        f"\n{'=' * 50}"
        f"\n  Total Kernel Duration : {total_kernel_us:.2f} us ({total_kernel_us / 1000:.3f} ms)"
        f"\n  Total Op-to-Op Latency: {total_latency_us:.2f} us ({total_latency_us / 1000:.3f} ms)"
        f"\n  E2E Time              : {e2e_time_us:.2f} us ({e2e_time_us / 1000:.3f} ms)"
        f"\n{'=' * 50}"
    )

    # ------------------------------------------------------------------
    # Step 6 – Assert against expected thresholds
    # ------------------------------------------------------------------
    logger.info("Step 6: Asserting performance thresholds …")

    _assert_within_margin("E2E Time", e2e_time_us, expected_e2e_time_us, margin)
    _assert_within_margin("Total Kernel Duration", total_kernel_us, expected_kernel_duration_us, margin)
    _assert_within_margin("Total Op-to-Op Latency", total_latency_us, expected_op_to_op_latency_us, margin)

    logger.info("All performance assertions passed!")
