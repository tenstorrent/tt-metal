# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


def _run_yolov8s_perf(name_suffix, batch_size, expected_perf, test_filter, min_mesh_devices):
    if min_mesh_devices > 1:
        import ttnn

        if not ttnn.using_distributed_env() and ttnn.get_num_devices() < min_mesh_devices:
            pytest.skip(
                f"{name_suffix} needs a {min_mesh_devices}-device mesh (skips single-device N150 and other undersized hosts)"
            )

    subdir = "ttnn_yolov8s"
    num_iterations = 1
    margin = 0.05
    command = f'pytest models/demos/yolov8s/tests/pcc/test_yolov8s.py -k "{test_filter}"'
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8s_{name_suffix}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8s_b1():
    _run_yolov8s_perf("b1_640", 1, 236.95, "test_yolov8s_640", 1)


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8s_b2():
    _run_yolov8s_perf("b2_640", 2, 0, "test_yolov8s_dp_batch2", 2)


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8s_b4():
    _run_yolov8s_perf("b4_640", 4, 0, "test_yolov8s_dp_batch4", 4)


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8s_b8():
    _run_yolov8s_perf("b8_640", 8, 0, "test_yolov8s_dp_batch8", 8)
