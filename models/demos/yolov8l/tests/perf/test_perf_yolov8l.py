# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


@pytest.mark.parametrize(
    "name_suffix,batch_size,resolution,expected_perf,test_selector,op_support_count",
    [
        [
            "b1_640",
            1,
            640,
            50.0,
            "test_yolov8l and 640 and not dp_batch8 and not test_yolov8l_640 and not test_yolov8l_1280",
            12000,
        ],
        [
            "b1_1280",
            1,
            1280,
            20.0,
            "models/demos/yolov8l/tests/pcc/test_yolov8l.py::test_yolov8l[1280-l1_1280_for_all_res-True]",
            12000,
        ],
        ["b2_640", 2, 640, 50.0, "test_yolov8l_dp_batch2 and n300_1x2 and 640", 12000],
        ["b2_1280", 2, 1280, 14.90, "test_yolov8l_dp_batch2 and n300_1x2 and 1280", 12000],
        ["b4_640", 4, 640, 50.0, "test_yolov8l_dp_batch4 and wh_1x4 and 640", 12000],
        ["b4_1280", 4, 1280, 14.90, "test_yolov8l_dp_batch4 and wh_1x4 and 1280", 12000],
        ["b8_640", 8, 640, 50.0, "test_yolov8l_dp_batch8 and t3k_1x8 and 640", 12000],
        ["b8_1280", 8, 1280, 14.90, "test_yolov8l_dp_batch8 and t3k_1x8 and 1280", 12000],
    ],
    ids=["b1_640", "b1_1280", "b2_640", "b2_1280", "b4_640", "b4_1280", "b8_640", "b8_1280"],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_yolov8l(name_suffix, batch_size, resolution, expected_perf, test_selector, op_support_count):
    subdir = "ttnn_yolov8l"
    num_iterations = 1
    margin = 0.03
    if test_selector.endswith(".py") or "::" in test_selector:
        command = f"pytest {test_selector}"
    else:
        command = "pytest models/demos/yolov8l/tests/pcc/test_yolov8l.py " f'-k "{test_selector}"'
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        op_support_count=op_support_count,
    )
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    avg_kernel_samples_per_s = post_processed_results[inference_time_key]
    kernel_iterations_per_s = avg_kernel_samples_per_s / batch_size
    kernel_latency_per_image_s = 1 / avg_kernel_samples_per_s
    kernel_latency_per_iteration_s = 1 / kernel_iterations_per_s
    logger.info(
        "Kernel throughput summary:\n"
        f"batch_size: {batch_size}\n"
        f"avg kernel samples/s: {avg_kernel_samples_per_s:.6f}\n"
        f"kernel iter/s: {kernel_iterations_per_s:.6f}\n"
        f"kernel latency/iter (sec): {kernel_latency_per_iteration_s:.6f}\n"
        f"kernel latency/image (sec): {kernel_latency_per_image_s:.6f}"
    )
    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_yolov8l_{name_suffix}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=f"resolution={resolution}",
    )
