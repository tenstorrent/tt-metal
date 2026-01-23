# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf


@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_maskformer_swin_b():
    batch_size = 1
    subdir = "maskformer_swin_base_coco"
    num_iterations = 1
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    # Run the PCC test under the device profiler to generate Ops_Perf.csv / ops_perf_results_*.csv.
    command = "pytest models/experimental/maskformer_swin/tests/pcc/test_maskformer_swin.py::test_maskformer_swin_b_pcc"
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)

    # No fixed perf target yet (bring-up baseline); report only.
    expected_results = {}
    logger.info(f"{post_processed_results}")

    prep_device_perf_report(
        model_name=f"ttnn_maskformer_swin_b_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="baseline",
    )
