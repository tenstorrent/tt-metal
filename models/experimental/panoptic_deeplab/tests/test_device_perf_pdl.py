# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.experimental.panoptic_deeplab.tt.common import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS


@pytest.mark.parametrize(
    "command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations, batch_size, margin",
    [
        (
            "pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py::test_model_panoptic_deeplab -k test_panoptic_deeplab",
            57_841_291,
            PANOPTIC_DEEPLAB,
            PANOPTIC_DEEPLAB,
            1,
            1,
            0.015,
        ),
        (
            "pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py::test_model_panoptic_deeplab -k test_deeplab_v3_plus",
            35_762_345,
            DEEPLAB_V3_PLUS,
            DEEPLAB_V3_PLUS,
            1,
            1,
            0.015,
        ),
    ],
    ids=["test_panoptic_deeplab", "test_deeplab_v3_plus"],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_pdl(
    command, expected_device_perf_cycles_per_iteration, subdir, model_name, num_iterations, batch_size, margin
):
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"
    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_cycles_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
