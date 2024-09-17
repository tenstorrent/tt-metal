# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import run_for_wormhole_b0, is_wormhole_b0
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@run_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [
            8,
            "silicon_arch_name=wormhole_b0-silicon_arch_wormhole_b0=True-sequence_size=384-batch_size=8-model_name=squeezebert/squeezebert-uncased-device_params={'l1_small_size': 16384}",
        ],
    ],
)
def test_perf_device_bare_metal(batch_size, test):
    subdir = "ttnn_squeezebert"
    num_iterations = 1
    margin = 0.03
    expected_perf = 290.35

    command = f"pytest tests/ttnn/integration_tests/squeezebert/test_ttnn_squeezebert_wh.py::test_squeezebert_for_question_answering"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = batch_size * 2 if mesh_device_flag else batch_size

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=False)
    prep_device_perf_report(
        model_name=f"ttnn_squeezebert_{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
