# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.utility_functions import skip_for_grayskull, is_wormhole_b0
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report


@skip_for_grayskull()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test",
    [
        [
            16,
            "silicon_arch_name=wormhole_b0-silicon_arch_wormhole_b0=True-sequence_size=384-batch_size=8-model_name=squeezebert/squeezebert-uncased-device_params={'l1_small_size': 16384}",
        ],
    ],
)
def test_perf_device_bare_metal(batch_size, test):
    subdir = "ttnn_squeezebert"
    num_iterations = 1
    margin = 0.03
    expected_perf = 219.5

    command = f"pytest tests/ttnn/integration_tests/squeezebert/test_ttnn_squeezebert.py::test_squeezebert_for_question_answering"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"ttnn_squeezebert_{batch_size}",
        batch_size=batch_size if mesh_device_flag else 8,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test.replace("/", "_"),
    )
