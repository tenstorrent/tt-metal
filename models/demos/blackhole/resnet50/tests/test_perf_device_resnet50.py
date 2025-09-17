# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.ttnn_resnet.tests.perf_device_resnet50 import run_perf_device
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import is_blackhole_p100


@run_for_blackhole()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf, device_type",
    [
        [16, "act_dtype0-weight_dtype0-math_fidelity0-16-device_params0", 10193.0, "p150"],
        [16, "act_dtype0-weight_dtype0-math_fidelity0-16-device_params0", 8371.0, "p100"],
        [32, "act_dtype0-weight_dtype0-math_fidelity0-32-device_params0", 12000.0, "p150"],
        [32, "act_dtype0-weight_dtype0-math_fidelity0-32-device_params0", 9380.0, "p100"],
    ],
)
def test_perf_device(batch_size, test, expected_perf, device_type):
    actual_device_type = None
    with ttnn.manage_device(0) as device:
        actual_device_type = "p100" if is_blackhole_p100(device) else "p150"

    if actual_device_type != device_type:
        pytest.skip(f"Skipping Test for device {actual_device_type}")

    command = (
        f"pytest models/demos/blackhole/resnet50/tests/test_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    )
    run_perf_device(batch_size, test, command, expected_perf)
