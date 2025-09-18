# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.ttnn_resnet.tests.perf_device_resnet50 import run_perf_device
from models.demos.ttnn_resnet.tt.ttnn_functional_resnet50_model_utils import is_blackhole_p100


# Determine device type at module level to avoid conflicts
def get_device_type():
    device = ttnn.open_device(device_id=0)
    device_type = "p100" if is_blackhole_p100(device) else "p150"
    ttnn.close_device(device)
    return device_type


DEVICE_TYPE = get_device_type()


@run_for_blackhole()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf, device_type",
    [
        pytest.param(
            16,
            "act_dtype0-weight_dtype0-math_fidelity0-16-device_params0",
            8371.0,
            "p100",
            marks=pytest.mark.skipif(DEVICE_TYPE != "p100", reason=f"Skipping P100 test on {DEVICE_TYPE} device"),
        ),
        pytest.param(
            16,
            "act_dtype0-weight_dtype0-math_fidelity0-16-device_params0",
            10601.0,
            "p150",
            marks=pytest.mark.skipif(DEVICE_TYPE != "p150", reason=f"Skipping P150 test on {DEVICE_TYPE} device"),
        ),
        pytest.param(
            32,
            "act_dtype0-weight_dtype0-math_fidelity0-32-device_params0",
            14779.0,
            "p150",
            marks=pytest.mark.skipif(DEVICE_TYPE != "p150", reason=f"Skipping P150 test on {DEVICE_TYPE} device"),
        ),
        pytest.param(
            32,
            "act_dtype0-weight_dtype0-math_fidelity0-32-device_params0",
            9380.0,
            "p100",
            marks=pytest.mark.skipif(DEVICE_TYPE != "p100", reason=f"Skipping P100 test on {DEVICE_TYPE} device"),
        ),
    ],
)
def test_perf_device(batch_size, test, expected_perf, device_type):
    command = (
        f"pytest models/demos/blackhole/resnet50/tests/test_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    )
    run_perf_device(batch_size, test, command, expected_perf)
