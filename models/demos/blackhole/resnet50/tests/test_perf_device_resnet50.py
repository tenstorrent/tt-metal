# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.ttnn_resnet.tests.perf_device_resnet50 import run_perf_device
from models.utility_functions import run_for_blackhole


@run_for_blackhole()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [16, "act_dtype0-weight_dtype0-math_fidelity0-16-device_params0", 10113.0],
        [32, "act_dtype0-weight_dtype0-math_fidelity0-32-device_params0", 11842.0],
    ],
)
def test_perf_device(batch_size, test, expected_perf):
    command = (
        f"pytest models/demos/blackhole/resnet50/tests/test_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    )
    run_perf_device(batch_size, test, command, expected_perf)
