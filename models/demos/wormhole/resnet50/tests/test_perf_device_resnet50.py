# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.ttnn_resnet.tests.perf_device_resnet50 import run_perf_device
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [16, "16-act_dtype0-weight_dtype0-math_fidelity0-device_params0", 5311.0],
    ],
)
def test_perf_device(batch_size, test, expected_perf):
    command = (
        f"pytest models/demos/wormhole/resnet50/tests/test_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    )
    run_perf_device(batch_size, test, command, expected_perf)
