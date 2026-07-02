# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.vision.classification.resnet50.quasar.tests.common.perf_device_resnet50 import run_perf_device


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, test, expected_perf",
    [
        [16, "True-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0", 6335.0],
    ],
)
def test_perf_device(batch_size, test, expected_perf):
    command = f"pytest models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_performant.py::test_run_resnet50_inference[{test}]"
    run_perf_device(batch_size, test, command, expected_perf)
