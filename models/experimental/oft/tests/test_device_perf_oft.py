# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_model_device_perf_test


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/experimental/oft/tests/pcc/test_oftnet.py::test_oftnet -k bfp16_use_device_oft",
            204_613_230,
            "oft_oftnet",
            "oft_oftnet",
            1,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/experimental/oft/tests/pcc/test_encoder.py::test_decode",
            4_539_775,
            "oft_decoder",
            "oft_decoder",
            1,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/experimental/oft/demo/demo.py::test_demo_inference",
            198_716_319,
            "oft_full_demo",
            "oft_full_demo",
            1,
            1,
            0.015,
            "",
        ),
    ],
    ids=[
        "device_perf_oft_oftnet",
        "device_perf_oft_decoder",
        "device_perf_oft_full",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_oft(
    command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments
):
    run_model_device_perf_test(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
