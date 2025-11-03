# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.perf.device_perf_utils import run_model_device_perf_test
from models.experimental.panoptic_deeplab.reference.pytorch_model import PANOPTIC_DEEPLAB, DEEPLAB_V3_PLUS


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py::test_model_panoptic_deeplab -k test_panoptic_deeplab",
            51_749_520,
            PANOPTIC_DEEPLAB,
            PANOPTIC_DEEPLAB,
            1,
            1,
            0.015,
            "",
        ),
        (
            "pytest models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py::test_model_panoptic_deeplab -k test_deeplab_v3_plus",
            25_065_032,
            DEEPLAB_V3_PLUS,
            DEEPLAB_V3_PLUS,
            1,
            1,
            0.015,
            "",
        ),
    ],
    ids=["test_deeplab_v3_plus"],
)
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_pdl(
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
