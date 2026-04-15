# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe"
            " -k 'mesh-2x4-3200_no_pcc and mesh-2x4 and not linear-8 and not mesh-4x2 and not mesh-8x4'",
            93_094_964,
            "deepseek_v3_moe",
            "deepseek_v3_moe_2x4",
            1,
            1,
            0.03,
            "seq3200_mesh_2x4",
        ),
    ],
    ids=[
        "moe_2x4_seq3200",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_deepseek_v3_moe_perf(
    command,
    expected_device_perf_ns_per_iteration,
    subdir,
    model_name,
    num_iterations,
    batch_size,
    margin,
    comments,
):
    run_model_device_perf_test_with_merge(
        command=command,
        expected_device_perf_ns_per_iteration=expected_device_perf_ns_per_iteration,
        subdir=subdir,
        model_name=model_name,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments,
    )
