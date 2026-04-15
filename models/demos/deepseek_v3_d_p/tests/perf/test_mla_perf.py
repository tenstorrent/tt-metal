# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla"
            " -k 'balanced-skip_check-seq100k-scaled_sl-random-line-2x4'",
            9_665_361,
            "deepseek_v3_mla",
            "deepseek_v3_mla_2x4",
            1,
            1,
            0.03,
            "seq100k_scaled_balanced_line_2x4",
        ),
    ],
    ids=[
        "mla_2x4_seq100k",
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_deepseek_v3_mla_perf(
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
