# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Performance tests for DeepSeek V3 MoE (Mixture of Experts) layer.

This test suite measures device kernel performance for the full MoE pipeline
on mesh-8x4-2 topology (8 devices in linear configuration with 2 links).

The test executes the existing PCC test and measures performance using the
Tracy profiler. Initial performance targets are dummy values that should be
updated after the first run with actual measured values.
"""

import pytest

from models.demos.deepseek_v3_d_p.tests.conftest import is_galaxy
from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe -k 'mesh-8x4-1600_no_pcc'",
            64_403_570,  # AVG DEVICE KERNEL DURATION [ns]
            "deepseek_v3_moe",
            "deepseek_v3_moe_mesh_8x4_2",
            1,
            1,
            0.03,  # 3% margin
            "seq_len_1600",
        ),
        (
            "pytest models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe -k 'mesh-8x4-3200_no_pcc'",
            85_869_760,  # AVG DEVICE KERNEL DURATION [ns]
            "deepseek_v3_moe",
            "deepseek_v3_moe_mesh_8x4_2",
            1,
            1,
            0.03,  # 3% margin
            "seq_len_3200",
        ),
    ],
    ids=[
        "mesh-8x4-2-seq1600_no_pcc",
        "mesh-8x4-2-seq3200_no_pcc",
    ],
)
@pytest.mark.skipif(not is_galaxy(), reason="Test only runs on a Galaxy.")
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
    """
    Performance test for DeepSeek V3 MoE.

    This test runs the full MoE pipeline (dispatch, routed experts, shared expert,
    combine, reduce) and measures device kernel execution time using Tracy profiler.

    Args:
        command: Pytest command to execute the PCC test
        expected_device_perf_ns_per_iteration: Target performance in ns (dummy value initially)
        subdir: Output directory for performance reports
        model_name: Model identifier for tracking
        num_iterations: Number of profiling iterations
        batch_size: Batch size (currently 1)
        margin: Acceptable performance variance (0.03 = 3%)
        comments: Test case identifier (sequence length)
    """
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
