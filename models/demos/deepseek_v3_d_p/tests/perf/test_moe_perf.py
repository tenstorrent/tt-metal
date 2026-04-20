# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MoE device-perf tests approximating glx 8x4 from LB (8-chip) proxies.

- `moe_lb_8x1_dispatch_combine`: LB linear-8. Measures dispatch + combine (SP all-to-all)
  using 64 experts + 2 picks/tok so per-chip traffic matches one glx column.
- `moe_lb_2x4_gate`: LB mesh-2x4. Measures DEVICE gate matmul + TP all-reduce
  (TP=4 matches glx); 256 experts required by the device grouped_gate kernel.
- `moe_glx_8x4`: glx 8x4 ground truth (skipped off-glx).

Sum of (1) + (2) approximates one glx column's MoE block kernel time.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_with_merge

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe"


@pytest.mark.parametrize(
    "command, expected_device_perf_ns_per_iteration, subdir, model_name, num_iterations, batch_size, margin, comments",
    [
        (
            f"pytest {_TEST_PATH} -k 'perf-host-64 and linear-8'",
            1,  # TODO: set baseline after first run
            "deepseek_v3_moe",
            "deepseek_v3_moe_lb_8x1_dispatch_combine",
            1,
            1,
            0.03,
            "seq3200_lb_8x1_dispatch_combine_proxy",
        ),
        (
            f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-2x4 and not linear-8 and not mesh-4x2 and not mesh-8x4'",
            1,  # TODO: set baseline after first run
            "deepseek_v3_moe",
            "deepseek_v3_moe_lb_2x4_gate",
            1,
            1,
            0.03,
            "seq3200_lb_2x4_gate_proxy",
        ),
        (
            f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-8x4 and not linear-8 and not mesh-4x2 and not mesh-2x4'",
            1,  # TODO: set baseline after first run
            "deepseek_v3_moe",
            "deepseek_v3_moe_glx_8x4",
            1,
            1,
            0.03,
            "seq3200_glx_8x4_ground_truth",
        ),
    ],
    ids=[
        "moe_lb_8x1_dispatch_combine",
        "moe_lb_2x4_gate",
        "moe_glx_8x4",
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
