# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MoE device-perf tests approximating glx 8x4 from LB (8-chip) proxies.

- `test_deepseek_v3_moe_perf_loudbox`: runs 8x1 and 2x4 proxies once each, validates
  each against its own perf baseline, and in the same pass computes the approximate
  8x4 galaxy total (SP ops from 8x1 + TP ops from 2x4). One test, two device runs,
  three signals: per-proxy perf regression catches + approximation artifact.
- `test_deepseek_v3_moe_perf_galaxy`: 8x4 ground truth (skipped off-glx).

Sum of per-op approximation approximates one glx column's MoE block kernel time;
the 8x4 ground-truth test is the reference the approximation is compared against.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    run_model_device_perf_test_with_merge,
    run_moe_perf_with_approximation,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ttnn_moe"

_CMD_8X1 = f"pytest {_TEST_PATH} -k 'perf-host-64 and linear-8'"
_CMD_2X4 = f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-2x4 and not linear-8 and not mesh-4x2 and not mesh-8x4'"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-8x4 and not linear-8 and not mesh-4x2 and not mesh-2x4'"


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(1000)
def test_deepseek_v3_moe_perf_loudbox():
    """
    Run 8x1 + 2x4 proxies once each on loudbox (BH-LoudBox, 8xP150).
    Validates each proxy against its own baseline AND computes the approximate
    8x4 total from the same two CSVs (no extra device work).
    """
    run_moe_perf_with_approximation(
        command_8x1=_CMD_8X1,
        expected_ns_8x1=30_021_495,
        model_name_8x1="deepseek_v3_moe_lb_8x1_dispatch_combine",
        command_2x4=_CMD_2X4,
        expected_ns_2x4=31_367_500,
        model_name_2x4="deepseek_v3_moe_lb_2x4_gate",
        subdir="deepseek_v3_moe",
        margin=0.03,
        comments_8x1="seq3200_lb_8x1_dispatch_combine_proxy",
        comments_2x4="seq3200_lb_2x4_gate_proxy",
    )


@pytest.mark.models_device_performance_bare_metal
def test_deepseek_v3_moe_perf_galaxy():
    """8x4 galaxy ground truth — the reference the loudbox approximation targets."""
    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        expected_device_perf_ns_per_iteration=36_771_539,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=0.03,
        comments="seq3200_glx_8x4_ground_truth",
    )
