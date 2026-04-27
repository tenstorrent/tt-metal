# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MLA device-perf tests approximating glx 8x4 from LB (8-chip) proxy.

- `test_deepseek_v3_mla_perf_loudbox`: runs 2x4 proxy on loudbox, validates against
  its own perf baseline, and computes the approximate 8x4 galaxy total. SDPA time
  is scaled by 4 (SP 2→8 = 4x, TP 4→4 = 1x) while other ops are added as-is.
- `test_deepseek_v3_mla_perf_galaxy`: 8x4 ground truth (skipped off-glx).
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    run_mla_perf_with_approximation,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla"

_CMD_2X4 = f"pytest {_TEST_PATH} -k 'balanced-skip_check-seq100k-scaled_sl-random-line-2x4'"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'balanced-skip_check-seq100k-scaled_sl-random-line-8x4'"


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(1000)
def test_deepseek_v3_mla_perf_loudbox():
    """
    Run 2x4 proxy on loudbox (BH-LoudBox, 8xP150).
    Validates proxy against its own baseline AND computes the approximate
    8x4 galaxy total: SDPA × 4 + other ops.
    """
    run_mla_perf_with_approximation(
        command_2x4=_CMD_2X4,
        expected_ns_2x4=8_800_538,
        model_name_2x4="deepseek_v3_mla_lb_2x4",
        subdir="deepseek_v3_mla",
        margin=0.03,
        comments_2x4="seq100k_scaled_lb_2x4_proxy",
    )


@pytest.mark.models_device_performance_bare_metal
def test_deepseek_v3_mla_perf_galaxy():
    """8x4 galaxy ground truth — the reference the loudbox approximation targets."""
    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        expected_device_perf_ns_per_iteration=18_199_125,
        subdir="deepseek_v3_mla",
        model_name="deepseek_v3_mla_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=0.03,
        comments="seq100k_scaled_glx_8x4_ground_truth",
    )
