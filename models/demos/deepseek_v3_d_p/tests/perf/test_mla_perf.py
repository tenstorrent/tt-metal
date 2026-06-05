# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    run_mla_perf_with_approximation,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla"

_CMD_2X4 = f"pytest {_TEST_PATH} -k 'balanced-skip_check-seq100k-scaled_sl-random-line-2x4'"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'balanced-skip_check-seq100k-scaled_sl-random-line-8x4'"


@pytest.mark.timeout(0)
def test_deepseek_v3_mla_perf_loudbox():
    """
    Measures perf on LB in 2x4 mesh shape, validates against its own perf baseline, and computes the approximate Galaxy perf.
    SDPA time is scaled by 4 (SP 2→8 = 4x, TP 4→4 = 1x) while other ops are added as-is.
    """
    run_mla_perf_with_approximation(
        command_2x4=_CMD_2X4,
        expected_ns_2x4=8_251_664,
        model_name_2x4="deepseek_v3_mla_lb_2x4",
        subdir="deepseek_v3_mla",
        margin=0.03,
        comments_2x4="seq100k_scaled_lb_2x4_proxy",
    )


@pytest.mark.timeout(0)
def test_deepseek_v3_mla_perf_galaxy():
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        expected_device_perf_ns_per_iteration=15_427_562,
        subdir="deepseek_v3_mla",
        model_name="deepseek_v3_mla_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=0.03,
        comments="seq100k_scaled_glx_8x4_ground_truth",
    )
