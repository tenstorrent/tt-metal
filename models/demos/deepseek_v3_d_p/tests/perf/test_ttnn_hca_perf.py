# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf for the HCA block at the production chunk width (4096 tokens) on the 8x4 galaxy."""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_hca.py::test_hca_forward_mesh"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'seq4096 and 8x4'"


@pytest.mark.timeout(0)
def test_hca_block_perf_galaxy():
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy.")

    margin = adjust_margin_for_ddr_speed(0.05)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        expected_device_perf_ns_per_iteration=3_237_000,
        subdir="deepseek_v4_hca_block",
        model_name="deepseek_v4_hca_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("HCA_START", "HCA_END"),
        comments="chunk4096_hca_prefill_glx_8x4_ground_truth",
    )
