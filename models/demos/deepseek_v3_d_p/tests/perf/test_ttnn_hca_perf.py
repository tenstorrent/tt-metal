# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf for the HCA block at the runner's chunk width (5120 tokens) on the 8x4 galaxy."""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_hca.py::test_hca_forward_mesh"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'seq5120 and 8x4'"


@pytest.mark.timeout(0)
def test_hca_block_perf_galaxy():
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy.")

    margin = adjust_margin_for_ddr_speed(0.05)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        # Highest of three 8x4 runs (4,255,437 / 4,267,156 / 4,291,993), which the 5% margin covers.
        # Up from 3,237,000 at seq4096: the widening is SDPA, 1,020 us -> 1,789 us, since Sk carries both
        # the longer chunk and the deeper compressed cache.
        expected_device_perf_ns_per_iteration=4_292_000,
        subdir="deepseek_v4_hca_block",
        model_name="deepseek_v4_hca_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("HCA_START", "HCA_END"),
        comments="chunk5120_hca_prefill_glx_8x4_ground_truth",
    )
