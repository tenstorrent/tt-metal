# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf for the HCA block over a chunked prefill at the runner's chunk width (5120) on 8x4.

Chunked and not one shot, because the per-chunk work is what the runner actually does and the only
thing a gate on this block can defend. Both tests allocate their state before HCA_START, so neither
measures the allocation."""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_hca.py::test_hca_chunked_prefill_mesh"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'chunk5120-full and 8x4'"  # two 5120-token chunks


@pytest.mark.timeout(0)
def test_hca_block_perf_galaxy():
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy.")

    margin = adjust_margin_for_ddr_speed(0.05)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        # Two chunks, so ~4.54 ms of device a chunk. Highest of three 8x4 runs
        # (9,054,311 / 9,060,226 / 9,079,936), which the 5% margin covers.
        #
        # Not comparable to the old 4,292,000, which measured one chunk with the state allocated inside
        # the region. On this same leg, moving the mask onto the device cost 8,620,131 -> 9,054,311 ns,
        # +217 us a chunk, in exchange for 102 ms a chunk of host time (wall 122.0 -> 20.0 ms).
        expected_device_perf_ns_per_iteration=9_080_000,
        subdir="deepseek_v4_hca_block",
        model_name="deepseek_v4_hca_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("HCA_START", "HCA_END"),
        comments="chunk5120_x2_hca_chunked_prefill_glx_8x4_ground_truth",
    )
