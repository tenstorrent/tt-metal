# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf for the HCA block over a chunked prefill at the runner's chunk width (5120) on 8x4.

Chunked and not one shot, because the per-chunk work is what the runner does and the only thing a gate on
this block can defend. The state is allocated before HCA_START, so the region does not measure it."""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_hca.py::test_hca_chunked_prefill_mesh"
# (variant, expected ns). Each number covers the WHOLE region -- BOTH chunks -- and is the highest of a
# few runs, rounded up. The variant has to be named in -k, or the region would cover four chunks.
_BASELINES = [
    pytest.param("flash", 9_080_000, id="flash"),  # 2 chunks -> 4.53 ms a chunk
    pytest.param("pro", 16_380_000, id="pro"),  # 2 chunks -> 8.18 ms a chunk
]


@pytest.mark.timeout(0)
@pytest.mark.parametrize("variant, expected_ns", _BASELINES)
def test_hca_block_perf_galaxy(variant, expected_ns):
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy.")

    margin = adjust_margin_for_ddr_speed(0.05)

    run_model_device_perf_test_with_merge(
        command=f"pytest {_TEST_PATH} -k '{variant} and chunk5120-full and 8x4'",
        expected_device_perf_ns_per_iteration=expected_ns,
        subdir=f"deepseek_v4_hca_block_{variant}",
        model_name=f"deepseek_v4_hca_{variant}_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("HCA_START", "HCA_END"),
        comments=f"chunk5120_x2_hca_chunked_prefill_{variant}_glx_8x4_ground_truth",
    )
