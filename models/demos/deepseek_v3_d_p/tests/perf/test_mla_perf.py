# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)

# Kimi K2.6 chunked prefill: 50k KV-cache prefix + one fresh 5k chunk (chunk_size_global=5120). On
# the 8x4 Galaxy (sp=8) this lands chunk_local=640 per chip, exercising the num_heads=64 chunked-only
# 640 matmul/SDPA configs. Functional reference (no PCC) keeps the measured region to the single
# forward (the 50k prefix is preloaded host->device before the MLA_START signpost, so it is not timed).
_CHUNKED_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill"
_CMD_CHUNKED_8X4 = f"pytest {_CHUNKED_TEST_PATH} -k 'deep-50k+5k and kimi and func and 8x4 and fabric2d'"


@pytest.mark.timeout(0)
def test_kimi_mla_chunked_perf_galaxy():
    """Kimi K2.6 chunked-prefill MLA perf on the 8x4 Galaxy: 50k KV-cache prefix + one fresh 5k chunk
    (640 tokens/chip). Functional (no reference), so the single timed forward exercises the chunked
    640 matmul/SDPA configs end to end. Ground-truth 8x4 measurement (no 2x4 approximation)."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_CHUNKED_8X4,
        expected_device_perf_ns_per_iteration=7_118_649,
        subdir="deepseek_v3_mla",
        model_name="kimi_mla_chunked_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        # Time only the forward: ops between the MLA_START/MLA_END signposts, excluding one-time
        # weight-load tilize/typecast at construction (dispatched before MLA_START).
        between_signposts=("MLA_START", "MLA_END"),
        comments="kimi_chunked_50k+5k_glx_8x4_ground_truth",
    )
