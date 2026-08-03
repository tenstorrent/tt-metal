# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf for the HCA block at the production chunk width (4096 tokens) on the 8x4 galaxy.

Single-shot rather than chunked on purpose: the op sequence is the one a chunk executes, minus the
128-row carry and the cache write (~3%), so the number tracks per-chunk cost without a one-chunk
scenario that would exercise no cross-chunk state.

Guards the whole block rather than one op, because that is what catches a silent undo of the tuned
constants -- q_chunk_size / k_chunk_size, ccl_num_links, the fused head ops. None of those move PCC.
Breakdown behind the number: hca_perf/GALAXY_PERF.md."""

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
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    # 5% rather than the usual 1.5-3%: SDPA alone drifts ~5% run to run. Still far tighter than
    # what it guards -- reverting q_chunk_size costs +60%, dropping the fused head ops +100%.
    margin = adjust_margin_for_ddr_speed(0.05)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        # Calibrated 2026-08-03 on bh-glx-110-c10u08, FABRIC_2D, 74 device ops.
        expected_device_perf_ns_per_iteration=3_237_000,
        subdir="deepseek_v4_hca_block",
        model_name="deepseek_v4_hca_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("HCA_START", "HCA_END"),
        comments="chunk4096_hca_prefill_glx_8x4_ground_truth",
    )
