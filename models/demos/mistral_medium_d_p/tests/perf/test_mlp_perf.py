# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE PERF: the dense SwiGLU MLP block, kernel-time gate.

Reuses ``tests/unit/test_dense_mlp_vs_ref.py`` as the worker: runs it in a subprocess under the
tracy profiler, sums the merged device kernel durations between the MLP_START/MLP_END signposts
(one forward — weight-load tilize and the input write are dispatched outside them), and asserts
a two-sided band around the baseline. Multi-chip rows merge as max-across-chips for compute ops
and avg for collectives (see utils/perf_utils.py, ported from the deepseek_v3_d_p harness).

Run:  MESH_DEVICE=TG pytest models/demos/mistral_medium_d_p/tests/perf/test_mlp_perf.py
"""

import pytest

from models.demos.mistral_medium_d_p.utils.perf_utils import (
    adjust_margin_for_ddr_speed,
    is_galaxy_env,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/mistral_medium_d_p/tests/unit/test_dense_mlp_vs_ref.py::test_dense_mlp_vs_ref"

# Keep the -k pinned to exactly one parametrization: the signpost filter keeps EVERY
# MLP_START/MLP_END region in the profile, so a selector matching two cases would sum two forwards
# (see deepseek_v3_d_p/tests/perf/test_mla_perf.py for the incident that rule comes from).
_CMD_8X4 = f"pytest {_TEST_PATH} -k '8x4 and s5k'"


@pytest.mark.timeout(0)
def test_mistral_medium_mlp_perf_galaxy():
    """Dense SwiGLU MLP (gate/up fused column-parallel matmul, silu-mul, row-parallel down,
    TP reduce-scatter) at s=5k global, SP-sharded to 640 tokens/chip, on the 8x4 Galaxy — the
    SP=8 x TP=4 hardware target."""
    if not is_galaxy_env():
        pytest.skip("This test requires an 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        # s=5k global SP-sharded across the 8 rows -> 640 tokens/chip (the worker previously
        # replicated all 5k tokens to every chip; the old baseline at that shape was 14_155_903).
        # Measured 2026-08-27 on this 8x4 BH Galaxy (FABRIC_1D linear). Breakdown at capture:
        # Matmul 1,569us / ReduceScatter 315us / Slice 105us / silu-mul 76us.
        expected_device_perf_ns_per_iteration=2_065_560,
        subdir="mistral_medium_mlp",
        model_name="mistral_medium_mlp_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("MLP_START", "MLP_END"),
        comments="dense_swiglu_s5k_sp8_glx_8x4",
    )
