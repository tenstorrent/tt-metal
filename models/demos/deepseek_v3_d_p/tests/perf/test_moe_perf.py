# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""MoE device-perf proxies on LoudBox plus production 8x4 TorusXY ground truth."""

import os

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
    run_moe_perf_with_approximation,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ds_moe"

# `and pad0`/`and pad50` pins the padding parametrize so each command selects
# exactly one production TorusXY case.
_CMD_8X4_pad0 = f"pytest {_TEST_PATH} -k 'perf-device-256 and torus-xy-8x4 and pad0' --wrapper-invocation"
_CMD_8X4_pad50 = f"pytest {_TEST_PATH} -k 'perf-device-256 and torus-xy-8x4 and pad50' --wrapper-invocation"
_CMD_8X1 = f"pytest {_TEST_PATH} -k 'perf-host-64 and torus-y-8x1 and pad0' --wrapper-invocation"
_CMD_2X4 = f"pytest {_TEST_PATH} -k 'perf-device-256 and fabric2d-mesh-2x4 and pad0' --wrapper-invocation"


# The four baselines below were all measured at seq_len_per_chip=3200. The rows now run
# ISL_TOKENS_PER_CHIP (640), and device time does not scale with ISL -- fixed dispatch overhead, CCL
# latency floors and expert-loop tails all stay put -- so the numbers describe a workload that no
# longer exists. They are kept as history and the assertions are disabled until someone re-measures
# on the gating box (Galaxy 8x4 for the two _galaxy tests, LoudBox 8x1 + 2x4 for the proxy pair).
_ISL_REBASELINE_SKIP = pytest.mark.skip(
    reason="baseline measured at 3200 tokens/chip; rows now run 640/chip. Re-measure on the gating "
    "box, then drop this mark."
)


def _require_certified_torus_xy():
    if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1" or not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        pytest.fail("TorusXY perf requires a certified Galaxy and explicit mesh graph descriptor")


@_ISL_REBASELINE_SKIP
@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_loudbox():
    """Run the existing 8x1 + 2x4 proxies and retain their 8x4 approximation signal.

    The 8x1 SP proxy runs Fabric2d TorusY; the 2x4 TP proxy runs unwrapped Fabric2d because its
    two-wide SP dimension cannot form a useful ring. approximate_8x4_perf takes every non-TP op
    from the 8x1 slot, so that slot must stay an SP=8 run.
    """
    run_moe_perf_with_approximation(
        command_8x1=_CMD_8X1,
        # Recalibrated 2026-08-21 on this LoudBox, Fabric2D TorusY, with the routed experts folded
        # into one program. Single run each, where the previous 8x1 was a mean of two.
        expected_ns_8x1=14_385_886,
        model_name_8x1="deepseek_v3_moe_lb_8x1_torus_y_dispatch_combine",
        command_2x4=_CMD_2X4,
        expected_ns_2x4=15_945_512,
        model_name_2x4="deepseek_v3_moe_lb_2x4_fabric2d_gate",
        subdir="deepseek_v3_moe",
        margin=0.03,
        comments_8x1="isl5k_lb_8x1_torus_y_dispatch_combine_proxy",
        comments_2x4="isl5k_lb_2x4_fabric2d_gate_proxy",
    )


@_ISL_REBASELINE_SKIP
@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_galaxy():
    """Measure the production 8x4 TorusXY Galaxy path without padding."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad0,
        # Historical baseline: measured with FABRIC_1D, not TorusXY.
        expected_device_perf_ns_per_iteration=21_028_751,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="isl5k_glx_8x4_ground_truth",
    )


@_ISL_REBASELINE_SKIP
@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_galaxy_pad50():
    """8x4 galaxy ground truth with 50% right-padding + padding-aware routing (zigzag placement)."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad50,
        # Historical baseline: measured with FABRIC_1D, not TorusXY.
        expected_device_perf_ns_per_iteration=14_107_228,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4_pad50",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="isl5k_glx_8x4_ground_truth_padded_50_percent_w_awareness",
    )
