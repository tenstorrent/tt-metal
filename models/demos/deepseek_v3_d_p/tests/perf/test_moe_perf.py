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
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ds_moe"

# `and pad0`/`and pad50` pins the padding parametrize so each command selects
# exactly one production TorusXY case.
_CMD_8X4_pad0 = f"pytest {_TEST_PATH} -k 'perf-device-256 and torus-xy-8x4 and pad0' --wrapper-invocation"
_CMD_8X4_pad50 = f"pytest {_TEST_PATH} -k 'perf-device-256 and torus-xy-8x4 and pad50' --wrapper-invocation"
_CMD_8X1 = f"pytest {_TEST_PATH} -k 'perf-host-64 and torus-y-8x1 and pad0' --wrapper-invocation"
_CMD_2X4 = f"pytest {_TEST_PATH} -k 'perf-device-256 and fabric2d-mesh-2x4 and pad0' --wrapper-invocation"


_IGNORE_POWER = os.environ.get("DS_PERF_IGNORE_POWER") == "1"
_REQUIRE_HIGH_POWER = pytest.mark.skipif(
    not (is_high_power() or _IGNORE_POWER),
    reason="galaxy perf baselines are cut on a >=130W TDP host; an 8kW galaxy measures differently. "
    "DS_PERF_IGNORE_POWER=1 runs it anyway, for bring-up only",
)


def _require_certified_torus_xy():
    if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1" or not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        pytest.fail("TorusXY perf requires a certified Galaxy and explicit mesh graph descriptor")


@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_loudbox():
    """Run the existing 8x1 + 2x4 proxies and retain their 8x4 approximation signal.

    The 8x1 SP proxy runs Fabric2d TorusY; the 2x4 TP proxy runs unwrapped Fabric2d because its
    two-wide SP dimension cannot form a useful ring. approximate_8x4_perf takes every non-TP op
    from the 8x1 slot, so that slot must stay an SP=8 run.
    """
    run_moe_perf_with_approximation(
        command_8x1=_CMD_8X1,
        # Re-cut 2026-08-28 on the CI LoudBox (bh_loudbox), run 33194029504. One sample.
        # The 2D matmul program configs are the whole delta: against main on the same box and day
        # the Matmul bucket falls 2,201,295 -> 381,174 ns while Other (3,615,143 -> 3,617,678) and
        # CCL (17,690 -> 22,700) hold, so 5_895_298 centred on a matmul shape nothing builds now.
        expected_ns_8x1=4_021_552,
        model_name_8x1="deepseek_v3_moe_lb_8x1_torus_y_dispatch_combine",
        command_2x4=_CMD_2X4,
        # Re-cut 2026-08-28 on the CI LoudBox (bh_loudbox), run 33194029504. One sample,
        # superseding the 9_339_547 two-run CI mean cut earlier the same day. Same cause as 8x1:
        # Matmul 715,503 -> 216,878 ns against main, Other flat within 0.2%. This gate still has
        # to be cut on the CI runner -- the dev box bh-lb-15 reads it 2.7% slower.
        expected_ns_2x4=8_840_595,
        model_name_2x4="deepseek_v3_moe_lb_2x4_fabric2d_gate",
        subdir="deepseek_v3_moe",
        margin=0.03,
        comments_8x1="isl5k_lb_8x1_torus_y_dispatch_combine_proxy",
        comments_2x4="isl5k_lb_2x4_fabric2d_gate_proxy",
    )


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_galaxy():
    """Measure the production 8x4 TorusXY Galaxy path without padding."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad0,
        # Measured 2026-08-22, 14kW BH galaxy bh-glx-110-c04u02, 8x4 TorusXY certified, DDR 16000.
        # Two runs 6.155 / 6.070 ms, spread 1.38%.
        expected_device_perf_ns_per_iteration=6_112_530,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="isl5k_glx_8x4_ground_truth",
    )


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_galaxy_pad50():
    """8x4 galaxy ground truth with 50% right-padding + padding-aware routing (zigzag placement)."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad50,
        # Measured 2026-08-22, 14kW BH galaxy bh-glx-110-c04u02, 8x4 TorusXY certified, DDR 16000.
        # Two runs 5.223 / 5.188 ms, spread 0.67%.
        expected_device_perf_ns_per_iteration=5_205_872,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4_pad50",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="isl5k_glx_8x4_ground_truth_padded_50_percent_w_awareness",
    )
