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
        # Re-measured 2026-08-22 at 640 tokens/chip, BH LoudBox bh-lb-15, DDR 16000, 150W.
        # Mean of 14 runs, 5.876-5.921 ms, 0.76% peak to peak.
        expected_ns_8x1=5_895_298,
        model_name_8x1="deepseek_v3_moe_lb_8x1_torus_y_dispatch_combine",
        command_2x4=_CMD_2X4,
        # Same box and method. 9.548-9.655 ms, 1.12% peak to peak.
        expected_ns_2x4=9_601_530,
        model_name_2x4="deepseek_v3_moe_lb_2x4_fabric2d_gate",
        subdir="deepseek_v3_moe",
        margin=0.03,
        comments_8x1="isl5k_lb_8x1_torus_y_dispatch_combine_proxy",
        comments_2x4="isl5k_lb_2x4_fabric2d_gate_proxy",
    )


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
