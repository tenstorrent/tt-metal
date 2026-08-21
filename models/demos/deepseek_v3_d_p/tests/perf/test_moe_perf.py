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
_CMD_2X4_HOST64 = f"pytest {_TEST_PATH} -k 'perf-host-64 and fabric2d-mesh-2x4 and pad0' --wrapper-invocation"
_CMD_2X4 = f"pytest {_TEST_PATH} -k 'perf-device-256 and fabric2d-mesh-2x4 and pad0' --wrapper-invocation"


def _require_certified_torus_xy():
    if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1" or not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        pytest.fail("TorusXY perf requires a certified Galaxy and explicit mesh graph descriptor")


@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_loudbox():
    """Run the host-64 + device-256 proxies on 2x4 and retain their 8x4 approximation signal.

    Both proxies run the unwrapped-Fabric2d 2x4 mesh: the SP dispatch/combine proxy (host-64
    shape) and the TP gate proxy (device-256 shape).
    """
    run_moe_perf_with_approximation(
        command_8x1=_CMD_2X4_HOST64,
        # Stale: carried over from the retired 8x1 TorusY proxy; recalibrate on 2x4 Fabric2D.
        expected_ns_8x1=15_393_888,
        model_name_8x1="deepseek_v3_moe_lb_2x4_host64_dispatch_combine",
        command_2x4=_CMD_2X4,
        expected_ns_2x4=17_217_341,  # Recalibrated 2026-08-14 on this LoudBox with Fabric2D.
        model_name_2x4="deepseek_v3_moe_lb_2x4_fabric2d_gate",
        subdir="deepseek_v3_moe",
        margin=0.03,
        comments_8x1="seq3200_lb_2x4_host64_dispatch_combine_proxy",
        comments_2x4="seq3200_lb_2x4_fabric2d_gate_proxy",
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
        # Historical baseline: measured with FABRIC_1D, not TorusXY.
        expected_device_perf_ns_per_iteration=21_028_751,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="seq3200_glx_8x4_ground_truth",
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
        # Historical baseline: measured with FABRIC_1D, not TorusXY.
        expected_device_perf_ns_per_iteration=14_107_228,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4_pad50",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="seq3200_glx_8x4_ground_truth_padded_50_percent_w_awareness",
    )
