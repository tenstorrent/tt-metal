# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""MoE device-perf tests on the production 8x4 TorusXY Galaxy profile."""

import os

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ds_moe"

# `and pad0`/`and pad50` pins the padding parametrize so each command selects
# exactly one production TorusXY case.
_CMD_8X4_pad0 = f"pytest {_TEST_PATH} -k 'perf-device-256 and torus-xy-8x4 and pad0'"
_CMD_8X4_pad50 = f"pytest {_TEST_PATH} -k 'perf-device-256 and torus-xy-8x4 and pad50'"


@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_galaxy():
    """Measure the production 8x4 TorusXY Galaxy path without padding."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad0,
        # Record-only until calibrated on TorusXY; 21,028,751 ns was measured on Fabric1D.
        expected_device_perf_ns_per_iteration=None,
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

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad50,
        # Record-only until calibrated on TorusXY; 14,107,228 ns was measured on Fabric1D.
        expected_device_perf_ns_per_iteration=None,
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4_pad50",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="seq3200_glx_8x4_ground_truth_padded_50_percent_w_awareness",
    )


# --- Kimi-K3 LatentMoE ---------------------------------------------------------------------------
# Own command constant rather than a variant of the DeepSeek ones: K3 has its own test function
# (test_kimi_k3_moe), and its ids are already model-scoped.
_K3_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_kimi_k3_moe"
_CMD_K3_8X4 = f"pytest {_K3_TEST_PATH} -k 'torus-xy-8x4 and kimi_k3-5k-perf'"


def _require_certified_torus_xy():
    if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1" or not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        pytest.skip("TorusXY perf requires a certified Galaxy and explicit mesh graph descriptor")


@pytest.mark.timeout(0)
def test_kimi_k3_moe_perf_galaxy():
    """Kimi-K3 LatentMoE device perf on the 8x4 Galaxy at the production shape: 640 tokens/chip
    (5120 total = 5K ISL), 896 experts, top-16, routed side at the 3584 latent width, FABRIC_2D.

    NOT comparable to test_deepseek_v3_moe_perf_galaxy: that baseline is 256 experts / top-8 at the
    full 7168 width with no latent projections AND runs 3200 tokens/chip, whereas this adds a
    7168->3584 down-projection plus all-gather before dispatch and a latent RMSNorm plus 3584->7168
    row-parallel up-projection plus reduce-scatter after the reduce, against half-width dispatch
    traffic and 2x the top-k accumulation depth -- at a fifth of the sequence.

    Measures the SiLU path, not the checkpoint's SiTU-GLU -- no TT kernel implements SiTU yet
    (#51335). Expect this baseline to move when that lands, since the activation sits inside the
    routed-expert FFN that dominates the block.

    Bracketed by the MoE_START/MoE_END signposts so the number is the forward only: TtMoe's
    constructor dispatches one-time weight-load tilize/typecast work before MoE_START, and at 896
    experts that is a large fraction of the process wall time but is not per-token cost.
    """
    _require_certified_torus_xy()
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_K3_8X4,
        # Historical unwrapped-Fabric2D result, measured 2026-08-07 on bh-glx-120-c04u02 (8x4,
        # DDR 14000 -- sub-nominal, so
        # adjust_margin_for_ddr_speed doubles the 3% margin to 6%). Mean of two consecutive runs,
        # 12_922_557 and 12_927_147, which agree to 0.036%. Signpost-filtered and device-row-merged.
        # Split at this shape: Matmul 2_196 us, CCL 1_042 us, Other 9_687 us -- the block is
        # dominated by the dispatch/combine/top-k path, not by the latent projections.
        #
        # Superseded the earlier 28_872_936, which was 3200 tokens/chip on FABRIC_1D. Dropping to
        # the 5K production shape cut the sequence 5x but the time only 2.2x, because the per-block
        # fixed costs and the CCLs do not scale with sequence.
        # Record-only until this unchanged production shape is recalibrated on certified TorusXY;
        # the historical 12_924_852 ns value must not gate Ring/Ring traffic.
        expected_device_perf_ns_per_iteration=None,
        subdir="kimi_k3_moe",
        model_name="kimi_k3_moe_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("MoE_START", "MoE_END"),
        comments="seq640_5k_isl_glx_8x4_torus_xy_recalibration_latent_moe_silu",
    )
