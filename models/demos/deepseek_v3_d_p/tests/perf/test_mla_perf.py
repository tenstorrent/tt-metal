# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_mla_perf_loudbox,
    run_model_device_perf_test_with_merge,
)
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_ds_mla"

# 640 tokens/chip on both: 8x4 takes seq5k literally, 2x4 scales it to 1280 global.
_CMD_2X4 = f"pytest {_TEST_PATH} -k 'balanced and skip_check and seq5k and scaled_sl and random and fabric2d-2x4' --wrapper-invocation"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'balanced and skip_check and seq5k and max_sl and random and torus-xy-8x4' --wrapper-invocation"

# Kimi K2.6 chunked prefill: 50k cache + one 5k chunk. The 50k prefix is preloaded before the
# MLA_START signpost, so only the single forward is timed.
_CHUNKED_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill"
_CMD_CHUNKED_8X4 = (
    f"pytest {_CHUNKED_TEST_PATH} -k 'deep-50k+5k and k2_7 and func and torus-xy-8x4' --wrapper-invocation"
)


_IGNORE_POWER = os.environ.get("DS_PERF_IGNORE_POWER") == "1"
_REQUIRE_HIGH_POWER = pytest.mark.skipif(
    not (is_high_power() or _IGNORE_POWER),
    reason="galaxy perf baselines are cut on a >=130W TDP host; an 8kW galaxy measures differently. "
    "DS_PERF_IGNORE_POWER=1 runs it anyway, for bring-up only",
)


def _require_certified_torus_xy():
    if os.getenv("PREFILL_TORUS_XY_CERTIFIED") != "1" or not os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        pytest.fail("TorusXY perf requires a certified Galaxy and explicit mesh graph descriptor")


# Kimi K3 (NoPE + output gate, 96 heads): same scenario/mesh as the K2.7 command above. 'k3' not
# 'k2_7' in the -k -- the ids are disjoint so the two selectors cannot cross-match.
#
# 'scalar' is load-bearing, not cosmetic: run_device_perf profiles the whole -k selection into one
# CSV and the signpost filter keeps every MLA_START/MLA_END region, so a selector matching both the
# scalar and metadata cases sums TWO forwards. Measured 2026-08-04: K2.6 reads 13_947_233 unpinned
# vs 6_984_119 pinned, ratio 1.997. K3 skips metadata anyway (no runtime), so this is one forward.
#
# Pre-existing, not from the K3 work: test_kimi_mla_chunked_perf_galaxy above is FAILING for exactly
# this reason (13_947_233 vs a 7_118_649 baseline that matches one forward within 2%). The metadata
# axis came in with 3d3c65f985b (#51624), predating both K3 commits. Fix is to pin 'and scalar'
# there too and keep 7_118_649; left alone here so this change doesn't touch another CI baseline.
_CMD_K3_CHUNKED_8X4 = (
    f"pytest {_CHUNKED_TEST_PATH} " "-k 'deep-50k+5k and k3 and func and torus-xy-8x4 and scalar' --wrapper-invocation"
)


def _ci_unsupported_param_combos(**params):
    on_ci = params["is_ci_env"] or params["is_ci_v2_env"]

    if not on_ci:
        return False
    # Measures the non-chunked balanced MLA path; production runs chunked+non_balanced,
    # covered by the chunked galaxy perf tests below.
    return True


@pytest.mark.uncollect_if(pred=_ci_unsupported_param_combos)
@pytest.mark.timeout(0)
def test_deepseek_v3_mla_perf_loudbox():
    """Retain the existing 2x4 LoudBox proxy on unwrapped Fabric2D."""
    run_mla_perf_loudbox(
        command_2x4=_CMD_2X4,
        # Re-measured 2026-08-22 at 640 tokens/chip, BH LoudBox bh-lb-15, DDR 16000, 150W.
        # Mean of 14 runs, 2.658-2.664 ms, 0.25% peak to peak.
        expected_ns_2x4=2_660_615,
        model_name_2x4="deepseek_v3_mla_lb_2x4_fabric2d",
        subdir="deepseek_v3_mla",
        margin=0.03,
        comments_2x4="isl5k_lb_2x4_fabric2d_proxy",
    )


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_deepseek_v3_mla_perf_galaxy():
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        # Measured 2026-08-22, 14kW BH galaxy bh-glx-110-c04u02, 8x4 TorusXY certified, DDR 16000.
        # Two runs 3.894 / 3.886 ms, spread 0.21%.
        expected_device_perf_ns_per_iteration=3_890_333,
        subdir="deepseek_v3_mla",
        model_name="deepseek_v3_mla_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="isl5k_glx_8x4_ground_truth",
    )


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_kimi_mla_chunked_perf_galaxy():
    """Kimi K2.6 chunked-prefill MLA perf on the 8x4 Galaxy: 50k KV-cache prefix + one fresh 5k chunk
    (640 tokens/chip). Functional (no reference), so the single timed forward exercises the chunked
    640 matmul/SDPA configs end to end. Ground-truth 8x4 measurement (no 2x4 approximation)."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_CHUNKED_8X4,
        # Historical baseline: measured with unwrapped FABRIC_2D, not TorusXY.
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


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_kimi_k3_mla_chunked_perf_galaxy():
    """Kimi-K3 chunked-prefill MLA perf on the 8x4 Galaxy: 50k KV-cache prefix + one fresh 5k chunk
    (640 tokens/chip, H_loc=24). Same scenario/mesh as test_kimi_mla_chunked_perf_galaxy, but the two
    BASELINES are not comparable -- that one sums two forwards, this is one (see _CMD_K3_CHUNKED_8X4).

    Per forward, 2026-08-04: K3 11_562_468 vs K2.6 6_984_119 (+65.5%). That delta is NOT mostly the
    gate: RingJointSDPA accounts for +4_044_046 of the +4_576_289, with g_proj and its TP all-gather
    only ~10%. SDPA is 86% of the forward, and its utilisation drops 67.1% -> 59.6% going H_loc
    16 -> 24 (ideal work scales exactly 1.5x, measured 1.689x) -- 12.6% efficiency lost at identical
    shapes/grid/fidelity. Worth a targeted look before treating this baseline as the floor.

    This stays a ground-truth 8x4 measurement; a local 2x4 extrapolation would ignore g_proj and its
    all-gather and read optimistic.
    """
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_K3_CHUNKED_8X4,
        # Migration starting threshold: measured 2026-08-04 on bh-glx-b06u08 with unwrapped FABRIC_2D.
        # The first certified TorusXY result must be used to recalibrate it.
        expected_device_perf_ns_per_iteration=11_562_468,
        subdir="kimi_k3_mla",
        model_name="kimi_k3_mla_chunked_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        # Time only the forward: ops between MLA_START/MLA_END, excluding the one-time weight-load
        # tilize/typecast dispatched at construction.
        between_signposts=("MLA_START", "MLA_END"),
        comments="kimi_k3_chunked_50k+5k_glx_8x4_ground_truth",
    )
