# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
)
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power

# Kimi K2.7 chunked prefill: 50k cache + one 5k chunk. The 50k prefix is preloaded before the
# MLA_START signpost, so only the single forward is timed.
#
# Every axis must be pinned: run_device_perf profiles the whole -k selection into one CSV, and the
# merge fails unless that CSV holds exactly one MLA_START/MLA_END region.
# sp_only isolates the SDPA from the TP latent gather; the default tp_sharded layout is gated in traced replays by
# test_mla.py::test_mla_chunked_tp_sharded_traced_perf (overhead vs sp_only) and end to end by the Blaze prefill legs.
_CHUNKED_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill"
_CMD_CHUNKED_8X4 = (
    f"pytest {_CHUNKED_TEST_PATH} "
    "-k 'deep-50k+5k and k2_7 and func and torus-xy-8x4 and scalar and no_determinism and sp_only'"
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


# Kimi K3 (NoPE + output gate, 96 heads): same scenario/mesh as the K2.7 command above, pinned the same way. 'k3'
# not 'k2_7' in the -k -- the ids are disjoint so the two selectors cannot cross-match.
_CMD_K3_CHUNKED_8X4 = (
    f"pytest {_CHUNKED_TEST_PATH} "
    "-k 'deep-50k+5k and k3 and func and torus-xy-8x4 and scalar and no_determinism and sp_only'"
)

# Same 50k+5k scenario and mesh as the two Kimi rows, so the three are comparable per forward.
# RingJointSDPA is the only op in a Mistral4 layer that grows with KV depth, so it is the lever for
# long context.
_CMD_MISTRAL4_CHUNKED_8X4 = (
    f"pytest {_CHUNKED_TEST_PATH} "
    "-k 'deep-50k+5k and mistral4 and func and torus-xy-8x4 and scalar and no_determinism and sp_only'"
)
# Record-only: one local run on DDR 14000, so the margin admits any measurement. Replace both with the first certified
# TorusXY CI result.
_MISTRAL4_MLA_CHUNKED_NS_UNCALIBRATED = 3_301_775

# Measures the actual SP8/TP1 shape; approximate_mla_galaxy_perf pins TP=4 and cannot express it.
_CMD_MISTRAL4_CHUNKED_8X1 = (
    "pytest models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mistral4_mla_chunked_prefill_loudbox"
)
# CI LoudBox measurement (LoFi chunked ring SDPA). SDPA is most of this total and CCL is 0, so the
# row tracks the depth-proportional term almost alone.
_MISTRAL4_MLA_CHUNKED_LB_8X1_NS = 7_878_336


@pytest.mark.timeout(0)
def test_mistral4_mla_chunked_perf_loudbox():
    """Gate the unapproximated PP4-stage MLA budget on eight Blackhole devices.

    The reported number sums merged operation durations inside one forward, not elapsed
    end-to-end request latency. Gated only on a LoudBox: a Galaxy column is a different
    platform and keeps recording.
    """
    galaxy = _is_galaxy_env()
    visible_devices = os.environ.get("TT_VISIBLE_DEVICES", "").split(",")
    if galaxy and (len(visible_devices) != 8 or len(set(visible_devices)) != 8):
        pytest.skip("Expose one eight-device Galaxy ring with TT_VISIBLE_DEVICES, or use a Blackhole LoudBox")
    platform = "glx_column" if galaxy else "lb"
    run_model_device_perf_test_with_merge(
        command=_CMD_MISTRAL4_CHUNKED_8X1,
        expected_device_perf_ns_per_iteration=None if galaxy else _MISTRAL4_MLA_CHUNKED_LB_8X1_NS,
        margin=adjust_margin_for_ddr_speed(0.03),
        subdir="mistral4_mla",
        model_name=f"mistral4_mla_chunked_{platform}_8x1_torus_y",
        num_iterations=1,
        batch_size=1,
        between_signposts=("MLA_START", "MLA_END"),
        comments=f"mistral4_chunked_50k+5k_{platform}_8x1_torus_y",
    )


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_kimi_mla_chunked_perf_galaxy():
    """Kimi K2.7 chunked-prefill MLA perf on the 8x4 Galaxy: 50k KV-cache prefix + one fresh 5k chunk
    (640 tokens/chip). Functional (no reference), so the single timed forward exercises the chunked
    640 matmul/SDPA configs end to end. Ground-truth 8x4 measurement (no 2x4 approximation)."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_CHUNKED_8X4,
        # One forward, TorusXY Galaxy (DDR 14000), median of 3 runs.
        expected_device_perf_ns_per_iteration=4_088_524,
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
    (640 tokens/chip, H_loc=24). Same scenario/mesh as test_kimi_mla_chunked_perf_galaxy; both time one forward.

    This stays a ground-truth 8x4 measurement; a local 2x4 extrapolation would ignore g_proj and its
    all-gather and read optimistic.
    """
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_K3_CHUNKED_8X4,
        # One forward, TorusXY Galaxy (DDR 14000), median of 3 runs.
        expected_device_perf_ns_per_iteration=5_882_912,
        subdir="kimi_k3_mla",
        model_name="kimi_k3_mla_chunked_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("MLA_START", "MLA_END"),
        comments="kimi_k3_chunked_50k+5k_glx_8x4_ground_truth",
    )


@_REQUIRE_HIGH_POWER
@pytest.mark.timeout(0)
def test_mistral4_mla_chunked_perf_galaxy():
    """Mistral Small 4 chunked-prefill MLA perf on the 8x4 Galaxy: 50k KV-cache prefix + one fresh 5k
    chunk. Same scenario and mesh as the two Kimi rows, so the three are comparable per forward.

    RECORD-ONLY: the baseline is uncalibrated (see _MISTRAL4_MLA_CHUNKED_NS_UNCALIBRATED) and the
    margin admits any measurement, so this row publishes a number without gating on it. It exists to
    give the PP=4 op-level optimisation work a fast iteration target -- Tracy capture, pick the hot
    op, reproduce it here -- instead of iterating on the full model. Wire it into CI only once a
    certified TorusXY run has supplied a real threshold.
    """
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")
    _require_certified_torus_xy()

    run_model_device_perf_test_with_merge(
        command=_CMD_MISTRAL4_CHUNKED_8X4,
        # None skips check_device_perf entirely. A finite margin does not, however wide: it still
        # builds an upper bound, so a genuine speedup could fail the row.
        expected_device_perf_ns_per_iteration=None,
        subdir="mistral4_mla",
        model_name="mistral4_mla_chunked_glx_8x4",
        num_iterations=1,
        batch_size=1,
        between_signposts=("MLA_START", "MLA_END"),
        comments="mistral4_chunked_50k+5k_glx_8x4_record_only",
    )
