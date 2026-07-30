# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_mla_perf_with_approximation,
    run_model_device_perf_test_with_merge,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_ds_mla"

_CMD_2X4 = f"pytest {_TEST_PATH} -k 'balanced-skip_check-seq100k-scaled_sl-random-line-2x4'"
_CMD_8X4 = f"pytest {_TEST_PATH} -k 'balanced-skip_check-seq100k-scaled_sl-random-line-8x4'"

# Kimi K2.6 chunked prefill: 50k KV-cache prefix + one fresh 5k chunk (chunk_size_global=5120). On
# the 8x4 Galaxy (sp=8) this lands chunk_local=640 per chip, exercising the num_heads=64 chunked-only
# 640 matmul/SDPA configs. Functional reference (no PCC) keeps the measured region to the single
# forward (the 50k prefix is preloaded host->device before the MLA_START signpost, so it is not timed).
_CHUNKED_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill"
_CMD_CHUNKED_8X4 = f"pytest {_CHUNKED_TEST_PATH} -k 'deep-50k+5k and kimi and func and 8x4 and fabric2d'"

# Kimi K3 chunked prefill (NoPE + output gate, 96 heads). chunk_size_global=1280 so a 2-SP box lands
# chunk_local=640 per chip -- the same per-device geometry the 8x4 Galaxy reaches at chunk 5120, and
# what the num_heads=96 chunked-only 640 configs are tuned for. 'k3' not 'kimi' in the -k: the ids are
# deliberately disjoint so this selector and the K2.6 one above cannot cross-match.
_CMD_K3_CHUNKED_2X4 = f"pytest {_CHUNKED_TEST_PATH} -k 'chunk1280-full and k3 and func and 2x4 and fabric2d'"


@pytest.mark.timeout(0)
def test_deepseek_v3_mla_perf_loudbox():
    """
    Measures perf on LB in 2x4 mesh shape, validates against its own perf baseline, and computes the approximate Galaxy perf.
    SDPA time is scaled by 4 (SP 2→8 = 4x, TP 4→4 = 1x) while other ops are added as-is.
    """
    run_mla_perf_with_approximation(
        command_2x4=_CMD_2X4,
        expected_ns_2x4=8_244_047,  # Recalibrated 2026-06-10 on BH LoudBox 2x4.
        model_name_2x4="deepseek_v3_mla_lb_2x4",
        subdir="deepseek_v3_mla",
        margin=0.03,
        comments_2x4="seq100k_scaled_lb_2x4_proxy",
    )


@pytest.mark.timeout(0)
def test_deepseek_v3_mla_perf_galaxy():
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4,
        expected_device_perf_ns_per_iteration=14_252_829,  # Recalibrated 2026-06-10 on bh-glx-110-c08u02; FABRIC_1D.
        subdir="deepseek_v3_mla",
        model_name="deepseek_v3_mla_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="seq100k_scaled_glx_8x4_ground_truth",
    )


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


@pytest.mark.timeout(0)
def test_kimi_k3_mla_chunked_perf_loudbox():
    """Kimi-K3 chunked-prefill MLA perf on a 2x4 Blackhole loudbox (SP2xTP4).

    Per-device geometry is identical to the 8x4 Galaxy's (H_loc=24, D_loc=1792, S_loc=640) -- only the
    ring size differs, 2 vs 8 -- so this exercises exactly the num_heads=96 tuned 640 matmul/SDPA
    configs. Functional reference (no PCC), so the timed region is a single forward.

    Deliberately NOT run through run_mla_perf_with_approximation: that helper predicts Galaxy by
    scaling only RingJointSDPADeviceOperation by 4x, and its TP_OPS/SDPA sets do not know about K3's
    new g_proj matmul or the all-gather feeding it, so the extrapolation would be optimistic. This
    asserts the 2x4 measurement on its own terms; a Galaxy ground-truth test needs a Galaxy run to
    calibrate (mirror test_kimi_mla_chunked_perf_galaxy with a 'chunk1280 and k3' or 5120-chunk -k).

    Measured breakdown at calibration: Matmul 1,443 us / CCL 2,013 us / SDPA 1,709 us / Other 812 us.
    CCL is the largest bucket, and the gate adds one TP all-gather -- worth watching if this regresses.

    Recalibrated after the kv_a_proj_with_mqa fix: K3 deliberately uses the untuned default for that
    one matmul because its tuned tiling degraded the KV cache enough to fail the 0.98 output PCC at
    depth (see mla_config.py and docs/KIMI_K3_MLA.md 5.1). That costs +6.3% on the matmul bucket and
    +1.7% overall (5,875,364 -> 5,976,584 ns) and buys passing 56320-token prefill instead of failing
    at 3840 -- do not "optimise" it back without re-running test_mla_chunked_prefill[k3-depth56k-1u].
    """
    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_K3_CHUNKED_2X4,
        expected_device_perf_ns_per_iteration=5_976_584,  # Recalibrated 2026-07-30 on BH LoudBox 2x4, FABRIC_2D.
        subdir="kimi_k3_mla",
        model_name="kimi_k3_mla_chunked_lb_2x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        # Time only the forward: ops between MLA_START/MLA_END, excluding the one-time weight-load
        # tilize/typecast dispatched at construction.
        between_signposts=("MLA_START", "MLA_END"),
        comments="kimi_k3_chunked_1280_lb_2x4_ground_truth",
    )
