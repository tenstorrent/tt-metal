# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
MoE device-perf tests approximating glx 8x4 from LB (8-chip) proxies.

- `test_deepseek_v3_moe_perf_loudbox`: runs 8x1 and 2x4 proxies once each, validates
  each against its own perf baseline, and in the same pass computes the approximate
  8x4 galaxy total (SP ops from 8x1 + TP ops from 2x4). One test, two device runs,
  three signals: per-proxy perf regression catches + approximation artifact.
- `test_deepseek_v3_moe_perf_galaxy`: 8x4 ground truth (skipped off-glx).

Sum of per-op approximation approximates one glx column's MoE block kernel time;
the 8x4 ground-truth test is the reference the approximation is compared against.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import (
    _is_galaxy_env,
    adjust_margin_for_ddr_speed,
    run_model_device_perf_test_with_merge,
    run_moe_perf_with_approximation,
)

_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ds_moe"

# `and pad0` pins the padding parametrize (test_ttnn_moe.py adds pad0/pad50 ids) so each
# command still selects exactly one case; pad0 keeps the no-padding baselines below valid.
_CMD_8X1 = f"pytest {_TEST_PATH} -k 'perf-host-64 and linear-8 and pad0'"
# `not fabric2d-` excludes the new FABRIC_2D parametrize ids in test_ttnn_moe.py (substring `mesh-2x4`/`mesh-8x4` would otherwise match).
_CMD_2X4 = f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-2x4 and not linear-8 and not mesh-4x2 and not mesh-8x4 and not fabric2d- and pad0'"
_CMD_8X4_pad0 = f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-8x4 and not linear-8 and not mesh-4x2 and not mesh-2x4 and not fabric2d- and pad0'"
_CMD_8X4_pad50 = f"pytest {_TEST_PATH} -k 'perf-device-256 and mesh-8x4 and not linear-8 and not mesh-4x2 and not mesh-2x4 and not fabric2d- and pad50'"


@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_loudbox():
    """
    Run 8x1 + 2x4 proxies once each on loudbox (BH-LoudBox, 8xP150).
    Validates each proxy against its own baseline AND computes the approximate
    8x4 total from the same two CSVs (no extra device work).
    """
    run_moe_perf_with_approximation(
        command_8x1=_CMD_8X1,
        # Recalibrated 2026-07-27 on BH LoudBox 8x1 after routed expert optimization with removing prezeroing
        # Was 15_506_174.
        expected_ns_8x1=14_549_108,
        model_name_8x1="deepseek_v3_moe_lb_8x1_dispatch_combine",
        command_2x4=_CMD_2X4,
        # Recalibrated 2026-07-27 on BH LoudBox 2x4 for. Was 23_956_009.
        expected_ns_2x4=15_954_784,
        model_name_2x4="deepseek_v3_moe_lb_2x4_gate",
        subdir="deepseek_v3_moe",
        margin=0.03,
        comments_8x1="seq3200_lb_8x1_dispatch_combine_proxy",
        comments_2x4="seq3200_lb_2x4_gate_proxy",
    )


@pytest.mark.timeout(0)
def test_deepseek_v3_moe_perf_galaxy():
    """8x4 galaxy ground truth — the reference the loudbox approximation targets."""
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_8X4_pad0,
        expected_device_perf_ns_per_iteration=21_028_751,  # Recalibrated 2026-07-26
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
        expected_device_perf_ns_per_iteration=14_107_228,  # Recalibrated 2026-07-27 (perf improvement, was 15_719_590).
        subdir="deepseek_v3_moe",
        model_name="deepseek_v3_moe_glx_8x4_pad50",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        comments="seq3200_glx_8x4_ground_truth_padded_50_percent_w_awareness",
    )


# --- Kimi-K3 LatentMoE ---------------------------------------------------------------------------
# Own command constant rather than a variant of the DeepSeek ones: K3 has its own test function
# (test_kimi_k3_moe), and its ids are already model-scoped, so the long "not linear-8 and not
# mesh-4x2 ..." exclusion chain the shared test_ds_moe ids need is unnecessary here.
_K3_TEST_PATH = "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_kimi_k3_moe"
_CMD_K3_8X4 = f"pytest {_K3_TEST_PATH} -k 'fabric2d-mesh-8x4 and kimi_k3-5k-perf'"


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
    if not _is_galaxy_env():
        pytest.skip("This test requires 8x4 mesh - galaxy. (set MESH_DEVICE=TG)")

    margin = adjust_margin_for_ddr_speed(0.03)

    run_model_device_perf_test_with_merge(
        command=_CMD_K3_8X4,
        # Measured 2026-08-07 on bh-glx-120-c04u02 (8x4, DDR 14000 -- sub-nominal, so
        # adjust_margin_for_ddr_speed doubles the 3% margin to 6%). Mean of two consecutive runs,
        # 12_922_557 and 12_927_147, which agree to 0.036%. Signpost-filtered and device-row-merged.
        # Split at this shape: Matmul 2_196 us, CCL 1_042 us, Other 9_687 us -- the block is
        # dominated by the dispatch/combine/top-k path, not by the latent projections.
        #
        # Superseded the earlier 28_872_936, which was 3200 tokens/chip on FABRIC_1D. Dropping to
        # the 5K production shape cut the sequence 5x but the time only 2.2x, because the per-block
        # fixed costs and the CCLs do not scale with sequence.
        expected_device_perf_ns_per_iteration=12_924_852,
        subdir="kimi_k3_moe",
        model_name="kimi_k3_moe_glx_8x4",
        num_iterations=1,
        batch_size=1,
        margin=margin,
        between_signposts=("MoE_START", "MoE_END"),
        comments="seq640_5k_isl_glx_8x4_fabric2d_ground_truth_latent_moe_silu",
    )
