# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance test for the unified routed expert FFN op.

Perf counterpart to ``test_single_routed_expert_isl_sweep``: for kimi and glm
across the exhaustive ISL (active-token) sweep against the fixed 5K allocated
buffer, spawn one worker per (model, active) that runs ``run_single_routed_expert``
on device under Tracy, and assert the ``UnifiedRoutedExpertFfnDeviceOperation``
device time against a per-case baseline. One op per worker, so a regression
localizes to the FFN kernel.

The ROW_MAJOR x layout (``x_rm``) is measured — the Blackhole fused-tilize
production fast path (x tilized + bf8-packed inside the op, fresh output). BOTH
weight memory layouts are covered, DRAM-interleaved and DRAM ND-sharded, each with
its own baselines, so a regression in either is caught and the gap between them
stays visible (see ``_WEIGHTS_IDS``).

Baselines in ``_EXPECTED_NS`` were MEASURED LOCALLY on a BH board on 2026-08-04
and must be RECALIBRATED on the perf CI runner: device times are DDR-speed
dependent, so the canonical baselines have to come from the CI runner the check
actually runs on (mirrors the dated recalibration comments in the sibling
``test_moe_perf`` / ``test_dispatch_combine_perf``). One sample per case; see
``_margin_for`` for the observed run-to-run spread, which is ISL-dependent.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
)

# Device-op code emitted by the routed expert FFN; the harness sums the rows whose
# OP CODE contains this substring, so incidental setup ops are excluded.
_OP_CODE = "UnifiedRoutedExpertFfnDeviceOperation"

# Worker that runs the op on device (its signposts + the op launch land in the
# Tracy CSV the harness reads back).
_WORKER = (
    "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/"
    "test_single_routed_expert.py::test_single_routed_expert_isl_sweep"
)
# x input layouts, both measured.
#   x_rm   — ROW_MAJOR bf16 x, tilized and bf8-packed inside the op. The Blackhole
#            fused-tilize production fast path, and what the dispatch actually produces.
#   x_tile — x already TILE bfp8, consumed directly (no in-op tilize).
_LAYOUT_IDS = ("x_rm", "x_tile")

# Weight memory layouts the worker sweeps, both measured here. The ``-k`` filter MUST pin
# one of them: without it a single perf invocation runs both cases, the ops CSV holds two
# FFN rows, and the harness sums them into a ~2x "measured" value.
#
#   w_interleaved — DRAM-interleaved weights (page = tile). One NoC request per tile.
#   w_ndshard     — DRAM ND-sharded on a [TILE, per_core_N*TILE] shard, so a core's whole
#                   K-row weight slice is ONE request and consecutive K-rows rotate DRAM
#                   banks. Worth 1.05-1.23x at isl <= 512, where the fixed weight read is
#                   the critical path, and neutral (within +-3%) from isl 1024 up, where
#                   the x read and the matmuls dominate instead.
#
# The ND-shard advantage used to be 1.19-1.48x at isl <= 512. It shrank because
# in0_block_w_gu went 8 -> 16 (per_core_M_max 8 -> 4): both changes attack the SAME
# fixed per-K-block weight-read cost, so the second one to land collects less. The one
# cell where they conflict outright is x_rm/w_ndshard at isl-512, which regressed ~8%
# (196,139 ns, now marginally slower than interleaved) -- with per-tile requests a
# wider block changes nothing about the outstanding-request count, but with whole-K-row
# requests it doubles the reads in flight per block. Unexplained; the other 63 cells
# are neutral-or-better.
_WEIGHTS_IDS = ("w_interleaved", "w_ndshard")

# Per-(x layout, weights layout, model, active) UnifiedRoutedExpertFfnDeviceOperation
# device time in ns, measured on a Blackhole P150 (2026-08-04, card 0, single sample per
# case except the two x_rm/w_ndshard isl-512 cells, which are the median of three).
# Recalibrate on the perf CI runner — device times are HW/DDR-speed dependent.
_EXPECTED_NS: dict[tuple[str, str, str, int], int] = {
    # ---- x_rm, w_interleaved, kimi_k26 ----
    ("x_rm", "w_interleaved", "kimi_k26", 0): 3_974,
    ("x_rm", "w_interleaved", "kimi_k26", 64): 143_402,
    ("x_rm", "w_interleaved", "kimi_k26", 128): 146_822,
    ("x_rm", "w_interleaved", "kimi_k26", 256): 157_270,
    ("x_rm", "w_interleaved", "kimi_k26", 512): 202_118,  # median of 4: 191,997 / 201,655 / 202,582 / 209,754
    ("x_rm", "w_interleaved", "kimi_k26", 1024): 302_813,
    ("x_rm", "w_interleaved", "kimi_k26", 2048): 600_780,
    ("x_rm", "w_interleaved", "kimi_k26", 4096): 1_182_065,
    ("x_rm", "w_interleaved", "kimi_k26", 5120): 1_474_070,
    # ---- x_rm, w_interleaved, glm_51 ----
    ("x_rm", "w_interleaved", "glm_51", 0): 4_034,
    ("x_rm", "w_interleaved", "glm_51", 64): 131_728,
    ("x_rm", "w_interleaved", "glm_51", 128): 128_137,
    ("x_rm", "w_interleaved", "glm_51", 256): 134_068,
    ("x_rm", "w_interleaved", "glm_51", 512): 182_172,
    ("x_rm", "w_interleaved", "glm_51", 1024): 267_038,
    ("x_rm", "w_interleaved", "glm_51", 2048): 518_590,
    ("x_rm", "w_interleaved", "glm_51", 4096): 1_029_836,
    ("x_rm", "w_interleaved", "glm_51", 5120): 1_309_436,
    # ---- x_rm, w_ndshard, kimi_k26 ----
    ("x_rm", "w_ndshard", "kimi_k26", 0): 4_061,
    ("x_rm", "w_ndshard", "kimi_k26", 64): 124_270,
    ("x_rm", "w_ndshard", "kimi_k26", 128): 130_336,
    ("x_rm", "w_ndshard", "kimi_k26", 256): 148_230,
    ("x_rm", "w_ndshard", "kimi_k26", 512): 196_139,
    ("x_rm", "w_ndshard", "kimi_k26", 1024): 300_452,
    ("x_rm", "w_ndshard", "kimi_k26", 2048): 586_801,
    ("x_rm", "w_ndshard", "kimi_k26", 4096): 1_164_910,
    ("x_rm", "w_ndshard", "kimi_k26", 5120): 1_460_956,
    # ---- x_rm, w_ndshard, glm_51 ----
    ("x_rm", "w_ndshard", "glm_51", 0): 4_008,
    ("x_rm", "w_ndshard", "glm_51", 64): 111_154,
    ("x_rm", "w_ndshard", "glm_51", 128): 120_854,  # median of 4: 115,641 / 115,724 / 125,984 / 126,724
    ("x_rm", "w_ndshard", "glm_51", 256): 128_213,
    ("x_rm", "w_ndshard", "glm_51", 512): 170_537,
    ("x_rm", "w_ndshard", "glm_51", 1024): 263_096,
    ("x_rm", "w_ndshard", "glm_51", 2048): 516_244,
    ("x_rm", "w_ndshard", "glm_51", 4096): 1_021_266,
    ("x_rm", "w_ndshard", "glm_51", 5120): 1_268_990,
    # ---- x_tile, w_interleaved, kimi_k26 ----
    ("x_tile", "w_interleaved", "kimi_k26", 0): 4_141,
    ("x_tile", "w_interleaved", "kimi_k26", 64): 144_059,
    ("x_tile", "w_interleaved", "kimi_k26", 128): 146_428,
    ("x_tile", "w_interleaved", "kimi_k26", 256): 149_935,
    ("x_tile", "w_interleaved", "kimi_k26", 512): 175_210,
    ("x_tile", "w_interleaved", "kimi_k26", 1024): 274_818,
    ("x_tile", "w_interleaved", "kimi_k26", 2048): 538_877,
    ("x_tile", "w_interleaved", "kimi_k26", 4096): 1_059_844,
    ("x_tile", "w_interleaved", "kimi_k26", 5120): 1_327_143,
    # ---- x_tile, w_interleaved, glm_51 ----
    ("x_tile", "w_interleaved", "glm_51", 0): 4_108,
    ("x_tile", "w_interleaved", "glm_51", 64): 125_851,
    ("x_tile", "w_interleaved", "glm_51", 128): 128_503,
    ("x_tile", "w_interleaved", "glm_51", 256): 129_864,
    ("x_tile", "w_interleaved", "glm_51", 512): 150_180,
    ("x_tile", "w_interleaved", "glm_51", 1024): 240_994,
    ("x_tile", "w_interleaved", "glm_51", 2048): 473_384,
    ("x_tile", "w_interleaved", "glm_51", 4096): 927_045,
    ("x_tile", "w_interleaved", "glm_51", 5120): 1_181_216,
    # ---- x_tile, w_ndshard, kimi_k26 ----
    ("x_tile", "w_ndshard", "kimi_k26", 0): 4_005,
    ("x_tile", "w_ndshard", "kimi_k26", 64): 119_066,  # median of 5: 116,118 / 116,742 / 119,066 / 122,199 / 135,516
    ("x_tile", "w_ndshard", "kimi_k26", 128): 134_356,
    ("x_tile", "w_ndshard", "kimi_k26", 256): 141_341,
    ("x_tile", "w_ndshard", "kimi_k26", 512): 163_760,  # median of 4: 157,821 / 162,901 / 164,619 / 175,358
    ("x_tile", "w_ndshard", "kimi_k26", 1024): 271_916,
    ("x_tile", "w_ndshard", "kimi_k26", 2048): 534_853,
    ("x_tile", "w_ndshard", "kimi_k26", 4096): 1_059_513,
    ("x_tile", "w_ndshard", "kimi_k26", 5120): 1_315_653,
    # ---- x_tile, w_ndshard, glm_51 ----
    ("x_tile", "w_ndshard", "glm_51", 0): 4_077,
    ("x_tile", "w_ndshard", "glm_51", 64): 106_466,
    ("x_tile", "w_ndshard", "glm_51", 128): 119_309,
    ("x_tile", "w_ndshard", "glm_51", 256): 129_819,  # median of 4: 116,797 / 124,962 / 134,676 / 134,843
    ("x_tile", "w_ndshard", "glm_51", 512): 145_490,  # median of 4: 141,610 / 142,120 / 148,861 / 158,885
    ("x_tile", "w_ndshard", "glm_51", 1024): 237_970,
    ("x_tile", "w_ndshard", "glm_51", 2048): 468_583,
    ("x_tile", "w_ndshard", "glm_51", 4096): 922_356,
    ("x_tile", "w_ndshard", "glm_51", 5120): 1_146_731,
}

# Wider than the usual 3%, and ISL-dependent because the spread is structurally
# ISL-dependent. Across two full 72-case runs of the SAME build on the same card, the
# run-to-run ratio had a median of 1.004 — no systematic drift — but the spread split
# sharply by ISL: within 4.4% for isl >= 1024, out to 15.5% for isl <= 512.
#
# The reason is chunk count. At isl <= 512 the op runs a SINGLE chunk, so one stalled
# weight-read round lands whole in the measurement with nothing to average it against;
# from isl 1024 up there are several chunks and the noise averages out. Resampling the
# outliers four times each shows it is BIMODAL, not drifting — e.g. w_ndshard glm isl-128
# gave 115,641 / 115,724 / 125,984 / 126,724 ns, two clusters ~11 us apart, and four of
# the five outliers were high by a near-identical +17-18 us regardless of model or ISL.
# That signature (a fixed additive step, not a proportional one) is what a single
# occasional stall looks like.
#
# So: 15% at isl <= 512 covers the observed bimodality, and 8% above it where the
# measurement is genuinely tighter. Cases with more than one observation are centred on
# the median of their samples. Tightening either band needs multi-iteration averaging in
# the harness, not a narrower band on a single sample.
_MARGIN_SHORT_ISL = 0.15
_MARGIN_LONG_ISL = 0.08
# isl at or below which the single-chunk regime (and its wider spread) applies.
_SHORT_ISL_MAX = 512


def _margin_for(active: int) -> float:
    return _MARGIN_SHORT_ISL if active <= _SHORT_ISL_MAX else _MARGIN_LONG_ISL


def _k_filter(model: str, active: int, x_layout: str, weights: str) -> str:
    """Pin exactly one ``test_single_routed_expert_isl_sweep`` case: model + isl + x
    layout + weight layout. Disambiguate substring collisions in the pytest ``-k`` match
    — e.g. ``isl-512`` is a substring of ``isl-5120`` — by excluding any other sweep value
    whose id contains this one."""
    parts = [f"{model}-isl-{active}", x_layout, weights]
    parts += [
        f"not isl-{other}" for other in _ISL_EXHAUSTIVE_SWEEP if other != active and f"isl-{active}" in f"isl-{other}"
    ]
    return " and ".join(parts)


def _perf_params():
    params = []
    for x_layout in _LAYOUT_IDS:
        for weights in _WEIGHTS_IDS:
            for model in _ISL_EXHAUSTIVE_MODELS:
                for active in _ISL_EXHAUSTIVE_SWEEP:
                    command = f"pytest {_WORKER} -k '{_k_filter(model, active, x_layout, weights)}'"
                    params.append(
                        pytest.param(
                            command,
                            {_OP_CODE: _EXPECTED_NS[(x_layout, weights, model, active)]},
                            f"single_routed_expert_{model}_isl{active}_{x_layout}_{weights}",
                            _margin_for(active),
                            id=f"{x_layout}-{weights}-{model}-isl-{active}",
                        )
                    )
    return params


@pytest.mark.parametrize("command, expected_per_op, model_name, margin", _perf_params())
@pytest.mark.models_device_performance_bare_metal
# Gate to P150 via tt-smi board telemetry (SMBus). This also skips Wormhole and any
# other board. Do NOT use ttnn.cluster.get_cluster_type() here: it opens and locks the
# chip, and since skipif is evaluated at collection time in the parent process, the
# spawned Tracy worker then deadlocks on CHIP_IN_USE. is_p150() reads tt-smi only, so
# it takes no device lock.
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
def test_single_routed_expert_perf(command, expected_per_op, model_name, margin):
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op=expected_per_op,
        subdir="prefill_single_routed_expert",
        model_name=model_name,
        margin=margin,
        comments=model_name,
    )
