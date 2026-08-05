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

Baselines in ``_EXPECTED_NS`` were MEASURED LOCALLY on a BH board on 2026-08-05
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
# device time in ns, measured on a Blackhole P150 (2026-08-05, card 0, single sample per
# case). Re-measured after the DOWN_SPLIT two-RISC down-weight read, which is worth
# 1.06-1.18x at isl <= 512 on DRAM-interleaved weights (both x layouts) and is neutral
# from isl 1024 up and on ND-sharded weights.
# Recalibrate on the perf CI runner — device times are HW/DDR-speed dependent.
_EXPECTED_NS: dict[tuple[str, str, str, int], int] = {
    # ---- x_rm, w_interleaved, kimi_k26 ----
    ("x_rm", "w_interleaved", "kimi_k26", 0): 3_938,
    ("x_rm", "w_interleaved", "kimi_k26", 64): 126_664,
    ("x_rm", "w_interleaved", "kimi_k26", 128): 130_063,
    ("x_rm", "w_interleaved", "kimi_k26", 256): 136_740,
    ("x_rm", "w_interleaved", "kimi_k26", 512): 191_253,
    ("x_rm", "w_interleaved", "kimi_k26", 1024): 318_801,
    ("x_rm", "w_interleaved", "kimi_k26", 2048): 596_930,
    ("x_rm", "w_interleaved", "kimi_k26", 4096): 1_177_078,
    ("x_rm", "w_interleaved", "kimi_k26", 5120): 1_477_543,
    # ---- x_rm, w_interleaved, glm_51 ----
    ("x_rm", "w_interleaved", "glm_51", 0): 3_948,
    ("x_rm", "w_interleaved", "glm_51", 64): 112_002,
    ("x_rm", "w_interleaved", "glm_51", 128): 113_821,
    ("x_rm", "w_interleaved", "glm_51", 256): 120_488,
    ("x_rm", "w_interleaved", "glm_51", 512): 177_421,
    ("x_rm", "w_interleaved", "glm_51", 1024): 267_296,
    ("x_rm", "w_interleaved", "glm_51", 2048): 517_673,
    ("x_rm", "w_interleaved", "glm_51", 4096): 1_082_301,
    ("x_rm", "w_interleaved", "glm_51", 5120): 1_281_589,
    # ---- x_rm, w_ndshard, kimi_k26 ----
    ("x_rm", "w_ndshard", "kimi_k26", 0): 3_902,
    ("x_rm", "w_ndshard", "kimi_k26", 64): 116_570,
    ("x_rm", "w_ndshard", "kimi_k26", 128): 127_144,
    ("x_rm", "w_ndshard", "kimi_k26", 256): 146_735,
    ("x_rm", "w_ndshard", "kimi_k26", 512): 180_393,
    ("x_rm", "w_ndshard", "kimi_k26", 1024): 300_025,
    ("x_rm", "w_ndshard", "kimi_k26", 2048): 589_709,
    ("x_rm", "w_ndshard", "kimi_k26", 4096): 1_170_085,
    ("x_rm", "w_ndshard", "kimi_k26", 5120): 1_455_277,
    # ---- x_rm, w_ndshard, glm_51 ----
    ("x_rm", "w_ndshard", "glm_51", 0): 4_133,
    ("x_rm", "w_ndshard", "glm_51", 64): 104_253,
    ("x_rm", "w_ndshard", "glm_51", 128): 111_343,
    ("x_rm", "w_ndshard", "glm_51", 256): 125_372,
    ("x_rm", "w_ndshard", "glm_51", 512): 157_579,
    ("x_rm", "w_ndshard", "glm_51", 1024): 266_990,
    ("x_rm", "w_ndshard", "glm_51", 2048): 513_814,
    ("x_rm", "w_ndshard", "glm_51", 4096): 1_028_036,
    ("x_rm", "w_ndshard", "glm_51", 5120): 1_273_404,
    # ---- x_tile, w_interleaved, kimi_k26 ----
    ("x_tile", "w_interleaved", "kimi_k26", 0): 4_088,
    ("x_tile", "w_interleaved", "kimi_k26", 64): 127_379,
    ("x_tile", "w_interleaved", "kimi_k26", 128): 132_069,
    ("x_tile", "w_interleaved", "kimi_k26", 256): 134_367,
    ("x_tile", "w_interleaved", "kimi_k26", 512): 156_947,
    ("x_tile", "w_interleaved", "kimi_k26", 1024): 275_946,
    ("x_tile", "w_interleaved", "kimi_k26", 2048): 535_527,
    ("x_tile", "w_interleaved", "kimi_k26", 4096): 1_058_557,
    ("x_tile", "w_interleaved", "kimi_k26", 5120): 1_325_944,
    # ---- x_tile, w_interleaved, glm_51 ----
    ("x_tile", "w_interleaved", "glm_51", 0): 4_041,
    ("x_tile", "w_interleaved", "glm_51", 64): 113_427,
    ("x_tile", "w_interleaved", "glm_51", 128): 120_631,
    ("x_tile", "w_interleaved", "glm_51", 256): 116_389,
    ("x_tile", "w_interleaved", "glm_51", 512): 137_347,
    ("x_tile", "w_interleaved", "glm_51", 1024): 239_600,
    ("x_tile", "w_interleaved", "glm_51", 2048): 467_721,
    ("x_tile", "w_interleaved", "glm_51", 4096): 930_212,
    ("x_tile", "w_interleaved", "glm_51", 5120): 1_169_606,
    # ---- x_tile, w_ndshard, kimi_k26 ----
    ("x_tile", "w_ndshard", "kimi_k26", 0): 4_124,
    ("x_tile", "w_ndshard", "kimi_k26", 64): 114_107,
    ("x_tile", "w_ndshard", "kimi_k26", 128): 128_243,
    ("x_tile", "w_ndshard", "kimi_k26", 256): 125_615,
    ("x_tile", "w_ndshard", "kimi_k26", 512): 166_313,
    ("x_tile", "w_ndshard", "kimi_k26", 1024): 275_992,
    ("x_tile", "w_ndshard", "kimi_k26", 2048): 533_737,
    ("x_tile", "w_ndshard", "kimi_k26", 4096): 1_059_316,
    ("x_tile", "w_ndshard", "kimi_k26", 5120): 1_315_324,
    # ---- x_tile, w_ndshard, glm_51 ----
    ("x_tile", "w_ndshard", "glm_51", 0): 3_913,
    ("x_tile", "w_ndshard", "glm_51", 64): 102_853,
    ("x_tile", "w_ndshard", "glm_51", 128): 115_464,
    ("x_tile", "w_ndshard", "glm_51", 256): 112_594,
    ("x_tile", "w_ndshard", "glm_51", 512): 141_131,
    ("x_tile", "w_ndshard", "glm_51", 1024): 245_976,
    ("x_tile", "w_ndshard", "glm_51", 2048): 466_882,
    ("x_tile", "w_ndshard", "glm_51", 4096): 922_344,
    ("x_tile", "w_ndshard", "glm_51", 5120): 1_154_307,
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
