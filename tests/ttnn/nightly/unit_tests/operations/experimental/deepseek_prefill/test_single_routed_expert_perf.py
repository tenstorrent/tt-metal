# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance test for the routed-expert FFN implementations.

Perf counterpart to ``test_single_routed_expert_isl_sweep``: for the selected
model and benchmark shapes across the exhaustive ISL (active-token) sweep
against the fixed 5K allocated buffer, spawn one worker per
(implementation, model, active) that runs
``run_single_routed_expert`` on device under Tracy, and assert the selected
device operation's time against a per-case baseline. One FFN op per worker, so
a regression localizes to the selected kernel.

Both activation paths are measured: ROW_MAJOR BF16 (``x_rm``), which exercises
the fused tilize/BFP8-pack path, and pre-tilized BFP8 (``x_tile``). Both weight
memory layouts are covered too—DRAM-interleaved and DRAM ND-sharded—each with
its own baselines.

The unified baselines were measured on 2026-08-05; the moe_fused baselines were
measured on 2026-08-07. They must be recalibrated on the perf CI runner because
device times are DDR-speed dependent. One sample per case; see ``_margin_for``
for the ISL-dependent run-to-run allowance.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
)

# Device-op code emitted by each implementation. The harness selects only this
# row, so weight conversion and other setup operations are excluded.
_OP_CODES = {
    "unified": "UnifiedRoutedExpertFfnDeviceOperation",
    "moe_fused": "MoeFusedSwiGluDeviceOperation",
}

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
# case). Re-measured after three changes landed together:
#   * DOWN_SPLIT      two-RISC down-weight read (1.06-1.18x at isl <= 512 on interleaved
#                     weights, both x layouts; neutral above and on ND-sharded)
#   * IN1_WRITER_MCAST the writer runs the gate/up multicast on its own NoC 1 while the
#                     reader keeps reading on NoC 0 (~6 us/call at isl <= 256, and it
#                     collapses the isl-512 bimodality from ~7.5% spread to ~0.5%)
#   * COUNTS_BCAST    one core reads counts/idx and multicasts them, instead of all 88
#                     cores hitting the same two DRAM pages (1.59 -> 0.93 us, fixed cost
#                     on every call, so worth ~3 us at isl-128 and ~9 us at isl-2048)
#
# Reference budget the work was steered by, x_rm only (the dispatch emits ROW_MAJOR):
# 10 x isl-128 + 2 x isl-2048 = 2146.5 us on ND-sharded weights, 2399.7 us on interleaved.
# Recalibrate on the perf CI runner — device times are HW/DDR-speed dependent.
_UNIFIED_EXPECTED_NS: dict[tuple[str, str, str, int], int] = {
    # ---- x_rm, w_interleaved, kimi_k26 ----
    ("x_rm", "w_interleaved", "kimi_k26", 0): 2_884,
    ("x_rm", "w_interleaved", "kimi_k26", 64): 121_366,
    ("x_rm", "w_interleaved", "kimi_k26", 128): 122_925,
    ("x_rm", "w_interleaved", "kimi_k26", 256): 127_024,
    ("x_rm", "w_interleaved", "kimi_k26", 512): 162_893,
    ("x_rm", "w_interleaved", "kimi_k26", 1024): 298_172,
    ("x_rm", "w_interleaved", "kimi_k26", 2048): 585_216,
    ("x_rm", "w_interleaved", "kimi_k26", 4096): 1_163_201,
    ("x_rm", "w_interleaved", "kimi_k26", 5120): 1_454_667,
    # ---- x_rm, w_interleaved, glm_51 ----
    ("x_rm", "w_interleaved", "glm_51", 0): 2_803,
    ("x_rm", "w_interleaved", "glm_51", 64): 106_450,
    ("x_rm", "w_interleaved", "glm_51", 128): 108_279,
    ("x_rm", "w_interleaved", "glm_51", 256): 111_831,
    ("x_rm", "w_interleaved", "glm_51", 512): 144_427,
    ("x_rm", "w_interleaved", "glm_51", 1024): 261_148,
    ("x_rm", "w_interleaved", "glm_51", 2048): 512_956,
    ("x_rm", "w_interleaved", "glm_51", 4096): 1_017_936,
    ("x_rm", "w_interleaved", "glm_51", 5120): 1_267_576,
    # ---- x_rm, w_interleaved, k7168_n3072 ----
    ("x_rm", "w_interleaved", "k7168_n3072", 0): 2_736,
    ("x_rm", "w_interleaved", "k7168_n3072", 64): 167_340,
    ("x_rm", "w_interleaved", "k7168_n3072", 128): 169_310,
    ("x_rm", "w_interleaved", "k7168_n3072", 256): 172_609,
    ("x_rm", "w_interleaved", "k7168_n3072", 512): 220_710,
    ("x_rm", "w_interleaved", "k7168_n3072", 1024): 417_367,
    ("x_rm", "w_interleaved", "k7168_n3072", 2048): 820_121,
    ("x_rm", "w_interleaved", "k7168_n3072", 4096): 1_635_696,
    ("x_rm", "w_interleaved", "k7168_n3072", 5120): 2_055_499,
    # ---- x_rm, w_ndshard, kimi_k26 ----
    ("x_rm", "w_ndshard", "kimi_k26", 0): 2_975,
    ("x_rm", "w_ndshard", "kimi_k26", 64): 94_900,
    ("x_rm", "w_ndshard", "kimi_k26", 128): 97_532,
    ("x_rm", "w_ndshard", "kimi_k26", 256): 110_242,
    ("x_rm", "w_ndshard", "kimi_k26", 512): 157_987,
    ("x_rm", "w_ndshard", "kimi_k26", 1024): 297_718,
    ("x_rm", "w_ndshard", "kimi_k26", 2048): 585_567,
    ("x_rm", "w_ndshard", "kimi_k26", 4096): 1_158_800,
    ("x_rm", "w_ndshard", "kimi_k26", 5120): 1_448_190,
    # ---- x_rm, w_ndshard, glm_51 ----
    ("x_rm", "w_ndshard", "glm_51", 0): 2_973,
    ("x_rm", "w_ndshard", "glm_51", 64): 85_815,
    ("x_rm", "w_ndshard", "glm_51", 128): 85_939,
    ("x_rm", "w_ndshard", "glm_51", 256): 97_253,
    ("x_rm", "w_ndshard", "glm_51", 512): 139_347,
    ("x_rm", "w_ndshard", "glm_51", 1024): 260_572,
    ("x_rm", "w_ndshard", "glm_51", 2048): 515_509,
    ("x_rm", "w_ndshard", "glm_51", 4096): 1_014_213,
    ("x_rm", "w_ndshard", "glm_51", 5120): 1_266_821,
    # ---- x_rm, w_ndshard, k7168_n3072 ----
    ("x_rm", "w_ndshard", "k7168_n3072", 0): 2_743,
    ("x_rm", "w_ndshard", "k7168_n3072", 64): 130_468,
    ("x_rm", "w_ndshard", "k7168_n3072", 128): 134_737,
    ("x_rm", "w_ndshard", "k7168_n3072", 256): 143_672,
    ("x_rm", "w_ndshard", "k7168_n3072", 512): 217_702,
    ("x_rm", "w_ndshard", "k7168_n3072", 1024): 416_345,
    ("x_rm", "w_ndshard", "k7168_n3072", 2048): 823_035,
    ("x_rm", "w_ndshard", "k7168_n3072", 4096): 1_636_924,
    ("x_rm", "w_ndshard", "k7168_n3072", 5120): 2_062_684,
    # ---- x_tile, w_interleaved, kimi_k26 ----
    ("x_tile", "w_interleaved", "kimi_k26", 0): 2_971,
    ("x_tile", "w_interleaved", "kimi_k26", 64): 121_399,
    ("x_tile", "w_interleaved", "kimi_k26", 128): 123_069,
    ("x_tile", "w_interleaved", "kimi_k26", 256): 126_712,
    ("x_tile", "w_interleaved", "kimi_k26", 512): 145_491,
    ("x_tile", "w_interleaved", "kimi_k26", 1024): 269_984,
    ("x_tile", "w_interleaved", "kimi_k26", 2048): 531_124,
    ("x_tile", "w_interleaved", "kimi_k26", 4096): 1_054_732,
    ("x_tile", "w_interleaved", "kimi_k26", 5120): 1_315_377,
    # ---- x_tile, w_interleaved, glm_51 ----
    ("x_tile", "w_interleaved", "glm_51", 0): 2_886,
    ("x_tile", "w_interleaved", "glm_51", 64): 106_844,
    ("x_tile", "w_interleaved", "glm_51", 128): 108_238,
    ("x_tile", "w_interleaved", "glm_51", 256): 111_109,
    ("x_tile", "w_interleaved", "glm_51", 512): 129_564,
    ("x_tile", "w_interleaved", "glm_51", 1024): 237_256,
    ("x_tile", "w_interleaved", "glm_51", 2048): 467_307,
    ("x_tile", "w_interleaved", "glm_51", 4096): 923_307,
    ("x_tile", "w_interleaved", "glm_51", 5120): 1_150_249,
    # ---- x_tile, w_ndshard, kimi_k26 ----
    ("x_tile", "w_ndshard", "kimi_k26", 0): 2_899,
    ("x_tile", "w_ndshard", "kimi_k26", 64): 92_016,
    ("x_tile", "w_ndshard", "kimi_k26", 128): 95_773,
    ("x_tile", "w_ndshard", "kimi_k26", 256): 102_804,
    ("x_tile", "w_ndshard", "kimi_k26", 512): 141_113,
    ("x_tile", "w_ndshard", "kimi_k26", 1024): 268_513,
    ("x_tile", "w_ndshard", "kimi_k26", 2048): 530_594,
    ("x_tile", "w_ndshard", "kimi_k26", 4096): 1_053_151,
    ("x_tile", "w_ndshard", "kimi_k26", 5120): 1_306_567,
    # ---- x_tile, w_ndshard, glm_51 ----
    ("x_tile", "w_ndshard", "glm_51", 0): 2_802,
    ("x_tile", "w_ndshard", "glm_51", 64): 83_205,
    ("x_tile", "w_ndshard", "glm_51", 128): 85_717,
    ("x_tile", "w_ndshard", "glm_51", 256): 90_936,
    ("x_tile", "w_ndshard", "glm_51", 512): 124_310,
    ("x_tile", "w_ndshard", "glm_51", 1024): 235_378,
    ("x_tile", "w_ndshard", "glm_51", 2048): 463_863,
    ("x_tile", "w_ndshard", "glm_51", 4096): 917_145,
    ("x_tile", "w_ndshard", "glm_51", 5120): 1_147_513,
}

# Same worker, shapes, 5120-row allocation, active-token sweep, input layouts,
# BFP4 weights, and 11x8 grid, measured for moe_fused_swiglu on 2026-08-07.
# Its W_down partition differs from unified, so w_ndshard uses the equivalent
# implementation-native contiguous K-row slice rather than unified's shard width.
_MOE_FUSED_EXPECTED_NS: dict[tuple[str, str, str, int], int] = {
    # ---- x_rm, w_interleaved, kimi_k26 ----
    ("x_rm", "w_interleaved", "kimi_k26", 0): 3_926,
    ("x_rm", "w_interleaved", "kimi_k26", 64): 89_079,
    ("x_rm", "w_interleaved", "kimi_k26", 128): 98_631,
    ("x_rm", "w_interleaved", "kimi_k26", 256): 115_702,
    ("x_rm", "w_interleaved", "kimi_k26", 512): 194_271,
    ("x_rm", "w_interleaved", "kimi_k26", 1024): 344_786,
    ("x_rm", "w_interleaved", "kimi_k26", 2048): 631_608,
    ("x_rm", "w_interleaved", "kimi_k26", 4096): 1_214_416,
    ("x_rm", "w_interleaved", "kimi_k26", 5120): 1_506_822,
    # ---- x_rm, w_interleaved, glm_51 ----
    ("x_rm", "w_interleaved", "glm_51", 0): 3_999,
    ("x_rm", "w_interleaved", "glm_51", 64): 79_493,
    ("x_rm", "w_interleaved", "glm_51", 128): 85_227,
    ("x_rm", "w_interleaved", "glm_51", 256): 106_544,
    ("x_rm", "w_interleaved", "glm_51", 512): 181_190,
    ("x_rm", "w_interleaved", "glm_51", 1024): 316_261,
    ("x_rm", "w_interleaved", "glm_51", 2048): 589_530,
    ("x_rm", "w_interleaved", "glm_51", 4096): 1_129_096,
    ("x_rm", "w_interleaved", "glm_51", 5120): 1_402_513,
    # ---- x_rm, w_interleaved, k7168_n3072 ----
    ("x_rm", "w_interleaved", "k7168_n3072", 0): 3_976,
    ("x_rm", "w_interleaved", "k7168_n3072", 64): 133_530,
    ("x_rm", "w_interleaved", "k7168_n3072", 128): 145_590,
    ("x_rm", "w_interleaved", "k7168_n3072", 256): 185_916,
    ("x_rm", "w_interleaved", "k7168_n3072", 512): 323_599,
    ("x_rm", "w_interleaved", "k7168_n3072", 1024): 581_733,
    ("x_rm", "w_interleaved", "k7168_n3072", 2048): 1_093_273,
    ("x_rm", "w_interleaved", "k7168_n3072", 4096): 2_124_113,
    ("x_rm", "w_interleaved", "k7168_n3072", 5120): 2_636_876,
    # ---- x_rm, w_ndshard, kimi_k26 ----
    ("x_rm", "w_ndshard", "kimi_k26", 0): 3_972,
    ("x_rm", "w_ndshard", "kimi_k26", 64): 79_874,
    ("x_rm", "w_ndshard", "kimi_k26", 128): 84_888,
    ("x_rm", "w_ndshard", "kimi_k26", 256): 109_488,
    ("x_rm", "w_ndshard", "kimi_k26", 512): 189_713,
    ("x_rm", "w_ndshard", "kimi_k26", 1024): 330_583,
    ("x_rm", "w_ndshard", "kimi_k26", 2048): 629_869,
    ("x_rm", "w_ndshard", "kimi_k26", 4096): 1_212_263,
    ("x_rm", "w_ndshard", "kimi_k26", 5120): 1_504_913,
    # ---- x_rm, w_ndshard, glm_51 ----
    ("x_rm", "w_ndshard", "glm_51", 0): 3_932,
    ("x_rm", "w_ndshard", "glm_51", 64): 71_702,
    ("x_rm", "w_ndshard", "glm_51", 128): 79_238,
    ("x_rm", "w_ndshard", "glm_51", 256): 102_153,
    ("x_rm", "w_ndshard", "glm_51", 512): 169_771,
    ("x_rm", "w_ndshard", "glm_51", 1024): 307_511,
    ("x_rm", "w_ndshard", "glm_51", 2048): 578_576,
    ("x_rm", "w_ndshard", "glm_51", 4096): 1_120_636,
    ("x_rm", "w_ndshard", "glm_51", 5120): 1_391_755,
    # ---- x_rm, w_ndshard, k7168_n3072 ----
    ("x_rm", "w_ndshard", "k7168_n3072", 0): 4_031,
    ("x_rm", "w_ndshard", "k7168_n3072", 64): 110_233,
    ("x_rm", "w_ndshard", "k7168_n3072", 128): 131_671,
    ("x_rm", "w_ndshard", "k7168_n3072", 256): 177_799,
    ("x_rm", "w_ndshard", "k7168_n3072", 512): 302_147,
    ("x_rm", "w_ndshard", "k7168_n3072", 1024): 544_490,
    ("x_rm", "w_ndshard", "k7168_n3072", 2048): 1_026_018,
    ("x_rm", "w_ndshard", "k7168_n3072", 4096): 1_992_864,
    ("x_rm", "w_ndshard", "k7168_n3072", 5120): 2_476_781,
    # ---- x_tile, w_interleaved, kimi_k26 ----
    ("x_tile", "w_interleaved", "kimi_k26", 0): 4_070,
    ("x_tile", "w_interleaved", "kimi_k26", 64): 87_781,
    ("x_tile", "w_interleaved", "kimi_k26", 128): 96_170,
    ("x_tile", "w_interleaved", "kimi_k26", 256): 116_160,
    ("x_tile", "w_interleaved", "kimi_k26", 512): 188_197,
    ("x_tile", "w_interleaved", "kimi_k26", 1024): 329_181,
    ("x_tile", "w_interleaved", "kimi_k26", 2048): 610_179,
    ("x_tile", "w_interleaved", "kimi_k26", 4096): 1_170_847,
    ("x_tile", "w_interleaved", "kimi_k26", 5120): 1_453_256,
    # ---- x_tile, w_interleaved, glm_51 ----
    ("x_tile", "w_interleaved", "glm_51", 0): 4_045,
    ("x_tile", "w_interleaved", "glm_51", 64): 79_105,
    ("x_tile", "w_interleaved", "glm_51", 128): 84_731,
    ("x_tile", "w_interleaved", "glm_51", 256): 103_956,
    ("x_tile", "w_interleaved", "glm_51", 512): 173_267,
    ("x_tile", "w_interleaved", "glm_51", 1024): 305_393,
    ("x_tile", "w_interleaved", "glm_51", 2048): 569_856,
    ("x_tile", "w_interleaved", "glm_51", 4096): 1_088_480,
    ("x_tile", "w_interleaved", "glm_51", 5120): 1_347_858,
    # ---- x_tile, w_ndshard, kimi_k26 ----
    ("x_tile", "w_ndshard", "kimi_k26", 0): 4_001,
    ("x_tile", "w_ndshard", "kimi_k26", 64): 80_326,
    ("x_tile", "w_ndshard", "kimi_k26", 128): 86_027,
    ("x_tile", "w_ndshard", "kimi_k26", 256): 105_258,
    ("x_tile", "w_ndshard", "kimi_k26", 512): 176_629,
    ("x_tile", "w_ndshard", "kimi_k26", 1024): 317_613,
    ("x_tile", "w_ndshard", "kimi_k26", 2048): 598_187,
    ("x_tile", "w_ndshard", "kimi_k26", 4096): 1_160_761,
    ("x_tile", "w_ndshard", "kimi_k26", 5120): 1_442_101,
    # ---- x_tile, w_ndshard, glm_51 ----
    ("x_tile", "w_ndshard", "glm_51", 0): 3_970,
    ("x_tile", "w_ndshard", "glm_51", 64): 69_708,
    ("x_tile", "w_ndshard", "glm_51", 128): 76_084,
    ("x_tile", "w_ndshard", "glm_51", 256): 94_720,
    ("x_tile", "w_ndshard", "glm_51", 512): 164_215,
    ("x_tile", "w_ndshard", "glm_51", 1024): 293_813,
    ("x_tile", "w_ndshard", "glm_51", 2048): 555_530,
    ("x_tile", "w_ndshard", "glm_51", 4096): 1_078_838,
    ("x_tile", "w_ndshard", "glm_51", 5120): 1_339_341,
}

_EXPECTED_BY_IMPLEMENTATION = {
    "unified": _UNIFIED_EXPECTED_NS,
    "moe_fused": _MOE_FUSED_EXPECTED_NS,
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


def _k_filter(implementation: str, model: str, active: int, x_layout: str, weights: str) -> str:
    """Pin exactly one ``test_single_routed_expert_isl_sweep`` case: model + isl + x
    layout + weight layout. Disambiguate substring collisions in the pytest ``-k`` match
    — e.g. ``isl-512`` is a substring of ``isl-5120`` — by excluding any other sweep value
    whose id contains this one."""
    parts = [implementation, f"{model}-isl-{active}", x_layout, weights]
    parts += [
        f"not isl-{other}" for other in _ISL_EXHAUSTIVE_SWEEP if other != active and f"isl-{active}" in f"isl-{other}"
    ]
    return " and ".join(parts)


def _perf_params():
    params = []
    for implementation, expected in _EXPECTED_BY_IMPLEMENTATION.items():
        for x_layout in _LAYOUT_IDS:
            for weights in _WEIGHTS_IDS:
                for model in _ISL_EXHAUSTIVE_MODELS:
                    for active in _ISL_EXHAUSTIVE_SWEEP:
                        key = (x_layout, weights, model, active)
                        if key not in expected:
                            continue
                        command = (
                            f"pytest {_WORKER} " f"-k '{_k_filter(implementation, model, active, x_layout, weights)}'"
                        )
                        params.append(
                            pytest.param(
                                command,
                                {_OP_CODES[implementation]: expected[key]},
                                f"single_routed_expert_{implementation}_{model}_isl{active}_{x_layout}_{weights}",
                                _margin_for(active),
                                id=f"{implementation}-{x_layout}-{weights}-{model}-isl-{active}",
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
