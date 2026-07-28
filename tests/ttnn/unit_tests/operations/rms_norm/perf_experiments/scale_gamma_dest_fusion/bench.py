# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for rms_norm phases 5+6 — `x * (1/rms) * gamma`.

WHAT IS ISOLATED
----------------
Only the two compute passes are reconstructed:

    phase 5   scaled = x * recip           (recip is a per-row column, BroadcastDim::Col)
    phase 6   out    = scaled * gamma      (gamma is a per-column row,  BroadcastDim::Row)

Everything else is held constant and trivial (perf-lab concept isolation):

  * `x`, `recip`, `gamma` and `out` are ALL zero-copy BLOCK_SHARDED L1 tensors, so
    the reader is a pure credit publisher and the writer does not exist. There is
    ZERO NoC traffic in the measured window — the device kernel duration is the
    two compute passes plus their CB handshakes and nothing else.
  * `recip` is supplied by the host, so the rsqrt (phase 4) that produces it in
    the real op is absent. That matters: in the real op phase 5's UNPACK thread
    enters `cb_wait_front(cb_rms_recip, ht)` and blocks on the rsqrt's MATH, so
    its measured 17.6 us is stall + work. Here it is work only.

VARIANTS (`variant=`)
---------------------
  baseline       the op's spelling: two `eltwise_chain` calls, `cb_scaled` in between
                 (HT_BLOCK*WT bf16 tiles of L1, one pack + one unpack per tile).
  fused          ONE chain: BinaryFpu<x, recip, Mul, Col> -> DestReuseBinary<gamma_full,
                 Mul> -> PackTile<out>. `DestReuseBinary` carries no BroadcastDim, so
                 gamma is first pre-expanded to WT FULL tiles (row 0 replicated down 32
                 rows) by a once-per-core `UnaryBcast<Row>` chain. `cb_scaled` is retired.
  fused_pre      the same fused chain, but the HOST hands in an already row-replicated
                 gamma so the pre-expansion pass is skipped. Not shippable (the op is
                 given a [1, W] gamma); it prices the pre-expansion apart from the saving.
  fused_srcb     ... with DestReuseType::DEST_TO_SRCB instead of DEST_TO_SRCA.
  fused_norc     ... with every dtype-reconfig knob None (legal only with a bf16 recip,
                 where srcA and srcB share one format) — prices the per-window reconfig.
  fused_sfpu     the other way to combine DEST with a second operand: gamma copied into
                 a SECOND DEST lane, multiply in the SFPU.
  bcast_free     TWO chains still, but phase 6 reads the pre-expanded gamma with
                 BroadcastDim::None instead of ::Row — prices the row-broadcast MOP.
  baseline_blk1  the op today at block_size 1 — prices the DEST-sync window itself.

`recip_dtype` is an independent knob (float32 = the op today, bfloat16 = an option
with a precision cost): it only changes the aliased CB's format, never the kernel.

MEASURED — blackhole_p150b, 8x8 = 64 cores, AICLK 1349.99 MHz, bf16 / HiFi2 /
fp32_dest_acc_en=False / math_approx_mode=False, DEVICE KERNEL DURATION [ns],
one dispatch per number. Focus shape (1,1,8192,1024) BLOCK_SHARDED [1024,128]:

    option                     ns    ratio       PCC   L1/core B
    baseline (fp32 recip)    8886    1.00x  0.999963     729_088
    baseline (bf16 recip)    8879    1.00x  0.999961     663_552
    bcast_free               9680    0.92x  0.999963     737_280
    fused_pre               12715    0.70x  0.999963     663_552
    fused                   13176    0.67x  0.999963     671_744
    fused_norc (bf16)       13188    0.67x  0.999961     606_208
    fused_srcb              13606    0.65x  0.999959     671_744
    baseline_blk1           14468    0.61x  0.999963     729_088
    fused_sfpu              75520    0.12x  0.999970     671_744

Per-core zone split (UNPACK / MATH / PACK ns, 4 instances summed):

    baseline    bx_scale     3770 / 4375 / 4110      per-RISC totals
                bx_gamma_mul 4167 / 3958 / 3992      7937 / 8333 / 8102
    fused       bx_fused    11744 /12336 /12102
                bx_expand     467 /  406 /  417

The baseline's three engines are each ~92% of the 8886 ns kernel, i.e. phases 5+6
are already at the TRISC-throughput floor for 2 FPU ops + 2 unpacks + 2 packs per
output tile; there is no idle engine to reclaim. Every route that removes the
`cb_scaled` round trip pays more for the DEST dependency than the round trip cost:
the fused chain's UNPACK, MATH and PACK all land on the same ~12.1 us, i.e. fully
serialized, because DEST is live across two dependent FPU ops.

`baseline_blk1` prices the DEST-sync window at (14468-8886)/(256-64) = 29 ns per
window. The baseline spends 64 x 29 = 1.86 us (21%) there, and half of it would be
recoverable if one window could span 2 tile-rows (8 tiles, which DEST holds) — but
that is NOT expressible: the `1/rms` operand is `OperandKind::Col`, whose index is
the row, so a window is pinned to one tile-row. That is the remaining headroom and
it is a helper-surface gap, not a kernel choice.

Predicate sweep (baseline vs candidates, one dispatch each):

    regime      baseline    fused           bcast_free
    focus           8891   13176 (0.67x)     9717 (0.92x)
    decode          1437    2136 (0.67x)     2151 (0.67x)
    no_gamma        4714   nothing to fuse   -
    streaming       8864   13238 (0.67x)     9764 (0.91x)
    rm             14502   18123 (0.80x)    15150 (0.96x)

Reference: the whole op on the same focus config measures 75_389 ns with
`cmp_scale` = 17634 / 3756 / 7760 and `cmp_gamma_mul` = 4185 / 3949 / 3995.
This bench reproduces `cmp_gamma_mul` to within 1% and `cmp_scale`'s MATH to
within 17%, but its phase-5 UNPACK is 3770 ns against the op's 17634 — so ~13.9 us
of `cmp_scale`'s UNPACK is the thread parked in `cb_wait_front(cb_rms_recip, ht)`
behind `cmp_rsqrt` (42403 / 56111 / 52771, itself waiting on the cross-core
combine: `rdr_mcast` 56493, `wtr_gather_hop` 59086). Stages 5+6 are 8.9 us of real
work — 11.8% of the kernel, not the 28% the zone table reads as.

TWO HELPER FINDINGS this bench measured (both silent, both caught only by the
non-uniform-gamma gate):
  1. `UnaryBcast::exec` hard-codes `in_tile_index = 0` (eltwise_chain.inl), so it
     always broadcasts the tile at the CB FRONT and ignores the chain's walk index.
     Only `InputLifecycle::Streaming` walks; a Bulk/CallerManaged lifecycle expands
     tile 0 WT times (measured: w-tile 0 PCC 0.99998, w-tiles 1..3 PCC 0.79-0.85).
  2. A `DestReuseBinary` chain at `block_size == DEST_AUTO_LIMIT` corrupts part of
     the HIGHEST DEST lane — one face of the last tile (decode geometry, WT == 8:
     w-tile 7 rows 16..31 x cols 16..31 wrong, w-tiles 0..6 exact). `block_size <=
     DEST_AUTO_LIMIT - 1` is exact, so the reuse path needs one spare slot that
     `chain_max_block_v` does not reserve. `SGDF_FUSED_BLK` re-runs the bisect.

REGIMES (`regime=`)
-------------------
All regimes use a 8x8 = 64-core grid and a BLOCK_SHARDED input, so one builder
serves them all; the shard geometry is what moves.

  focus       shard [1024, 128] -> per core ht=32 tile-rows, WT=4, HT_BLOCK=8
              (the perf-flagged (1,1,8192,1024) case, byte-identical blocking)
  decode      shard [32, 256]   -> per core ht=1 tile-row,   WT=8, HT_BLOCK=1
              (the WIDTH_SHARDED decode geometry's per-core block)
  no_gamma    focus geometry, HAS_GAMMA=0 (phase 5 packs straight to the output;
              there is nothing to fuse)
  streaming   focus geometry, x consumed with InputLifecycle::Bulk + no TileOffset
              (the op's !X_RESIDENT spelling of the same two passes)
  rm          focus geometry + an `untilize` consumer of cb_output_tiles (the op's
              IS_RM spelling). cb_out stays aliased on the TILED output shard, so the
              numeric gate is unchanged and the untilize is pure added cost; a writer
              kernel drains its tile-paged output CB, as the real writer does.
"""

from __future__ import annotations

import os

import ttnn

TILE = 32

CB_X = 0
CB_GAMMA = 1
CB_RECIP = 2
CB_GAMMA_FULL = 3
CB_OUT = 16
CB_OUT_RM = 17
CB_SCALED = 28

VARIANTS = (
    "baseline",  # the op today: two chains, cb_scaled in between
    "fused",  # one chain, DEST_TO_SRCA, kernel-expanded gamma
    "fused_pre",  # ... with a host-expanded gamma (prices the expansion away)
    "fused_srcb",  # ... with the other DEST-reuse direction
    "fused_norc",  # ... with every dtype reconfig knob off (needs a bf16 recip)
    "bcast_free",  # TWO chains, but phase 6 drops BroadcastDim (pre-expanded gamma)
    "baseline_blk1",  # the op today at block_size 1 — prices the DEST-sync window
    "fused_sfpu",  # one chain, gamma copied to a 2nd DEST lane, SFPU multiply
)
_VARIANT_ID = {name: i for i, name in enumerate(VARIANTS)}

# Variants whose srcA/srcB formats never change, so the reconfig knobs may be off.
_NEEDS_UNIFORM_FORMAT = ("fused_norc",)

REGIMES = ("focus", "decode", "no_gamma", "streaming", "rm")

# regime -> (shard_rows, shard_wt, ht_block, has_gamma, x_resident, is_rm)
#
# ht_block mirrors the op's derivation: min(TILE_BLOCK_BUDGET // WT_CHUNK, rows_core_max).
# For the focus cell the op derives 8; the decode cell has one tile-row so it is 1.
_REGIME_SPEC = {
    "focus": (1024, 4, 8, True, True, False),
    "decode": (32, 8, 1, True, True, False),
    "no_gamma": (1024, 4, 8, False, True, False),
    "streaming": (1024, 4, 8, True, False, False),
    "rm": (1024, 4, 8, True, True, True),
}

GRID_X = 8
GRID_Y = 8

# The user's precision contract for the perf-flagged cell. FIXED for every
# variant — never a lever (see the part-optimizer's precision contract).
MATH_FIDELITY = ttnn.MathFidelity.HiFi2
FP32_DEST_ACC_EN = False
MATH_APPROX_MODE = False
DST_FULL_SYNC_EN = False


def regime_geometry(regime):
    """(shard_rows, shard_wt, ht_block, has_gamma, x_resident, is_rm) for a regime."""
    if regime not in _REGIME_SPEC:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    return _REGIME_SPEC[regime]


def regime_shapes(regime):
    """The four tensor shapes + shard shapes this regime needs.

    Returned as a dict of name -> (shape, shard_shape).  The whole-tensor shape is
    just `grid x shard`, so the per-core block is exactly the geometry above.
    """
    hs, wt, _ht_block, _g, _xr, _rm = regime_geometry(regime)
    ws = wt * TILE
    W = GRID_X * ws
    H = GRID_Y * hs
    return {
        # x: [H, W], one [hs, ws] shard per core
        "x": ((1, 1, H, W), (hs, ws)),
        # recip: one W-TILE per core, holding the per-row 1/rms in every column
        # (BroadcastDim::Col reads column 0). Replicated across the 8 column bands
        # so each core's own shard carries the values for its own row band.
        "recip": ((1, 1, H, GRID_X * TILE), (hs, TILE)),
        # gamma: one TILE-ROW per core band, WT tiles wide (the op's per-core slice)
        "gamma": ((1, 1, GRID_Y * TILE, W), (TILE, ws)),
        "out": ((1, 1, H, W), (hs, ws)),
    }


def compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=MATH_FIDELITY,
        fp32_dest_acc_en=FP32_DEST_ACC_EN,
        math_approx_mode=MATH_APPROX_MODE,
        dst_full_sync_en=DST_FULL_SYNC_EN,
    )


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])


# =============================================================================
# Kernels (inline sources)
# =============================================================================

_CB_DEFS = r"""
namespace {
constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_recip = 2;
constexpr uint32_t cb_gamma_full = 3;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_rm = 17;
constexpr uint32_t cb_scaled = 28;
}  // namespace
"""

# Credit publisher. Every input CB is ALIASED onto this core's own resident L1
# shard (ttnn.cb_descriptor_from_sharded_tensor), so there is nothing to read —
# exactly the `rdr_shard_publish` step of the real reader. One push per CB makes
# the resident pages visible to compute; nothing is ever re-pushed.
_READER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
"""
    + _CB_DEFS
    + r"""
void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(1);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(2) != 0;

    MaybeDeviceZoneScope("bx_publish");
    cb_reserve_back(cb_x, NUM_ROWS * WT);
    cb_push_back(cb_x, NUM_ROWS * WT);
    cb_reserve_back(cb_recip, NUM_ROWS);
    cb_push_back(cb_recip, NUM_ROWS);
    if constexpr (HAS_GAMMA) {
        cb_reserve_back(cb_gamma, WT);
        cb_push_back(cb_gamma, WT);
    }
}
"""
)

# Drains the untilized row pages on the IS_RM path — the real writer's job (there
# it also NoC-writes them out; here the write is not part of the concept under
# test, so only the CB handshake is kept).
_WRITER = (
    r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
"""
    + _CB_DEFS
    + r"""
void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(1);
    MaybeDeviceZoneScope("bx_drain");
    for (uint32_t h = 0; h < NUM_ROWS; ++h) {
        cb_wait_front(cb_out_rm, WT);
        cb_pop_front(cb_out_rm, WT);
    }
}
"""
)

_COMPUTE = (
    r"""
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
"""
    + _CB_DEFS
    + r"""
namespace ckl = compute_kernel_lib;

// Phases 5 and 6 of rms_norm, three ways. Zone names: bx_gamma_expand /
// bx_scale / bx_gamma_mul / bx_fused / bx_untilize.
void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(3);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(4) != 0;
    constexpr bool X_RESIDENT = get_compile_time_arg_val(5) != 0;
    constexpr bool IS_RM = get_compile_time_arg_val(6) != 0;
    // 0 => the chain's own DEST capacity. A smaller value caps the FUSED chains'
    // DEST-sync window only (see FUSED_BLOCK below).
    constexpr uint32_t FUSED_BLK_ARG = get_compile_time_arg_val(7);

    // Same single source as the op: the coarsest DEST-sync window that fits.
    // `baseline_blk1` (VARIANT 6) forces 1 tile per window so the per-window
    // acquire/commit/wait/release + reserve/push cost can be priced directly.
    constexpr uint32_t DEST_BLOCK = (VARIANT == 6) ? 1u : ckl::DEST_AUTO_LIMIT;
    // MEASURED CONSTRAINT: a DestReuseBinary chain running block_size ==
    // DEST_AUTO_LIMIT corrupts part of the HIGHEST DEST lane (decode geometry,
    // WT == 8: w-tile 7's rows 16..31, cols 16..31 — one face — came back wrong
    // while w-tiles 0..6 were exact). The two-chain spellings at the same block
    // size are exact, so it is specific to feeding DEST back into srcA/srcB.
    constexpr uint32_t FUSED_BLOCK = (FUSED_BLK_ARG == 0) ? (ckl::DEST_AUTO_LIMIT - 1u) : FUSED_BLK_ARG;
    // A REDUCE_ROW result is column-shaped -> BroadcastDim::Col; the operand is
    // indexed by row (Col) when the block spans rows, else it is the one tile.
    constexpr auto rms_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Col : ckl::OperandKind::Scalar;
    constexpr auto gamma_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Row : ckl::OperandKind::Block;
    constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_scaled : cb_out;

    constexpr bool FUSED = (VARIANT >= 1 && VARIANT <= 4) && HAS_GAMMA;
    constexpr bool FUSED_SFPU = (VARIANT == 7) && HAS_GAMMA;
    // VARIANT 5 (bcast_free) keeps TWO chains but its phase 6 also needs the
    // pre-expanded gamma. VARIANT 2 is handed an already-row-replicated gamma by
    // the host, so for it the expansion pass does not exist at all.
    constexpr bool EXPAND_GAMMA = (VARIANT == 1 || VARIANT == 3 || VARIANT == 4 || VARIANT == 5 || VARIANT == 7) && HAS_GAMMA;
    constexpr uint32_t cb_gamma_wide = (VARIANT == 2) ? cb_gamma : cb_gamma_full;
    constexpr auto REUSE = (VARIANT == 3) ? ckl::DestReuseType::DEST_TO_SRCB : ckl::DestReuseType::DEST_TO_SRCA;
    // VARIANT 4 asserts srcA and srcB are already programmed at the one shared
    // format (only legal when cb_recip carries the input dtype), so no dtype
    // reconfig is emitted at all — this prices the per-window reconfig.
    constexpr auto BFPU_RC =
        (VARIANT == 4) ? ckl::BinaryDataFormatReconfig::None : ckl::BinaryDataFormatReconfig::Input;
    constexpr auto DR_RC = (VARIANT == 4) ? ckl::DestReuseReconfig::None : ckl::DestReuseReconfig::Input;
    constexpr bool BCAST_FREE = (VARIANT == 5) && HAS_GAMMA;

    constexpr auto x_life = X_RESIDENT ? ckl::InputLifecycle::CallerManaged : ckl::InputLifecycle::Bulk;
    constexpr auto x_off = X_RESIDENT ? ckl::TileOffset::Set : ckl::TileOffset::Unset;

    compute_kernel_hw_startup(cb_x, cb_recip, cb_out);

    if constexpr (HAS_GAMMA) {
        cb_wait_front(cb_gamma, WT);
    }

    // ---- gamma pre-expansion (once per core) ----------------------------
    // DestReuseBinary has no BroadcastDim, so gamma has to already be a FULL
    // tile. UnaryBcast<Row> is the row-broadcast datacopy: it reads row 0 of a
    // gamma tile and writes all 32 rows of DEST, so WT packs turn the [1, W]
    // gamma into WT full tiles. Cost is O(WT), not O(NUM_ROWS * WT).
    //
    // HELPER CONSTRAINT (measured, not guessed): `UnaryBcast::exec` hard-codes
    // `in_tile_index = 0` — it always broadcasts the tile at the CB FRONT and
    // ignores the chain's walk index. So the operand lifecycle has to be
    // InputLifecycle::Streaming (wait 1 / pop 1 per iter) to walk gamma's WT
    // tiles; a Bulk/CallerManaged lifecycle silently expands gamma tile 0 WT
    // times (measured: w-tile 0 PCC 0.99998, w-tiles 1..3 PCC 0.79-0.85 — a bug
    // an all-ones-gamma test cannot see, which is why the ramp gate exists).
    // Popping cb_gamma here is safe precisely BECAUSE of the fusion: nothing
    // reads the [1, W] gamma again once cb_gamma_full exists.
    if constexpr (EXPAND_GAMMA) {
        MaybeDeviceZoneScope("bx_gamma_expand");
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(WT),
            ckl::UnaryBcast<ckl::BroadcastDim::Row, cb_gamma, ckl::InputLifecycle::Streaming>{},
            ckl::PackTile<cb_gamma_full, ckl::OutputLifecycle::Streaming>{});
        cb_wait_front(cb_gamma_full, WT);
    }

    const uint32_t num_row_blocks = (NUM_ROWS + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        uint32_t ht = NUM_ROWS - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }
        const auto blk = ckl::EltwiseShape::grid(ht, WT, DEST_BLOCK);
        const auto fblk = ckl::EltwiseShape::grid(ht, WT, FUSED_BLOCK);

        if constexpr (FUSED) {
            // ONE dst-sync window for both multiplies. D0 holds x * recip, then
            // DestReuseBinary feeds that DEST tile straight back into the FPU
            // against the pre-expanded gamma tile — no L1 round trip.
            MaybeDeviceZoneScope("bx_fused");
            ckl::eltwise_chain(
                fblk,
                ckl::BinaryFpu<
                    cb_x,
                    cb_recip,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Col,
                    x_life,
                    ckl::InputLifecycle::HeldBulk,
                    BFPU_RC,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    rms_kind,
                    x_off,
                    ckl::TileOffset::Unset>{0, 0},
                ckl::DestReuseBinary<
                    cb_gamma_wide,
                    ckl::BinaryFpuOp::Mul,
                    REUSE,
                    ckl::InputLifecycle::CallerManaged,
                    DR_RC,
                    ckl::Dst::D0,
                    ckl::Dst::D0,
                    gamma_kind,
                    ckl::TileOffset::Unset>{},
                ckl::PackTile<cb_out, ckl::OutputLifecycle::Chunked>{});
        } else if constexpr (FUSED_SFPU) {
            // The other way to combine DEST with a second operand: copy the
            // pre-expanded gamma into a SECOND DEST lane and multiply in the
            // SFPU. chain_lane_width becomes 2, so max_block halves to 4 — which
            // costs nothing at WT == 4, where the walk already caps the window at
            // WT. One chain, so HALF the DEST-sync windows of the baseline, and
            // no cb_scaled round trip; the price is one FPU mul traded for an
            // SFPU mul plus a full-tile gamma unpack per output tile.
            MaybeDeviceZoneScope("bx_fused");
            ckl::eltwise_chain(
                fblk,
                ckl::BinaryFpu<
                    cb_x,
                    cb_recip,
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::Col,
                    x_life,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    rms_kind,
                    x_off,
                    ckl::TileOffset::Unset>{0, 0},
                ckl::CopyTile<
                    cb_gamma_full,
                    ckl::Dst::D1,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::CopyTileReconfig::Input,
                    gamma_kind>{},
                ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                ckl::PackTile<cb_out, ckl::OutputLifecycle::Chunked>{});
        } else {
            // ---- phase 5: x * (1/rms) ----
            {
                MaybeDeviceZoneScope("bx_scale");
                ckl::eltwise_chain(
                    blk,
                    ckl::BinaryFpu<
                        cb_x,
                        cb_recip,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col,
                        x_life,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        rms_kind,
                        x_off,
                        ckl::TileOffset::Unset>{0, 0},
                    ckl::PackTile<cb_scale_out, ckl::OutputLifecycle::Chunked>{});
            }
            // ---- phase 6: * gamma ----
            // BCAST_FREE reads the PRE-EXPANDED gamma with BroadcastDim::None, so
            // the row-broadcast unpack MOP (which re-unpacks the srcB operand once
            // per FACE — `ckernel_template tmp(outer, inner, unpack_srcb, srca_op)`
            // in llk_unpack_AB.h, vs COL's single `set_start_op(unpack_srcb)`) is
            // replaced by the plain two-operand one.
            if constexpr (HAS_GAMMA) {
                MaybeDeviceZoneScope("bx_gamma_mul");
                constexpr uint32_t cb_g = BCAST_FREE ? cb_gamma_full : cb_gamma;
                constexpr auto g_bcast = BCAST_FREE ? ckl::BroadcastDim::None : ckl::BroadcastDim::Row;
                ckl::eltwise_chain(
                    blk,
                    ckl::BinaryFpu<
                        cb_scaled,
                        cb_g,
                        ckl::BinaryFpuOp::Mul,
                        g_bcast,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::Dst::D0,
                        ckl::OperandKind::Block,
                        gamma_kind,
                        ckl::TileOffset::Unset,
                        ckl::TileOffset::Unset>{0, 0},
                    ckl::PackTile<cb_out, ckl::OutputLifecycle::Chunked>{});
            }
        }

        if constexpr (IS_RM) {
            MaybeDeviceZoneScope("bx_untilize");
            ckl::untilize<WT, cb_out, cb_out_rm>(ht);
        }

        // Held CBs the chain deliberately never popped.
        cb_pop_front(cb_recip, ht);
        if constexpr (X_RESIDENT) {
            cb_pop_front(cb_x, ht * WT);
        }
    }
}
"""
)


# =============================================================================
# Program descriptor
# =============================================================================


def _scratch_cb(cb_id, dtype, page_size, num_pages):
    return ttnn.CBDescriptor(
        total_size=page_size * max(1, num_pages),
        core_ranges=_grid(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(x, recip, gamma, out, *, variant, regime):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    hs, wt, ht_block, has_gamma, x_resident, is_rm = regime_geometry(regime)
    num_rows = hs // TILE
    if not has_gamma and variant != "baseline":
        raise ValueError("the no_gamma regime has nothing to fuse — only `baseline` is defined")

    ct = [
        _VARIANT_ID[variant],
        wt,
        ht_block,
        num_rows,
        1 if has_gamma else 0,
        1 if x_resident else 0,
        1 if is_rm else 0,
        int(os.environ.get("SGDF_FUSED_BLK", "0")),
    ]

    tb = ttnn.tile_size(x.dtype)

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RECIP, recip),
    ]
    # Every CB is declared even when a variant does not use it, so the compile-time
    # CB descriptors the helpers read stay valid in `if constexpr`-discarded branches.
    if has_gamma:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, gamma))
    else:
        cbs.append(_scratch_cb(CB_GAMMA, x.dtype, tb, 1))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out))
    # The untilize helper is symmetric (tile-sized pages on BOTH sides), so
    # cb_out_rm is tile-paged exactly as the op sizes it. On the RM path the
    # output shard still receives the TILED result (cb_out is aliased on it), so
    # the numeric gate is unchanged and the untilize is pure added cost.
    cbs.append(_scratch_cb(CB_OUT_RM, x.dtype, tb, (2 * ht_block * wt) if is_rm else 1))
    two_pass = variant in ("baseline", "bcast_free", "baseline_blk1")
    cbs.append(_scratch_cb(CB_SCALED, x.dtype, tb, (ht_block * wt) if (two_pass and has_gamma) else 1))
    wide = has_gamma and variant in ("fused", "fused_srcb", "fused_norc", "bcast_free", "fused_sfpu")
    cbs.append(_scratch_cb(CB_GAMMA_FULL, x.dtype, tb, wt if wide else 1))

    reader = ttnn.KernelDescriptor(
        kernel_source=_READER,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_grid(),
        compile_time_args=[wt, num_rows, 1 if has_gamma else 0],
        runtime_args=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    kernels = [reader]
    if is_rm:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=_grid(),
                compile_time_args=[wt, num_rows],
                runtime_args=[],
                config=ttnn.WriterConfigDescriptor(),
            )
        )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_grid(),
        compile_time_args=ct,
        runtime_args=[],
        config=compute_config(),
    )
    kernels.append(compute)
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def run(x, recip, gamma, out, *, variant, regime):
    io = [x, recip]
    if gamma is not None:
        io.append(gamma)
    io.append(out)
    descriptor = create_program_descriptor(x, recip, gamma, out, variant=variant, regime=regime)
    return ttnn.generic_op(io, descriptor)


def l1_bytes(regime, variant, *, recip_dtype=ttnn.float32, dtype=ttnn.bfloat16):
    """Per-core L1 this variant commits (aliased shards + program CBs), in bytes."""
    hs, wt, ht_block, has_gamma, _xr, is_rm = regime_geometry(regime)
    num_rows = hs // TILE
    tb = ttnn.tile_size(dtype)
    total = num_rows * wt * tb  # x shard
    total += num_rows * ttnn.tile_size(recip_dtype)  # recip shard
    if has_gamma:
        total += wt * tb  # gamma shard
    total += num_rows * wt * tb  # output shard (tiled or the RM equivalent)
    if is_rm:
        total += 2 * ht_block * wt * tb  # cb_out_rm (program CB, tile-paged)
    if has_gamma and variant in ("baseline", "bcast_free", "baseline_blk1"):
        total += ht_block * wt * tb  # cb_scaled
    if has_gamma and variant in ("fused", "fused_srcb", "fused_norc", "bcast_free", "fused_sfpu"):
        total += wt * tb  # cb_gamma_full
    return total
