# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: collapse rms_norm's TWO streaming eltwise passes over x into ONE DEST window.

Scope — ONLY the apply stage of rms_norm's compute kernel:

    out = x * (1/rms)  * gamma
              ^col-shaped  ^row-shaped

Everything else is held trivial/constant (perf-lab concept isolation):
  * x, out live in L1 as the core's RESIDENT shard; the CBs are bound to them (zero-copy),
    exactly as the real op does on the TILE + sharded path.  There is NO DRAM, no reader
    payload, no writer payload, no cross-core combine, no Sum(x^2).
  * `1/rms` arrives as a precomputed fp32 column-tile operand (one stat tile per tile-row),
    the same page format the op's reduce produces.
  * gamma arrives as a resident bf16 tile-row whose ROW 0 carries the [W] vector (rows 1..31
    are deliberately garbage, so a variant that ignores the row-0 broadcast fails correctness).
  * a 1-page private CB provides the PACK->UNPACK ordering edge the op calls
    `sync_pack_to_unpack()`.

A "publisher" data-movement kernel marks the three resident CBs available (one reserve/push
each, zero bytes moved) — the same "publish the resident shard" push the op's reader does.

Precision contract is FIXED for every variant and is NOT a lever:
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False, bf16 x/out/gamma,
fp32 stat.  Variants differ ONLY in how the two multiplies are issued.

Variants
--------
two_pass       BASELINE — the op today.  Pass 1: BinaryFpu Mul (bcast Col) -> in-place pack into
               the x shard; sync_pack_to_unpack(); Pass 2: BinaryFpu Mul (bcast Row) -> out.
               Four L1 crossings per tile for two FPU ops.
dest_srca      2nd BASELINE — the fusion already in the op, PARKED OFF (GAMMA_FUSED): one DEST
               window, `DestReuseBinary<..., DEST_TO_SRCA>`, gamma pre-expanded to full tiles
               once by a `UnaryBcast<Row>` in-place pass.  Recorded as +5.7% on the focus shape.
dest_srcb      the mirror side, `DEST_TO_SRCB` (CB -> srcA, DEST -> srcB).  Untried.
sfpu_dest      NEW MECHANISM (a): keep the FPU for x*(1/rms) into D0, `CopyTile` the expanded
               gamma tile into D1, and finish with an SFPU DEST x DEST `MulBinary<D0,D1,D0>`.
               No DEST->Src round trip, no second FPU broadcast load.  Costs one DEST lane
               (chain lane_width 2 -> half the DEST window) and one datacopy per tile.
sfpu_dest_norc same, with the gamma CopyTile's data-format reconfig disabled (x and gamma are
               both bf16, so the per-tile srcA reconfig between the two elements is provably
               redundant here) — isolates the reconfig cost from the mechanism.
sfpu_raw       RAW LLK (helper bypassed on purpose): the same SFPU mechanism, but the gamma tile
               is loaded into a HIGH DEST slot ONCE PER WINDOW instead of once per tile.  Only
               expressible by walking the block hidden-column-outer / row-inner, which the
               chain's row-major grid walk cannot express; see the kernel-head note.
sfpu_bcast_raw RAW LLK, mechanism (d): `sfpu_mul_bcast_row` (api/compute/sfpu_binary_bcast.h) —
               an SFPU DEST x DEST multiply that broadcasts ROW 0 of the second DEST slot.  It
               needs NO gamma expansion pass at all (the row-0 vector goes straight into DEST)
               and the vector slot is loaded once per window.  NOTE: the header documents
               "both source registers must contain FP32 data (compile with fp32_dest_acc_en =
               true)"; our fixed precision contract is fp32_dest_acc_en=False, so this variant
               is measured mainly to record whether it is expressible at all under the user's
               config.  Its PCC is the verdict, and precision is NEVER traded for speed here.
interm_cb      NOT a fusion — the counter-hypothesis the sweep pointed at.  The measured cost the
               fusion is trying to remove turns out to be the baseline's PER-BLOCK overhead, not
               its L1 crossings, and part of that overhead is the private-CB `sync_pack_to_unpack`
               round trip the IN-PLACE rewrite forces.  So: keep TWO independent DEST windows
               (which pipeline against the packer, the reason the fused path lost) but pack pass 1
               into a one-block-deep intermediate CB instead of over x.  The CB's own
               push/wait IS the PACK->UNPACK edge, so no sync op is needed at all.  Costs
               BLOCK_TILES pages of extra L1.
interm_cb_bulk same, but ONE reserve + ONE push per block instead of one per DEST group.  The
               per-group lifecycle degenerates to per-tile when DEST_BLOCK == 1, which is the
               ONLY place `interm_cb` regresses; the bulk lifecycle removes that.

Measured — Blackhole p150b @ 1350 MHz, 64 cores (8x8), 128 tiles/core, bf16, HiFi2,
fp32_dest_acc_en=False.  DEVICE KERNEL DURATION [ns], one run per variant.

  (B,S,blocks)     two_pass   interm_cb_bulk   interm_cb   dest_srca   dest_srcb  sfpu_bcast_raw
  (8,4,4)  FOCUS      8836       8621 .976x    8594 .970x  13018 1.47x 13413 1.52x  29800 3.37x
  (1,4,32)           16196      15329 .946x   15102 .932x  13046 0.80x 13416 0.83x  29276 1.80x
  (1,4,1)  decode     1076        954 .887x     944 .902x   1352 1.27x  1336 1.25x   1418 1.33x
  (1,5,25)           14261      13703 .961x   13525 .948x  12599 0.88x 12944 0.91x  28504 2.00x
  (1,8,16)           12495      12288 .983x   12201 .976x  12708 1.02x*13066 1.05x* 29074 2.33x
  (32,4,1)            8087       8013 .991x    8019 .992x  13044 1.61x 13413 1.66x  29728 3.67x
  (8,1,16)           14233      14167 .995x   16297 1.14x   16635 1.17x 16821 1.18x 29736 2.09x
  * dest_srca / dest_srcb are NUMERICALLY WRONG here (pcc 0.9866): see the DEST-reuse note below.
  Every other cell is pcc 0.99998-0.99999 vs an fp32 torch reference.

  sfpu_dest 75373 (8.51x) and sfpu_raw 73761 (8.32x, and my raw implementation is also
  numerically broken - timing valid, data not) at the focus point: `mul_binary_tile` costs
  ~510 ns/tile against ~30 ns/tile for the FPU broadcast multiply it replaces.  Amortizing the
  per-tile gamma DEST load over a whole window is worth only ~2% of that (75373 -> 73761), so
  the SFPU tile op itself is the cost.  `sfpu_mul_bcast_row` is 3x cheaper than
  `mul_binary_tile` (one full-tile replay pass vs a per-face VectorMode::RC loop) but still
  ~5x more expensive than the FPU.

  DEST_BLOCK knob (two_pass): batching still pays everywhere it is expressible —
  (8,4) 12461 -> 8836 (blk 1 -> 4), (32,4) 11799 -> 8087, (1,5) 15529 -> 14261,
  (1,4) 17138 -> 16196, (1,4,R=1) 1100 -> 1076, (8,1) unchanged (S=1 pins DEST_BLOCK to 1).
  At S=8, though, blk 4 (11992) beats blk 8 (12495) by 4%: a full-DEST window leaves the packer
  nothing to overlap with.

  DEST-REUSE CORRECTNESS: `DestReuseBinary` returns WRONG DATA when the DEST window is the full
  DEST (block_size == DEST_AUTO_LIMIT == 8 here).  Measured wrong at (B=1,S=8), (B=8,S=8) and
  (B=1,S=16) — all block_size 8 — and correct at the same S=8 shape once the window is capped to
  4, so the trigger is the window size, not the shape.  `chain_max_block_v` advertises
  DEST_AUTO_LIMIT / lane_width (= 8) as legal, so the caller gets no signal.  The op's parked
  GAMMA_FUSED path would therefore be silently wrong on any geometry whose S makes DEST_BLOCK 8.
"""

import ttnn

TILE = 32
BF16_TILE = 2048
FP32_TILE = 4096

# CB indices mirror the real op's numbering so the two kernels read side by side.
CB_IN = 0  # x — bound to the core's resident input shard
CB_GAMMA = 1  # gamma, row-0-valid tile row (resident)
CB_INTERM = 3  # interm_cb variant only: x*(1/rms), one block deep
CB_RMS = 6  # 1/rms, one fp32 stat tile per tile-row (resident)
CB_OUT = 9  # out — bound to the core's resident output shard
CB_SYNC = 12  # 1 page, private: the PACK->UNPACK ordering edge

VARIANTS = (
    "two_pass",
    "dest_srca",
    "dest_srcb",
    "sfpu_dest",
    "sfpu_dest_norc",
    "sfpu_raw",
    "sfpu_bcast_raw",
    "interm_cb",
    "interm_cb_bulk",
)
BASELINE = "two_pass"
_VARIANT_ID = {name: i for i, name in enumerate(VARIANTS)}

# Variants whose chain spends a second DEST lane on gamma (lane_width == 2), so the largest
# legal DEST window is halved.  Kept in sync with the kernel's LANES constant.
_TWO_LANE = ("sfpu_dest", "sfpu_dest_norc")

DEST_AUTO_LIMIT = 8  # bf16 DEST, half sync, fp32_dest_acc_en=False
GRID = (8, 8)  # 64 cores, the focus geometry's grid


def _divisor(width, cap):
    """Largest d <= cap that divides width (the op's `dest_block_divisor`)."""
    for d in range(min(cap, width), 1, -1):
        if width % d == 0:
            return d
    return 1


def dest_block_for(variant, s, dest_cap=DEST_AUTO_LIMIT):
    """Tiles per DEST window, exactly as the kernel computes it."""
    cap = dest_cap // 2 if variant in _TWO_LANE else dest_cap
    return _divisor(s, max(cap, 1))


def plan_for(b, s, row_tiles=None, target_tiles=128):
    """Per-core plan: R tile-rows of S tiles, processed in R/B blocks of B rows.

    R defaults to the largest multiple of B with R*S <= target_tiles (and R <= 128), so every
    (B, S) point in the sweep does a comparable amount of work per core.
    """
    if row_tiles is None:
        row_tiles = max(b, (min(128, max(target_tiles // s, 1)) // b) * b)
    if row_tiles % b:
        raise ValueError(f"row_tiles={row_tiles} must be a multiple of B={b}")
    return {
        "B": b,
        "S": s,
        "row_tiles": row_tiles,
        "num_blocks": row_tiles // b,
        "capacity": row_tiles * s,
    }


# =============================================================================
# Publisher (data movement): mark the resident CBs available.  Moves zero bytes.
# =============================================================================
_PUBLISH_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

// x / gamma / (1/rms) are already resident in this core's L1 (the CBs are bound to sharded
// tensors).  All that has to happen is the "publish" push the real reader does for its
// resident shard: make the pages available to compute without moving a byte.
void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_gamma = 1, cb_rms = 6;
    constexpr uint32_t capacity  = get_compile_time_arg_val(0);   // R * S
    constexpr uint32_t row_tiles = get_compile_time_arg_val(1);   // R  (one stat tile per row)
    constexpr uint32_t s_tiles   = get_compile_time_arg_val(2);   // S

    cb_reserve_back(cb_in, capacity);
    cb_push_back(cb_in, capacity);
    cb_reserve_back(cb_rms, row_tiles);
    cb_push_back(cb_rms, row_tiles);
    cb_reserve_back(cb_gamma, s_tiles);
    cb_push_back(cb_gamma, s_tiles);
}
"""

# =============================================================================
# Compute: the apply stage, one variant per VARIANT id.
# =============================================================================
_COMPUTE_KERNEL = r"""
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_interm = 3;
constexpr uint32_t cb_rms = 6;
constexpr uint32_t cb_out = 9;
constexpr uint32_t cb_sync = 12;

constexpr uint32_t V_TWO_PASS = 0;
constexpr uint32_t V_DEST_SRCA = 1;
constexpr uint32_t V_DEST_SRCB = 2;
constexpr uint32_t V_SFPU_DEST = 3;
constexpr uint32_t V_SFPU_DEST_NORC = 4;
constexpr uint32_t V_SFPU_RAW = 5;
constexpr uint32_t V_SFPU_BCAST_RAW = 6;
constexpr uint32_t V_INTERM_CB = 7;
constexpr uint32_t V_INTERM_CB_BULK = 8;

constexpr uint32_t dest_block_divisor(uint32_t width, uint32_t cap) {
    for (uint32_t d = (cap < width ? cap : width); d > 1; --d) {
        if (width % d == 0) {
            return d;
        }
    }
    return 1;
}

// The op's PACK -> UNPACK ordering edge for an in-place handoff: two caller-managed chains that
// both address cb_in exchange no CB handshake, so nothing orders chain N's pack against chain
// N+1's unpack of the same tile.  cb_reserve/push are PACK-only ops, wait/pop UNPACK-only.
ALWI void sync_pack_to_unpack() {
    cb_reserve_back(cb_sync, 1);
    cb_push_back(cb_sync, 1);
    cb_wait_front(cb_sync, 1);
    cb_pop_front(cb_sync, 1);
}

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(2);
    constexpr uint32_t CAPACITY = get_compile_time_arg_val(3);
    constexpr uint32_t VARIANT = get_compile_time_arg_val(4);
    constexpr uint32_t DEST_CAP = get_compile_time_arg_val(5);

    constexpr uint32_t BLOCK_TILES = B * S;
    // Chain lane width: the SFPU variants keep gamma in a second DEST lane per tile.
    constexpr uint32_t LANES = (VARIANT == V_SFPU_DEST || VARIANT == V_SFPU_DEST_NORC) ? 2 : 1;
    constexpr uint32_t DEST_BLOCK = dest_block_divisor(S, DEST_CAP / LANES);

    compute_kernel_hw_startup(cb_in, cb_gamma, cb_out);

    // ---- operand configurations (verbatim from the op) ----
    constexpr auto x_held =
        ckl::input(cb_in, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto rms_col = ckl::input(
        cb_rms,
        ckl::BroadcastDim::Col,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::OperandKind::Col,
        ckl::TileOffset::Unset);
    constexpr auto gamma_row = ckl::input(
        cb_gamma,
        ckl::BroadcastDim::Row,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::TileOffset::Unset);
    // The fused path's gamma: a FULL tile (no intra-tile broadcast is available to DEST reuse
    // or to an SFPU DEST-DEST binary), still indexed per hidden column.
    constexpr auto gamma_full = ckl::input(
        cb_gamma, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row, ckl::TileOffset::Unset);
    constexpr auto gamma_full_norc = ckl::input(
        cb_gamma,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::DataFormatReconfig::Disabled,
        ckl::TileOffset::Unset);
    constexpr auto gamma_expand_in = ckl::input(
        cb_gamma, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block, ckl::TileOffset::Unset);
    constexpr auto gamma_expand_out =
        ckl::output(cb_gamma, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Set);
    constexpr auto in_place =
        ckl::output(cb_in, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::TileOffset::Set);
    constexpr auto to_output_batched =
        ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    // interm_cb: producer synchronizes per DEST group; consumer takes the whole block window
    // upfront (the CB holds exactly one block, so the group-wise pushes can never block).
    constexpr auto to_interm_batched =
        ckl::output(cb_interm, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    // ...and its bulk twin: ONE reserve + ONE push per block, so the intermediate CB's
    // lifecycle cost stops scaling with the number of DEST groups (which is what makes
    // `interm_cb` lose at DEST_BLOCK == 1, where PerBlockSize degenerates to per-tile).
    constexpr auto to_interm_bulk = ckl::output(cb_interm, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);
    constexpr auto interm_block = ckl::input(
        cb_interm,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::AtEnd,
        ckl::OperandKind::Block,
        ckl::TileOffset::Unset);
    constexpr auto block_shape_batched = ckl::IterationShape::grid(B, S).block_size(DEST_BLOCK);

    // ---- the raw variants: DEST window height on a hidden-column-outer walk.
    //      One slot is reserved for the gamma operand, so the window is at most LIMIT-1.
    constexpr uint32_t RAW_ROWS = dest_block_divisor(B, ckl::DEST_AUTO_LIMIT - 1);
    constexpr uint32_t GAMMA_SLOT = ckl::DEST_AUTO_LIMIT - 1;
    constexpr bool IS_RAW = (VARIANT == V_SFPU_RAW) || (VARIANT == V_SFPU_BCAST_RAW);
    if constexpr (IS_RAW) {
        // srcB carries the fp32 stat tile for every mul_tiles_bcast<COL> below; srcA alternates
        // between cb_in and cb_gamma, which are the same format, so one boot reconfig covers it.
        reconfig_data_format_srcb(cb_rms);
    }

    for (uint32_t block = 0; block < NUM_BLOCKS; ++block) {
        const uint32_t pack_base = (block * BLOCK_TILES) % CAPACITY;
        cb_wait_front(cb_in, BLOCK_TILES);
        cb_wait_front(cb_gamma, S);

        if constexpr (VARIANT == V_TWO_PASS) {
            // pass 1 — x *= 1/rms, packed back over x in place
            ckl::eltwise_chain(
                block_shape_batched,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                ckl::PackTile<in_place>{pack_base});
            sync_pack_to_unpack();
            // pass 2 — x *= gamma -> out
            ckl::mul<x_held, gamma_row, to_output_batched>(block_shape_batched);
        } else if constexpr (VARIANT == V_INTERM_CB || VARIANT == V_INTERM_CB_BULK) {
            // pass 1 — x*(1/rms) into a one-block intermediate CB.  The CB's own
            // push/wait supplies the PACK->UNPACK edge, so no sync op at all.
            if constexpr (VARIANT == V_INTERM_CB) {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                    ckl::PackTile<to_interm_batched>{});
            } else {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                    ckl::PackTile<to_interm_bulk>{});
            }
            // pass 2 — interm *= gamma -> out.  Still two independent DEST windows.
            ckl::mul<interm_block, gamma_row, to_output_batched>(block_shape_batched);
        } else if constexpr (VARIANT == V_DEST_SRCA || VARIANT == V_DEST_SRCB) {
            if (block == 0) {
                // gamma: row-0 vector -> full tiles, once, in place.
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(S),
                    ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_expand_in>{},
                    ckl::PackTile<gamma_expand_out>{0});
                sync_pack_to_unpack();
            }
            constexpr auto REUSE =
                (VARIANT == V_DEST_SRCA) ? ckl::DestReuseType::DEST_TO_SRCA : ckl::DestReuseType::DEST_TO_SRCB;
            ckl::eltwise_chain(
                block_shape_batched,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col>{},
                ckl::DestReuseBinary<gamma_full, ckl::BinaryFpuOp::Mul, REUSE>{},
                ckl::PackTile<to_output_batched>{});
        } else if constexpr (VARIANT == V_SFPU_DEST || VARIANT == V_SFPU_DEST_NORC) {
            if (block == 0) {
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(S),
                    ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_expand_in>{},
                    ckl::PackTile<gamma_expand_out>{0});
                sync_pack_to_unpack();
            }
            // One DEST window: FPU does x*(1/rms) into D0, the expanded gamma tile is copied
            // into D1, and an SFPU DEST x DEST multiply finishes the tile.  Nothing is packed
            // between the two multiplies and DEST never round-trips through a Src register.
            if constexpr (VARIANT == V_SFPU_DEST) {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col, ckl::Dst::D0>{},
                    ckl::CopyTile<gamma_full, ckl::Dst::D1>{},
                    ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                    ckl::PackTile<to_output_batched, ckl::Dst::D0>{});
            } else {
                ckl::eltwise_chain(
                    block_shape_batched,
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_held, rms_col, ckl::Dst::D0>{},
                    ckl::CopyTile<gamma_full_norc, ckl::Dst::D1>{},
                    ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
                    ckl::PackTile<to_output_batched, ckl::Dst::D0>{});
            }
        } else {
            // ---- the two RAW variants ----
            // RAW LLK, helper bypassed ON PURPOSE.  Mechanism: an SFPU DEST x DEST multiply
            // needs gamma IN DEST, and a chain re-loads it FOR EVERY TILE because its grid walk
            // is row-major (the lanes of one DEST window differ in hidden column, so each lane
            // needs a different gamma tile).  Walking hidden-column-OUTER / row-INNER instead
            // makes gamma CONSTANT across a whole DEST window: one load per window instead of
            // one per tile.  That walk is not expressible through `IterationShape::grid` +
            // `TileOffset::Strided` (a strided Block index is base + r*stride + c, and this walk
            // needs the OUTER index on the column: c + r*S with r inner), which is why this is
            // raw.  Helpers bypassed: ckl::eltwise_chain / CopyTile / BinaryFpu / MulBinary /
            // PackTile.  The primitives emitted are exactly the ones those elements emit
            // (mul_tiles_bcast<COL>, copy_tile, mul_binary_tile, pack_tile), so the comparison
            // against `sfpu_dest` isolates the gamma-load amortization alone.
            //
            // sfpu_bcast_raw additionally drops the gamma EXPANSION pass: `sfpu_mul_bcast_row`
            // broadcasts row 0 of its DEST operand, so the row-0-only vector goes straight into
            // DEST.  There is no chain element for it at all (kernel_lib has no SFPU-broadcast
            // binary), so that one is raw by capability, not by ergonomics.
            if constexpr (VARIANT == V_SFPU_RAW) {
                if (block == 0) {
                    ckl::eltwise_chain(
                        ckl::IterationShape::tiles(S),
                        ckl::UnaryBcast<ckl::BroadcastDim::Row, gamma_expand_in>{},
                        ckl::PackTile<gamma_expand_out>{0});
                    sync_pack_to_unpack();
                }
            }
            cb_wait_front(cb_rms, B);
            cb_reserve_back(cb_out, BLOCK_TILES);
            for (uint32_t c = 0; c < S; ++c) {
                for (uint32_t r0 = 0; r0 < B; r0 += RAW_ROWS) {
                    tile_regs_acquire();
                    // gamma for this hidden column: ONE load for the whole window.
                    copy_tile_to_dst_init_short(cb_gamma);
                    copy_tile(cb_gamma, c, GAMMA_SLOT);
                    mul_bcast_cols_init(cb_in, cb_rms);
                    for (uint32_t j = 0; j < RAW_ROWS; ++j) {
                        mul_tiles_bcast<ckernel::BroadcastType::COL>(cb_in, cb_rms, (r0 + j) * S + c, r0 + j, j);
                    }
                    if constexpr (VARIANT == V_SFPU_RAW) {
                        mul_binary_tile_init();
                        for (uint32_t j = 0; j < RAW_ROWS; ++j) {
                            mul_binary_tile(j, GAMMA_SLOT, j);
                        }
                    } else {
                        sfpu_mul_bcast_row_init();
                        for (uint32_t j = 0; j < RAW_ROWS; ++j) {
                            sfpu_mul_bcast_row(j, GAMMA_SLOT);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < RAW_ROWS; ++j) {
                        pack_tile<true>(j, cb_out, (r0 + j) * S + c);
                    }
                    tile_regs_release();
                }
            }
            cb_push_back(cb_out, BLOCK_TILES);
            cb_pop_front(cb_rms, B);
        }

        cb_pop_front(cb_in, BLOCK_TILES);
    }
}
"""


def _core_range_set(grid=GRID):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))])


def sharded_memory_config(shard_shape, grid=GRID):
    """Height-sharded row-major over the grid; `shard_shape` is the PER-CORE shard."""
    return ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=_core_range_set(grid),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(cb_id, pages, page_size, dtype, crs):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(x, rms, gamma, out, *, variant, plan, dest_cap=DEST_AUTO_LIMIT, grid=GRID):
    if variant not in _VARIANT_ID:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    crs = _core_range_set(grid)
    s, b = plan["S"], plan["B"]

    compute_ct = [s, b, plan["num_blocks"], plan["capacity"], _VARIANT_ID[variant], dest_cap]
    publish_ct = [plan["capacity"], plan["row_tiles"], s]

    publisher = ttnn.KernelDescriptor(
        kernel_source=_PUBLISH_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=publish_ct,
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=crs,
        compile_time_args=compute_ct,
        # THE USER'S PRECISION CONTRACT — identical for every variant, never a perf lever.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RMS, rms),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, gamma),
        _scratch_cb(CB_SYNC, 1, BF16_TILE, ttnn.bfloat16, crs),
    ]
    if variant in ("interm_cb", "interm_cb_bulk"):
        # exactly one block deep: pass 1's group-wise pushes total BLOCK_TILES == capacity, so
        # pass 2's upfront wait for the whole block can never deadlock against a reserve.
        cbs.append(_scratch_cb(CB_INTERM, b * s, BF16_TILE, ttnn.bfloat16, crs))
    return ttnn.ProgramDescriptor(kernels=[publisher, compute], semaphores=[], cbs=cbs)


def run_variant(x, rms, gamma, out, *, variant, plan, dest_cap=DEST_AUTO_LIMIT, grid=GRID):
    descriptor = create_program_descriptor(x, rms, gamma, out, variant=variant, plan=plan, dest_cap=dest_cap, grid=grid)
    return ttnn.generic_op([x, rms, gamma, out], descriptor)
