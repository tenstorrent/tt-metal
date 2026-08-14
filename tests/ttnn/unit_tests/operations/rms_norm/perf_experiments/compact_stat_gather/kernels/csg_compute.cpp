// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compact_stat_gather micro-benchmark — COMPUTE.
//
// Isolates ONE part of rms_norm's sharded pipeline: the cross-core stat combine.
// Everything downstream of it (scale x, apply gamma, store x) is deliberately
// absent so the measured delta is attributable to the combine alone.
//
// Four MODES, all computing the SAME quantity (per-row 1/sqrt(mean(x^2)+eps)):
//
//   MODE_RAW_TILE (0)  — the op's CURRENT approach (the honest baseline).
//        Contributor ships its RAW per-column partial tile (4 KB fp32, all 32x32
//        lanes valid); root does ONE reduce<SUM, REDUCE_ROW> over the (B, s)
//        gathered block with the finalize fused in.
//
//   MODE_COLLAPSE_4K (1) — contributor collapses first (reduce<SUM, REDUCE_ROW>
//        -> column-0-valid), still ships the whole 4 KB tile.  Same root code.
//        Measures "how much of the root's cost is MATH vs BYTES".
//
//   MODE_COLLAPSE_2K (2) — same collapse, but ship only the two faces that can
//        hold a column-0 vector (face0 + face2 = 2 KB).  A 2x byte cut with no
//        transpose.  Needs the landing CB's other two faces to be zero.
//
//   MODE_ROW_128B (3) — the idea under test.  Contributor collapses AND
//        transposes IN DEST, so its 32 per-row sums land in ROW 0 (two contiguous
//        64 B face-rows).  It writes those 2 x 64 B into ROW `slice_index` of the
//        root's SINGLE landing tile for that tile-row.  The root then combines
//        with ONE reduce<SUM, REDUCE_COL> over that single tile, transposes back
//        in DEST (so the broadcast operand keeps the column shape the scale step
//        needs) and finalizes.  Payload per contributor per tile-row: 128 B
//        instead of 4 KB; root tiles unpacked per block: B instead of B*s.
//
// ---------------------------------------------------------------------------
// RAW-LLK NOTE (why this bench bypasses compute_kernel_lib::reduce in MODE 3)
// ---------------------------------------------------------------------------
// The fused "reduce, then transpose the reduce's own DEST tile" step cannot be
// expressed through `compute_kernel_lib::reduce`'s `post_reduce_op`:
// `transpose_dest` needs `transpose_dest_init` (which rewrites the MATH addrmods
// + MOP and, per llk_math_reduce_uninit's own comment, needs the reduce's SrcA
// ALU format undone first).  The helper calls `reduce_init` ONCE outside its
// output loop and never re-inits per output tile, so a post_reduce_op that
// re-configured MATH would corrupt every later output tile of the same call.
// Hence the hand-rolled window here: reduce_init -> N x reduce_tile ->
// reduce_uninit -> transpose_dest_init -> N x transpose_dest -> finalize -> pack.
// One init pair per DEST window, not per tile.  `ReduceWithinTile::Skip` (the
// template value that WOULD express "the contributor already collapsed") is
// unreachable — see the note in the test file.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_slice_stat = 3;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_scaler = 7;

constexpr uint32_t MODE_RAW_TILE = 0;
constexpr uint32_t MODE_COLLAPSE_4K = 1;
constexpr uint32_t MODE_COLLAPSE_2K = 2;
constexpr uint32_t MODE_ROW_128B = 3;

// DEST window, in tiles, without fp32 DEST accumulation.
constexpr uint32_t DEST_LIMIT = 8;

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t MODE = get_compile_time_arg_val(3);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t LANDING_ROWS = get_compile_time_arg_val(5);  // ceil(s/32) for MODE 3, else s

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t CHUNK = BLOCK_ROWS < DEST_LIMIT ? BLOCK_ROWS : DEST_LIMIT;
    static_assert(BLOCK_ROWS % CHUNK == 0, "BLOCK_ROWS must be a whole number of DEST windows");
    constexpr uint32_t GATHER_PAGES = LANDING_ROWS * BLOCK_ROWS;

    // Same dispatch as the op: pairwise add beats matmul-with-ones past ~4 tiles.
    constexpr auto COMBINE_ALGORITHM =
        NUM_HIDDEN_SLICES >= 4 ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_root = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);
    const uint32_t eps_bits = get_arg_val<uint32_t>(3);

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_sq_partials);

    auto finalize = [inv_w_bits, eps_bits](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, inv_w_bits);
        add_unary_tile(dst_idx, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    };

    constexpr auto x_held =
        ckl::input(cb_input_tiles, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto block_shape = ckl::IterationShape::grid(BLOCK_ROWS, SLICE_HIDDEN_TILES);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_wait_front(cb_input_tiles, IN_WAIT_TILES);

        // ---- square_accumulate_block (identical in every mode) ----
        ckl::sum_of_squares<x_held, ckl::row_output(cb_sq_partials)>(block_shape);

        // ---- the contributor's half of the combine ----
        if constexpr (MODE == MODE_COLLAPSE_4K || MODE == MODE_COLLAPSE_2K) {
            // collapse_partial_block, straight through the helper.  The helper's
            // reduce_init leaves the packer's REDUCE_ROW edge mask set for every
            // pack in the call, so the emitted tile is column-0-valid with ZEROS
            // everywhere else — which is what makes the root's within-tile
            // REDUCE_ROW over it (and MODE 2's two-face shipment) exact.
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sq_partials,
                cb_scaler,
                cb_slice_stat,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(ckl::ReduceInputBlockShape::of(BLOCK_ROWS, 1));
        } else if constexpr (MODE == MODE_ROW_128B) {
            // collapse AND transpose, in ONE DEST window per CHUNK tiles.
            cb_wait_front(cb_sq_partials, BLOCK_ROWS);
            cb_reserve_back(cb_slice_stat, BLOCK_ROWS);
            for (uint32_t base = 0; base < BLOCK_ROWS; base += CHUNK) {
                // REDUCE_ROW SUM swaps operands (scaler -> SrcA, data -> SrcB).
                reconfig_data_format(cb_scaler, cb_sq_partials);
                pack_reconfig_data_format(cb_slice_stat);
                tile_regs_acquire();
                reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_sq_partials, cb_scaler, cb_slice_stat);
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                        cb_sq_partials, cb_scaler, base + i, 0, i);
                }
                // Undo the reduce's SrcA ALU format AND clear the packer's
                // column-0 edge mask: after the transpose the live lanes are in
                // ROW 0, so a REDUCE_ROW mask would zero exactly the answer.
                reduce_uninit(cb_sq_partials);
                transpose_dest_init<false, true>(cb_sq_partials);
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    transpose_dest<false, true>(i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    pack_tile(i, cb_slice_stat, base + i);
                }
                tile_regs_release();
            }
            cb_push_back(cb_slice_stat, BLOCK_ROWS);
            cb_pop_front(cb_sq_partials, BLOCK_ROWS);
        }

        // ---- combine_block, root half ----
        if (is_root) {
            if constexpr (MODE == MODE_ROW_128B) {
                cb_wait_front(cb_gathered_partials, GATHER_PAGES);
                cb_reserve_back(cb_rms_bcast, BLOCK_ROWS);
                for (uint32_t base = 0; base < BLOCK_ROWS; base += CHUNK) {
                    reconfig_data_format(cb_gathered_partials, cb_scaler);
                    pack_reconfig_data_format(cb_rms_bcast);
                    tile_regs_acquire();
                    reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
                        cb_gathered_partials, cb_scaler, cb_rms_bcast);
                    for (uint32_t i = 0; i < CHUNK; ++i) {
                        // Landing layout: page (ht * B + b).  ht > 0 only when
                        // s > 32 (more contributors than a tile has rows), and
                        // REDUCE_COL accumulates the ht tiles into the same DEST.
                        for (uint32_t ht = 0; ht < LANDING_ROWS; ++ht) {
                            reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
                                cb_gathered_partials, cb_scaler, ht * BLOCK_ROWS + base + i, 0, i);
                        }
                    }
                    reduce_uninit(cb_gathered_partials);
                    // row-0 combined sums -> col 0, so the mcast operand keeps the
                    // column shape `BroadcastDim::Col` needs downstream.
                    transpose_dest_init<false, true>(cb_gathered_partials);
                    for (uint32_t i = 0; i < CHUNK; ++i) {
                        transpose_dest<false, true>(i);
                        finalize(i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < CHUNK; ++i) {
                        pack_tile(i, cb_rms_bcast, base + i);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_rms_bcast, BLOCK_ROWS);
                cb_pop_front(cb_gathered_partials, GATHER_PAGES);
            } else {
                cb_wait_front(cb_gathered_partials, GATHER_PAGES);
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_gathered_partials,
                    cb_scaler,
                    cb_rms_bcast,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    COMBINE_ALGORITHM,
                    ckl::NoAccumulation,
                    decltype(finalize)>(
                    ckl::ReduceInputBlockShape::of(BLOCK_ROWS, NUM_HIDDEN_SLICES),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    finalize);
            }
        }

        cb_pop_front(cb_input_tiles, BLOCK_TILES);
    }
}
