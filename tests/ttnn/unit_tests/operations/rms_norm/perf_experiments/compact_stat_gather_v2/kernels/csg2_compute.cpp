// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compact_stat_gather_v2 micro-benchmark — COMPUTE.
//
// Isolates ONE part of rms_norm's sharded pipeline: the cross-core stat gather
// and the owner's combine, on the POST-Perf-1 reduce-scatter topology.
//
//   MODE_RAW_4K (0) — the op's CURRENT approach == the honest baseline.
//        Contributor ships its RAW per-column partial tile (4 KB fp32, all
//        32x32 lanes valid); the OWNER runs ONE reduce<SUM, REDUCE_ROW> over the
//        (OWN_ROWS, s) gathered block with the finalize fused in.  The owner
//        therefore unpacks s*OWN_ROWS tiles per block.
//
//   MODE_ROW_128B (1) — the idea under test.  The contributor collapses AND
//        transposes IN DEST, so its 32 per-row sums land in ROW 0 (two
//        contiguous 64 B face-rows).  It writes those 2 x 64 B into ROW
//        `slice_index` of the owner's landing tile for that owned row.  The
//        owner combines with ONE reduce<SUM, REDUCE_COL> per owned row over that
//        single tile, transposes back in DEST (so the broadcast operand keeps the
//        column shape `BroadcastDim::Col` needs downstream) and finalizes.
//        Payload per contributor per row: 128 B instead of 4 KB; owner tiles
//        unpacked per block: OWN_ROWS instead of s*OWN_ROWS.
//
//   MODE_COLLAPSE_2K (2) — CONTROL.  Same collapse, no transpose, ship only the
//        two column-0-bearing faces (2 KB) into the BASELINE landing layout.
//        Separates "fewer bytes" from "fewer tiles at the owner": the owner code
//        is byte-identical to the baseline's.
//
//   MODE_ROW_64B_PROBE (3) — ABLATION PROBE, wrong by construction (see writer).
//        Compute path identical to MODE 1.
//
// ---------------------------------------------------------------------------
// RAW-LLK NOTE (why MODE 1 bypasses compute_kernel_lib::reduce)
// ---------------------------------------------------------------------------
// The fused "reduce, then transpose the reduce's own DEST tile" step cannot be
// expressed through `compute_kernel_lib::reduce`'s `post_reduce_op`:
// `transpose_dest` needs `transpose_dest_init` (which rewrites the MATH addrmods
// + MOP and, per llk_math_reduce_uninit's own comment, needs the reduce's SrcA
// ALU format undone first).  The helper calls `reduce_init` ONCE outside its
// output loop and never re-inits per output tile, so a post_reduce_op that
// re-configured MATH would corrupt every later output tile of the same call.
// Hence the hand-rolled window here: reduce_init -> N x reduce_tile ->
// reduce_uninit -> transpose_dest_init -> N x transpose_dest -> finalize ->
// N x pack.  ONE init pair per DEST window, not per tile.
// `ReduceWithinTile::Skip` (the template value that WOULD express "the
// contributor already collapsed this axis") is unreachable — the
// "Skip is AccumulateViaAdd-only" static_assert at
// reduce_helpers_compute.inl:885-891 sits at FUNCTION scope AFTER the
// `if constexpr (AccumulateViaAdd) { ...; return; }` block, so it is not in a
// discarded statement and fires for the very instantiation it means to permit.

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
constexpr uint32_t cb_stat_compact = 10;

constexpr uint32_t MODE_RAW_4K = 0;
constexpr uint32_t MODE_ROW_128B = 1;
constexpr uint32_t MODE_COLLAPSE_2K = 2;
constexpr uint32_t MODE_ROW_64B_PROBE = 3;

// DEST window, in tiles, without fp32 DEST accumulation.
constexpr uint32_t DEST_LIMIT = 8;

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t MODE = get_compile_time_arg_val(3);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t LANDING_ROWS = get_compile_time_arg_val(5);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(6);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(7);

    constexpr bool COMPACT_ROW = (MODE == MODE_ROW_128B) || (MODE == MODE_ROW_64B_PROBE);
    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t CHUNK = BLOCK_ROWS < DEST_LIMIT ? BLOCK_ROWS : DEST_LIMIT;
    static_assert(BLOCK_ROWS % CHUNK == 0, "BLOCK_ROWS must be a whole number of DEST windows");
    constexpr uint32_t OCHUNK = OWN_ROWS < DEST_LIMIT ? OWN_ROWS : DEST_LIMIT;
    static_assert(OWN_ROWS % OCHUNK == 0, "OWN_ROWS must be a whole number of DEST windows");
    constexpr uint32_t GATHER_PAGES = COMPACT_ROW ? (LANDING_ROWS * OWN_ROWS) : (NUM_HIDDEN_SLICES * OWN_ROWS);
    constexpr uint32_t cb_combine_out = (NUM_OWNERS > 1) ? cb_slice_stat : cb_rms_bcast;

    // Same dispatch as the op: pairwise add beats matmul-with-ones past ~4 tiles.
    constexpr auto COMBINE_ALGORITHM =
        NUM_HIDDEN_SLICES >= 4 ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t is_owner = get_arg_val<uint32_t>(1);
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
        if constexpr (MODE == MODE_COLLAPSE_2K) {
            // collapse_partial_block, straight through the helper.  The helper's
            // reduce_init leaves the packer's REDUCE_ROW edge mask set for every
            // pack in the call, so the emitted tile is column-0-valid with ZEROS
            // everywhere else — which is what makes the two-face shipment exact.
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sq_partials,
                cb_scaler,
                cb_stat_compact,
                ckl::ReduceInputPolicy::BulkWaitBulkPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(ckl::ReduceInputBlockShape::of(BLOCK_ROWS, 1));
        } else if constexpr (COMPACT_ROW) {
            // collapse AND transpose, in ONE DEST window per CHUNK tiles.
            cb_wait_front(cb_sq_partials, BLOCK_ROWS);
            cb_reserve_back(cb_stat_compact, BLOCK_ROWS);
            for (uint32_t base = 0; base < BLOCK_ROWS; base += CHUNK) {
                // REDUCE_ROW SUM swaps operands (scaler -> SrcA, data -> SrcB).
                reconfig_data_format(cb_scaler, cb_sq_partials);
                pack_reconfig_data_format(cb_stat_compact);
                tile_regs_acquire();
                reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_sq_partials, cb_scaler, cb_stat_compact);
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
                    pack_tile(i, cb_stat_compact, base + i);
                }
                tile_regs_release();
            }
            cb_push_back(cb_stat_compact, BLOCK_ROWS);
            cb_pop_front(cb_sq_partials, BLOCK_ROWS);
        }

        // ---- combine_block, OWNER half ----
        if (is_owner) {
            if constexpr (COMPACT_ROW) {
                cb_wait_front(cb_gathered_partials, GATHER_PAGES);
                cb_reserve_back(cb_combine_out, OWN_ROWS);
                for (uint32_t base = 0; base < OWN_ROWS; base += OCHUNK) {
                    reconfig_data_format(cb_gathered_partials, cb_scaler);
                    pack_reconfig_data_format(cb_combine_out);
                    tile_regs_acquire();
                    reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
                        cb_gathered_partials, cb_scaler, cb_combine_out);
                    for (uint32_t i = 0; i < OCHUNK; ++i) {
                        // Landing layout: page (ht * OWN_ROWS + j).  ht > 0 only
                        // when s > 32 (more contributors than a tile has rows),
                        // and REDUCE_COL accumulates the ht tiles into one DEST.
                        for (uint32_t ht = 0; ht < LANDING_ROWS; ++ht) {
                            reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>(
                                cb_gathered_partials, cb_scaler, ht * OWN_ROWS + base + i, 0, i);
                        }
                    }
                    reduce_uninit(cb_gathered_partials);
                    // row-0 combined sums -> col 0, so the broadcast operand keeps
                    // the column shape `BroadcastDim::Col` needs downstream.
                    transpose_dest_init<false, true>(cb_gathered_partials);
                    for (uint32_t i = 0; i < OCHUNK; ++i) {
                        transpose_dest<false, true>(i);
                        finalize(i);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < OCHUNK; ++i) {
                        pack_tile(i, cb_combine_out, base + i);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_combine_out, OWN_ROWS);
                cb_pop_front(cb_gathered_partials, GATHER_PAGES);
            } else {
                cb_wait_front(cb_gathered_partials, GATHER_PAGES);
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_gathered_partials,
                    cb_scaler,
                    cb_combine_out,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,
                    COMBINE_ALGORITHM,
                    ckl::NoAccumulation,
                    decltype(finalize)>(
                    ckl::ReduceInputBlockShape::of(OWN_ROWS, NUM_HIDDEN_SLICES),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    finalize);
            }
        }

        cb_pop_front(cb_input_tiles, BLOCK_TILES);
    }
}
