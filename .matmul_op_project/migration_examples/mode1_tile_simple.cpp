// Migration Examples: Mode 1 (Low-Level Tile) -- Simple patterns
// Call sites: T5, T6, T7, T8, T9
//
// These call sites use matmul_tiles as a building block within larger compute
// pipelines (reduction, moreh operations). MatmulOp provides the matmul() call
// and init/init_short methods; the caller manages all DST, CB, and pack ops.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// T5: reduce_w.cpp -- Width reduction via matmul with scaler tile
// Source: ttnn/.../reduction/.../reduce_w.cpp
//
// ORIGINAL CODE (REDUCE_ROW_SUM_VIA_MM path):
//   mm_init(c_0, c_2, c_3);
//   cb2.wait_front(1);  // scaler tile
//   for (nc) for (ht) {
//       acquire_dst();
//       for (wt) { cb0.wait_front(1); matmul_tiles(c_0, c_2, 0, 0, 0); cb0.pop_front(1); }
//       cb3.reserve_back(1); pack_tile(0, c_3); cb3.push_back(1);
//       release_dst();
//   }
//
// The matmul is used as a row reduction tool: multiply each input tile by a
// scaler tile (all 1s) to sum rows. This is a simple accumulation pattern.
// ============================================================================
namespace t5_reduce_w {

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb3(tt::CBIndex::c_3);

    // --- NEW: MatmulOp replaces mm_init + matmul_tiles ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = tt::CBIndex::c_0,
        .in1_cb_id = tt::CBIndex::c_2,
        .out_cb_id = tt::CBIndex::c_3,
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init();  // replaces mm_init(c_0, c_2, c_3)
    // --- END NEW ---

    cb2.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            acquire_dst();
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb0.wait_front(onetile);
                // --- NEW: mm.matmul replaces matmul_tiles ---
                mm.matmul(0, 0, 0);
                // --- END NEW ---
                cb0.pop_front(onetile);
            }
            cb3.reserve_back(onetile);
            pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
            cb3.push_back(onetile);
            release_dst();
        }
    }
}

}  // namespace t5_reduce_w

// ============================================================================
// T6: moreh_matmul -- single tile matmul with transpose/mask (first code path)
// Source: ttnn/.../moreh/moreh_matmul/.../moreh_matmul.cpp (line 283)
//
// ORIGINAL CODE (inside the matmul_with_transpose_and_mask function):
//   tile_regs_acquire();
//   if (enable_reload) { copy_tile_to_dst_init_short(cb_intermed0); copy_tile(...); }
//   mm_init_short(mm_src0, mm_src1);
//   matmul_tiles(mm_src0, mm_src1, 0, 0, 0);
//   tile_regs_commit();
//   // ... pack to cb_intermed0 or cb_out0 ...
//
// This is a single-tile matmul inside a complex pipeline with mask/transpose
// preprocessing. The mm_src0/mm_src1 may change each iteration (depending on
// whether the input was transposed/masked into a scratch CB).
// ============================================================================
namespace t6_moreh_matmul_single {

void matmul_with_transpose_and_mask_snippet(
    uint32_t mm_src0, uint32_t mm_src1, uint32_t cb_intermed0, uint32_t cb_out0, bool enable_reload, bool last_out) {
    // --- NEW: MatmulOp for single-tile matmul ---
    // NOTE: in0_cb_id/in1_cb_id may vary per iteration (due to mask/transpose),
    // so we construct a fresh config each time. The overhead is negligible since
    // MatmulOpConfig is a plain struct with no heap allocation.
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = mm_src0,
        .in1_cb_id = mm_src1,
        .out_cb_id = cb_out0,
    };
    ckernel::TileMatmulOp mm(cfg);
    // --- END NEW ---

    tile_regs_acquire();
    if (enable_reload) {
        cb_wait_front(cb_intermed0, 1);
#if defined FP32_DEST_ACC_EN
        reconfig_data_format_srca(cb_intermed0);
#endif
        copy_tile_to_dst_init_short(cb_intermed0);
        copy_tile(cb_intermed0, 0, 0);
        cb_pop_front(cb_intermed0, 1);
    }

#if defined FP32_DEST_ACC_EN
    reconfig_data_format(mm_src0, mm_src1);
#endif
    // --- NEW: init_short + matmul replaces mm_init_short + matmul_tiles ---
    mm.init_short();
    mm.matmul(0, 0, 0);
    // --- END NEW ---
    tile_regs_commit();

    // UNCHANGED: pop inputs, pack to output or partials
    cb_pop_front(mm_src0, 1);  // simplified; actual code has conditional pops
    cb_pop_front(mm_src1, 1);

    if (last_out) {
        // pack to cb_out0
    } else {
        // pack to cb_intermed0
    }
}

}  // namespace t6_moreh_matmul_single

// ============================================================================
// T7: moreh_matmul -- simple K loop (second code path)
// Source: ttnn/.../moreh/moreh_matmul/.../moreh_matmul.cpp (lines 316-330)
//
// ORIGINAL CODE:
//   mm_init(cb_in0, cb_in1, cb_out0);
//   for (i < num_output_tiles) {
//       tile_regs_acquire();
//       for (kt < Kt) {
//           cb_wait_front(cb_in0, 1); cb_wait_front(cb_in1, 1);
//           matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
//           cb_pop_front(cb_in0, 1); cb_pop_front(cb_in1, 1);
//       }
//       tile_regs_commit();
//       pack_onetile_to_cb(cb_out0);
//   }
//
// Simple: M output tiles, each accumulating K inner tiles. Nearly the same
// as T1, but with per-tile CB pop instead of block-level.
// ============================================================================
namespace t7_moreh_matmul_simple {

void matmul_snippet(uint32_t num_output_tiles, uint32_t Kt) {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr int onetile = 1;

    // --- NEW: MatmulOp replaces mm_init + matmul_tiles ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_in0,
        .in1_cb_id = cb_in1,
        .out_cb_id = cb_out0,
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init();  // replaces mm_init(cb_in0, cb_in1, cb_out0)
    // --- END NEW ---

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_in0, onetile);
            cb_wait_front(cb_in1, onetile);
            // --- NEW: mm.matmul replaces matmul_tiles ---
            mm.matmul(0, 0, 0);
            // --- END NEW ---
            cb_pop_front(cb_in0, onetile);
            cb_pop_front(cb_in1, onetile);
        }
        tile_regs_commit();
        // UNCHANGED: pack_onetile_to_cb(cb_out0) -- helper that does
        // reserve_back, tile_regs_wait, pack_tile, tile_regs_release, push_back
    }
}

}  // namespace t7_moreh_matmul_simple

// ============================================================================
// T8: moreh_mean_w -- Width reduction with matmul + masking
// Source: ttnn/.../moreh/moreh_mean/.../moreh_mean_w.cpp
//
// ORIGINAL CODE (two matmul_tiles call sites):
//   Call site 1 (line 56, accumulation loop Wt-1 tiles):
//     mm_init_short(cb_input, cb_scaler, false);
//     matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
//
//   Call site 2 (line 105, last tile after optional masking):
//     mm_init_short(cb_input, cb_scaler, false);
//     matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
//
// Both use the same pattern: mm_init_short before each matmul_tiles call
// (because mask/copy operations in between change the unpack config).
// The cb_input may be different (c_0 or cb_masked_input) depending on masking.
// ============================================================================
namespace t8_moreh_mean_w {

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);

    auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask_w = tt::CBIndex::c_3;
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr bool do_mask_w = (origin_W % 32) != 0;
    constexpr int onetile = 1;
    int reduce_dst_idx = 0;

    binary_op_init_common(cb_input, cb_input, cb_out);
    cb_wait_front(cb_scaler, 1);

    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_input = tt::CBIndex::c_0;
            bool is_w_single_tile = (Wt == 1);

            // Call site 1: accumulate Wt-1 tiles
            if (!is_w_single_tile) {
                tile_regs_acquire();

                // --- NEW: MatmulOp for accumulation loop ---
                // Constructed here because cb_input is known at this point.
                ckernel::MatmulOpConfig cfg1{
                    .in0_cb_id = cb_input,
                    .in1_cb_id = cb_scaler,
                    .out_cb_id = cb_out,
                };
                ckernel::TileMatmulOp mm1(cfg1);
                // --- END NEW ---

                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    cb_wait_front(cb_input, onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, cb_scaler);
#endif
                    // --- NEW: init_short + matmul replaces mm_init_short + matmul_tiles ---
                    mm1.init_short();
                    mm1.matmul(0, 0, reduce_dst_idx);
                    // --- END NEW ---
                    cb_pop_front(cb_input, onetile);
                }
                tile_regs_commit();

                cb_reserve_back(cb_accum_dst, onetile);
                tile_regs_wait();
                pack_tile_with_dt(reduce_dst_idx, cb_accum_dst);
                tile_regs_release();
                cb_push_back(cb_accum_dst, onetile);
            }

            // UNCHANGED: Optional masking phase (copy, mask_tile)
            if (do_mask_w) {
                // ... mask processing ...
                cb_input = cb_masked_input;
            }

            // Call site 2: final tile (possibly masked)
            tile_regs_acquire();
            cb_wait_front(cb_input, onetile);
            if (!is_w_single_tile) {
                cb_wait_front(cb_accum_dst, onetile);
                copy_tile_to_dst_init_short(cb_accum_dst);
                copy_tile(cb_accum_dst, 0, reduce_dst_idx);
            }

            // --- NEW: MatmulOp for final tile ---
            // cb_input may have changed due to masking, so we use updated value.
            ckernel::MatmulOpConfig cfg2{
                .in0_cb_id = cb_input,
                .in1_cb_id = cb_scaler,
                .out_cb_id = cb_out,
            };
            ckernel::TileMatmulOp mm2(cfg2);
            // --- END NEW ---

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_input, cb_scaler);
#endif
            // --- NEW: init_short + matmul ---
            mm2.init_short();
            mm2.matmul(0, 0, reduce_dst_idx);
            // --- END NEW ---
            tile_regs_commit();

            cb_reserve_back(cb_out, onetile);
            tile_regs_wait();
            pack_tile_with_dt(reduce_dst_idx, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, onetile);

            cb_pop_front(cb_input, onetile);
            if (!is_w_single_tile) {
                cb_pop_front(cb_accum_dst, onetile);
            }
        }
    }
}

}  // namespace t8_moreh_mean_w

// ============================================================================
// T9: moreh_sum_w -- Width reduction with matmul + masking
// Source: ttnn/.../moreh/moreh_sum/.../moreh_sum_w.cpp
//
// This kernel is structurally identical to T8 (moreh_mean_w). The only
// difference is the semantic (sum vs mean) and minor FP32_DEST_ACC_EN handling.
// The matmul migration is exactly the same pattern as T8.
//
// MIGRATION: Identical to T8 above. See T8 for the complete pattern.
// ============================================================================
namespace t9_moreh_sum_w {
// Migration is identical to T8. Both use the same two-call-site pattern:
// 1. Accumulation loop for Wt-1 tiles
// 2. Final tile after optional masking
// Each call site: mm.init_short() + mm.matmul(0, 0, reduce_dst_idx)
}  // namespace t9_moreh_sum_w
