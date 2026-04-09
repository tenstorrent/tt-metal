// Migration Examples: Mode 1 (Low-Level Tile) -- Complex patterns
// Call sites: T2, T3, T4, T13, T14
//
// These call sites use matmul_tiles with complex control flow: subblock
// tiling with per-tile index arithmetic, interleaved untilize/retilize,
// architecture-conditional paths, and multi-step pipelines.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "api/compute/tile_move_copy.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// T2: bmm_large_block_zm.cpp -- tile-mode subblocked matmul with spill/reload
// Source: ttnn/.../matmul/.../bmm_large_block_zm.cpp
//
// ORIGINAL CODE:
//   mm_init(cb_in0, cb_in1, cb_intermed0);
//   for (batch) for (num_blocks) {
//     wait_front(in0_block); wait_front(in1_block);
//     for (in0_sub) for (in1_sub) {
//       acquire_dst();
//       if (enable_reload) { copy_tile_to_dst_init_short_with_dt(...); copy_tile loop; mm_init_short_with_dt(...); }
//       for (h) for (w) for (inner_dim) {
//           in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
//           in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
//           matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index);
//           in1_index_inner_dim_offset += in1_per_core_w;
//       }
//       // pack to output or partials
//       release_dst();
//     }
//   }
//
// The h*w*inner_dim tile indexing pattern does NOT map cleanly to accumulate()
// because the in0/in1 stride pattern is per-tile with complex offsets.
// Mode 1 matmul() is the right choice here.
// ============================================================================
namespace t2_bmm_large_block_zm {

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    uint32_t num_blocks = get_compile_time_arg_val(7);
    uint32_t out_subblock_h = get_compile_time_arg_val(8);
    uint32_t out_subblock_w = get_compile_time_arg_val(9);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);
    uint32_t batch = get_compile_time_arg_val(11);

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_intermed0 = get_named_compile_time_arg_val("cb_intermed0");

    experimental::CircularBuffer in0_cb(cb_in0);
    experimental::CircularBuffer in1_cb(cb_in1);
    experimental::CircularBuffer out_cb(cb_out);
    experimental::CircularBuffer intermed0_cb(cb_intermed0);

    // --- NEW: MatmulOp replaces mm_init ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_in0,
        .in1_cb_id = cb_in1,
        .out_cb_id = cb_intermed0,  // init targets the intermed CB
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init();  // replaces mm_init(cb_in0, cb_in1, cb_intermed0)
    // --- END NEW ---

    for (uint32_t b = 0; b < batch; b++) {
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            bool last_out = block == (num_blocks - 1);

            in0_cb.wait_front(in0_block_num_tiles);
            in1_cb.wait_front(in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    acquire_dst();

                    // UNCHANGED: Reload from partials
                    if (enable_reload) {
                        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_intermed0);
                        intermed0_cb.wait_front(out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(cb_intermed0, i, i);
                        }
                        intermed0_cb.pop_front(out_subblock_num_tiles);
                        // --- NEW: init_short_with_dt replaces mm_init_short_with_dt ---
                        mm.init_short_with_dt(cb_intermed0);
                        // --- END NEW ---
                    }

                    // Compute output sub-block -- h*w*inner_dim tile indexing
                    int dst_index = 0;
                    int in0_index_h_offset = 0;
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            int in1_index_inner_dim_offset = 0;
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                // --- NEW: mm.matmul replaces matmul_tiles ---
                                mm.matmul(in0_index, in1_index, dst_index);
                                // --- END NEW ---
                                in1_index_inner_dim_offset += in1_per_core_w;
                            }
                            dst_index++;
                        }
                        in0_index_h_offset += in0_block_w;
                    }

                    // UNCHANGED: Pack to output or partials
                    if (last_out) {
                        out_cb.reserve_back(out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_out);
                        }
                        out_cb.push_back(out_subblock_num_tiles);
                    } else {
                        if (block == 0) {
                            out_cb.reserve_back(out_num_tiles_to_wait);
                            out_num_tiles_to_wait += out_subblock_num_tiles;
                        }
                        intermed0_cb.reserve_back(out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_intermed0);
                        }
                        intermed0_cb.push_back(out_subblock_num_tiles);
                    }

                    release_dst();
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (spill) {
                enable_reload = true;
            }

            in0_cb.pop_front(in0_block_num_tiles);
            in1_cb.pop_front(in1_block_num_tiles);
        }
    }
}

}  // namespace t2_bmm_large_block_zm

// ============================================================================
// T3: transformer_attn_matmul.cpp -- Per-row matmul with untilize/retilize
// Source: ttnn/.../experimental/matmul/attn_matmul/.../transformer_attn_matmul.cpp
//
// ORIGINAL CODE:
//   mm_init(cb_in0, cb_in1, cb_intermed0, transpose_hw);
//   for (batch) for (Mt) for (Nt) for (tile_row=0..31) {
//       tile_regs_acquire();
//       for (Kt) { wait_front; matmul_tiles(cb_in0, cb_in1, kt, 0, 0); pop_front; }
//       tile_regs_commit();
//       pack_tile(0, cb_intermed0);
//       // UNTILIZE (changes data format config)
//       mm_init_short_with_dt(cb_in0, cb_in1, cb_intermed0, transpose_hw);
//   }
//   // TILIZE output
//   mm_block_init_short_with_both_dt(cb_in0, cb_in1, ...);
//
// Mode 1 is required because of interleaved untilize/tilize operations.
// ============================================================================
namespace t3_transformer_attn_matmul {

void kernel_main() {
    constexpr uint32_t transpose_hw = get_compile_time_arg_val(0);
    uint32_t batch = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_intermed0 = tt::CBIndex::c_2;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_5;

    // --- NEW: MatmulOp replaces mm_init ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_in0,
        .in1_cb_id = cb_in1,
        .out_cb_id = cb_intermed0,
        .transpose = static_cast<bool>(transpose_hw),
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init();  // replaces mm_init(cb_in0, cb_in1, cb_intermed0, transpose_hw)
    // --- END NEW ---

    constexpr uint32_t num_rows_in_one_tile = 32;

    for (uint32_t nb = 0; nb < batch; ++nb) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) {
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; ++tile_row_id) {
                    tile_regs_acquire();
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        if (tile_row_id == 0) {
                            cb_wait_front(cb_in0, kt + 1);
                        }
                        cb_wait_front(cb_in1, 1);

                        // --- NEW: mm.matmul replaces matmul_tiles ---
                        mm.matmul(kt, 0, 0);
                        // --- END NEW ---

                        cb_pop_front(cb_in1, 1);
                    }
                    tile_regs_commit();

                    cb_reserve_back(cb_intermed0, 1);
                    tile_regs_wait();
                    pack_tile(0, cb_intermed0);
                    tile_regs_release();
                    cb_push_back(cb_intermed0, 1);

                    // UNCHANGED: Untilize then retilize
                    // untilize<...>(...);

                    // --- NEW: init_short_with_dt replaces mm_init_short_with_dt ---
                    mm.init_short_with_dt(cb_intermed0);
                    // --- END NEW ---
                }
                cb_pop_front(cb_in0, Kt);

                // UNCHANGED: Tilize output
                // tilize<...>(...);

                // Reinit for next output tile
                pack_reconfig_data_format(out_cb_id, cb_intermed0);
                // NOTE: mm_block_init_short_with_both_dt is a block-mode call
                // that this kernel uses for data format reconfig only. This is
                // an existing quirk of the kernel, not something MatmulOp changes.
            }
        }
    }
}

}  // namespace t3_transformer_attn_matmul

// ============================================================================
// T4: transformer_group_attn_matmul.cpp -- Arch-conditional tile matmul
// Source: ttnn/.../experimental/matmul/group_attn_matmul/.../transformer_group_attn_matmul.cpp
//
// Uses the same h*w*inner_dim tile indexing pattern as T2, but with
// pack_untilize_dest instead of normal pack. Mode 1 is required.
//
// ORIGINAL CODE:
//   mm_init(cb_in0, cb_in1, cb_intermed0, transpose_hw);
//   // h*w*inner_dim loop with matmul_tiles
//   // pack_untilize_dest at end of each subblock
//
// MIGRATION: Same as T2 for the matmul part, with the addition of
// pack_untilize_dest (unchanged).
// ============================================================================
namespace t4_group_attn_matmul {

void kernel_main_snippet() {
    constexpr uint32_t transpose_hw = 0;  // from compile time args
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_intermed0 = tt::CBIndex::c_2;

    uint32_t in0_block_w = 1;     // from compile time args
    uint32_t out_subblock_h = 1;  // from compile time args
    uint32_t out_subblock_w = 1;  // from compile time args
    uint32_t in1_per_core_w = 1;  // from compile time args

    // --- NEW: MatmulOp replaces mm_init ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_in0,
        .in1_cb_id = cb_in1,
        .out_cb_id = cb_intermed0,
        .transpose = static_cast<bool>(transpose_hw),
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init();
    // --- END NEW ---

    // Inside the subblock loop:
    uint32_t in0_index_subblock_offset = 0;
    uint32_t in1_index_subblock_offset = 0;

    tile_regs_acquire();

    // h*w*inner_dim matmul loop (same as T2)
    int dst_index = 0;
    int in0_index_h_offset = 0;
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        for (uint32_t w = 0; w < out_subblock_w; w++) {
            int in1_index_inner_dim_offset = 0;
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                // --- NEW: mm.matmul replaces matmul_tiles ---
                mm.matmul(in0_index, in1_index, dst_index);
                // --- END NEW ---
                in1_index_inner_dim_offset += in1_per_core_w;
            }
            dst_index++;
        }
        in0_index_h_offset += in0_block_w;
    }

    tile_regs_commit();
    tile_regs_wait();
    // UNCHANGED: pack_untilize_dest<...>(cb_intermed0, ...);
    tile_regs_release();
}

}  // namespace t4_group_attn_matmul

// ============================================================================
// T13: bmm_tilize_untilize.cpp -- 6-loop nest with tilize/untilize/bias/SFPU
// Source: ttnn/kernel/compute/bmm_tilize_untilize.cpp
//
// ORIGINAL CODE:
//   for batch, in1_block_w_i, in0_block_w_i:
//     if (tilize_in0) { tilize; mm_init_short_with_dt(...); }
//     for in0_sub, in1_sub:
//       tile_regs_acquire();
//       if (enable_reload) { copy_tile_to_dst; mm_init_short_with_dt(...); }
//       for (h) for (w) for (inner_dim) {
//           matmul_tiles(in0_cb, in1_cb, ..., dst_index);
//       }
//       // optional bias, SFPU activation
//       // pack
//       tile_regs_release();
//
// Mode 1 is required due to:
//   - Interleaved tilize preprocessing
//   - Complex bias fusion path (add_tiles_bcast_rows)
//   - SFPU activation
//   - Custom mm_init_short_with_dt calls for data format reconfig
// ============================================================================
namespace t13_bmm_tilize_untilize {

void kernel_main_snippet() {
    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t tilized_in0_cb_id = tt::CBIndex::c_6;  // from compile time args
    constexpr uint32_t matmul_partials_cb = tt::CBIndex::c_24;
    constexpr bool tilize_in0 = true;  // from compile time args

    uint32_t in0_block_w = 1;
    uint32_t out_subblock_h = 1;
    uint32_t out_subblock_w = 1;
    uint32_t in1_block_w = 1;

    // --- NEW: MatmulOp for the tile-mode matmul ---
    // When tilize_in0 is true, the actual in0 CB is tilized_in0_cb_id.
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = tilize_in0 ? tilized_in0_cb_id : in0_cb_id,
        .in1_cb_id = in1_cb_id,
        .out_cb_id = matmul_partials_cb,
    };
    ckernel::TileMatmulOp mm(cfg);
    // NOTE: init() is called after tilize configuration setup (not shown)
    // --- END NEW ---

    // Inside the subblock loop:
    uint32_t in0_index_subblock_offset = 0;
    uint32_t in1_index_subblock_offset = 0;

    tile_regs_acquire();

    // UNCHANGED: reload from partials if needed
    // if (enable_reload) { copy_tile_to_dst_init_short; ... mm.init_short_with_dt(matmul_partials_cb); }

    // h*w*inner_dim matmul loop
    int dst_index = 0;
    int in0_index_h_offset = 0;
    for (uint32_t h = 0; h < out_subblock_h; ++h) {
        for (uint32_t w = 0; w < out_subblock_w; ++w) {
            int in1_index_inner_dim_offset = 0;
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                // --- NEW: mm.matmul replaces matmul_tiles ---
                mm.matmul(
                    in0_index_subblock_offset + in0_index_h_offset + inner_dim,
                    in1_index_subblock_offset + in1_index_inner_dim_offset + w,
                    dst_index);
                // --- END NEW ---
                in1_index_inner_dim_offset += in1_block_w;
            }
            ++dst_index;
        }
        in0_index_h_offset += in0_block_w;
    }

    // UNCHANGED: Bias fusion, SFPU activation, pack (all after matmul)
}

}  // namespace t13_bmm_tilize_untilize

// ============================================================================
// T14: DeepSeek V3 RoPE -- matmul as step 1 of a 4-step pipeline
// Source: models/demos/deepseek_v3_b1/unified_kernels/rope.hpp (lines 140-240)
//
// ORIGINAL CODE:
//   mm_init_short(args.in_cb, args.trans_mat_cb);
//   tile_regs_acquire();
//   for (j < Wt) { matmul_tiles(args.in_cb, args.trans_mat_cb, j, 0, j); }
//   tile_regs_commit(); tile_regs_wait();
//   for (j < Wt) { pack_tile(j, args.rotated_in_interm_cb, j); }
//   tile_regs_release();
//
// This is a clean Mode 1 case: matmul is one step in a 4-step pipeline
// (matmul, mul_bcast, mul_bcast, add). The matmul accumulates Wt input tiles
// multiplied by the same transformation matrix tile.
// ============================================================================
namespace t14_rope {

void rope_compute_snippet(
    uint32_t in_cb,
    uint32_t trans_mat_cb,
    uint32_t rotated_in_interm_cb,
    uint32_t cos_sin_cb,
    uint32_t cos_sin_interm_cb,
    uint32_t out_cb,
    uint32_t Wt,
    uint32_t Ht) {
    // --- NEW: MatmulOp for RoPE step 1 ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = in_cb,
        .in1_cb_id = trans_mat_cb,
        .out_cb_id = rotated_in_interm_cb,
    };
    ckernel::TileMatmulOp mm(cfg);
    // --- END NEW ---

    for (uint32_t ht = 0; ht < Ht; ht++) {
        cb_reserve_back(rotated_in_interm_cb, Wt);
        cb_reserve_back(cos_sin_interm_cb, Wt * 2);
        cb_reserve_back(out_cb, Wt);
        cb_wait_front(in_cb, Wt);

        // ============================================================
        // Step 1: rotated = input @ trans_mat (matmul for rotate_half)
        // ============================================================
        // --- NEW: init_short + matmul replaces mm_init_short + matmul_tiles ---
        mm.init_short();
        tile_regs_acquire();
        for (uint32_t j = 0; j < Wt; ++j) {
            mm.matmul(j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Wt; ++j) {
            pack_tile(j, rotated_in_interm_cb, j);
        }
        tile_regs_release();
        // --- END NEW ---

        cb_push_back(rotated_in_interm_cb, Wt);
        cb_wait_front(rotated_in_interm_cb, Wt);

        // UNCHANGED: Steps 2-4 (mul_bcast sin, mul_bcast cos, add)
        // ...
    }
}

}  // namespace t14_rope
