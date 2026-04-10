// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test compute kernel for MatmulOp class.
// Exercises Mode 1, Mode 2, and Mode 3 via compile-time arg selection.
//
// Compile-time args:
//   0: test_mode     (0=mode3_tile, 1=mode2_semi, 2=mode1_tile,
//                      3=mode1_block, 4=mode2_no_spill, 5=mode3_block)
//   1: batch
//   2: Mt            (M tiles, or num_blocks_h for mode3 block)
//   3: Kt            (K tiles, or num_blocks_inner)
//   4: Nt            (N tiles, or num_blocks_w)
//   5: out_subblock_h
//   6: out_subblock_w
//   7: in0_block_w   (kt_dim -- inner dimension block size in tiles)
//   8: in0_num_subblocks
//   9: in1_num_subblocks

#include <cstdint>

#include "api/compute/matmul_op.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reg_api.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t test_mode = get_compile_time_arg_val(0);
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    constexpr uint32_t Mt = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(9);

    // Derived constants
    constexpr uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    constexpr uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    constexpr uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    constexpr uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_partials = tt::CBIndex::c_24;

    // =========================================================================
    // TEST MODE 0: Mode 3 Tile Auto (TileMatmulOp::run)
    // Maps to call site T1 (bmm.cpp)
    // =========================================================================
    if constexpr (test_mode == 0) {
        MatmulOpConfig cfg{
            .in0_cb_id = cb_in0,
            .in1_cb_id = cb_in1,
            .out_cb_id = cb_out,
        };
        TileMatmulOp mm(cfg);
        mm.init();
        mm.run(
            batch,
            Mt,
            Nt,
            Kt,
            1,   // in0_num_subblocks (1 for tile mode)
            1,   // in1_num_subblocks
            1,   // in0_block_num_tiles
            1,   // in1_block_num_tiles
            1);  // in1_block_w
    }

    // =========================================================================
    // TEST MODE 1: Mode 2 Semi-Automatic with Spill/Reload (BlockMatmulOp)
    // Maps to call sites B1/B2/B3/B16 (bmm_large_block_zm_fused*)
    // =========================================================================
    else if constexpr (test_mode == 1) {
        constexpr uint32_t num_blocks_inner = Kt;  // Kt = num_blocks_inner for this mode

        MatmulOpConfig cfg{
            .in0_cb_id = cb_in0,
            .in1_cb_id = cb_in1,
            .out_cb_id = cb_out,
            .ct_dim = out_subblock_w,
            .rt_dim = out_subblock_h,
            .kt_dim = in0_block_w,
            .partials_cb_id = cb_partials,
        };
        BlockMatmulOp mm(cfg);
        mm.init();

        bool enable_reload = false;

        for (uint32_t block = 0; block < num_blocks_inner; block++) {
            bool last_out = (block == (num_blocks_inner - 1));

            cb_wait_front(cb_in0, in0_block_num_tiles);
            cb_wait_front(cb_in1, in1_block_num_tiles);

            uint32_t in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                uint32_t in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    mm.begin_subblock();

                    if (enable_reload) {
                        mm.reload_partials(out_subblock_num_tiles);
                    }

                    mm.accumulate(
                        in0_index_subblock_offset, in1_index_subblock_offset, 0, in0_block_w, 1, in1_per_core_w, 0);

                    if (last_out) {
                        mm.end_to_output(cb_out, out_subblock_num_tiles);
                    } else {
                        mm.end_to_partials(out_subblock_num_tiles);
                    }

                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if (num_blocks_inner > 1) {
                enable_reload = true;
            }

            cb_pop_front(cb_in0, in0_block_num_tiles);
            cb_pop_front(cb_in1, in1_block_num_tiles);
        }
    }

    // =========================================================================
    // TEST MODE 2: Mode 1 Tile-Level Single Tile (TileMatmulOp::matmul)
    // Maps to call sites T5/T6/T7 (simple tile matmul in reduction pipeline)
    // =========================================================================
    else if constexpr (test_mode == 2) {
        MatmulOpConfig cfg{
            .in0_cb_id = cb_in0,
            .in1_cb_id = cb_in1,
            .out_cb_id = cb_out,
        };
        TileMatmulOp mm(cfg);
        mm.init();

        tile_regs_acquire();

        // Accumulate over K inner tiles
        for (uint32_t k = 0; k < Kt; k++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            mm.accumulate(0, 0, 0, 1, 0, 0, 0);

            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        // Pack result
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }

    // =========================================================================
    // TEST MODE 3: Mode 1 Block-Level (BlockMatmulOp::accumulate)
    // Maps to call sites B8/B11 (MOE gate with ct_dim > 1)
    // =========================================================================
    else if constexpr (test_mode == 3) {
        // For this mode: out_subblock_w = ct_dim, out_subblock_h = rt_dim, in0_block_w = kt_dim
        constexpr uint32_t ct_dim = out_subblock_w;
        constexpr uint32_t rt_dim = out_subblock_h;
        constexpr uint32_t kt_dim = in0_block_w;
        constexpr uint32_t block_in0_tiles = rt_dim * kt_dim;
        constexpr uint32_t block_in1_tiles = kt_dim * ct_dim;
        constexpr uint32_t block_out_tiles = rt_dim * ct_dim;

        MatmulOpConfig cfg{
            .in0_cb_id = cb_in0,
            .in1_cb_id = cb_in1,
            .out_cb_id = cb_out,
            .ct_dim = ct_dim,
            .rt_dim = rt_dim,
            .kt_dim = kt_dim,
        };
        BlockMatmulOp mm(cfg);
        mm.init();

        tile_regs_acquire();

        // Accumulate over K blocks
        for (uint32_t k = 0; k < Kt; k++) {
            cb_wait_front(cb_in0, block_in0_tiles);
            cb_wait_front(cb_in1, block_in1_tiles);

            mm.accumulate(0, 0, 0, 1, 0, 0, 0);

            cb_pop_front(cb_in0, block_in0_tiles);
            cb_pop_front(cb_in1, block_in1_tiles);
        }

        // Pack result
        tile_regs_commit();
        cb_reserve_back(cb_out, block_out_tiles);
        tile_regs_wait();
        for (uint32_t i = 0; i < block_out_tiles; i++) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, block_out_tiles);
    }

    // =========================================================================
    // TEST MODE 4: Mode 2 No Spill (BlockMatmulOp, single inner block)
    // Maps to call sites B4/B5 (SDPA matmul_blocks, no spill)
    // =========================================================================
    else if constexpr (test_mode == 4) {
        MatmulOpConfig cfg{
            .in0_cb_id = cb_in0,
            .in1_cb_id = cb_in1,
            .out_cb_id = cb_out,
            .ct_dim = out_subblock_w,
            .rt_dim = out_subblock_h,
            .kt_dim = in0_block_w,
        };
        BlockMatmulOp mm(cfg);
        mm.init();

        cb_wait_front(cb_in0, in0_block_num_tiles);
        cb_wait_front(cb_in1, in1_block_num_tiles);

        uint32_t in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            uint32_t in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                mm.begin_subblock();

                mm.accumulate(
                    in0_index_subblock_offset, in1_index_subblock_offset, 0, in0_block_w, 1, in1_per_core_w, 0);

                mm.end_to_output(cb_out, out_subblock_num_tiles);

                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        cb_pop_front(cb_in0, in0_block_num_tiles);
        cb_pop_front(cb_in1, in1_block_num_tiles);
    }

    // =========================================================================
    // TEST MODE 5: Mode 3 Block Auto (BlockMatmulOp::run)
    // Maps to call sites B9/B10/B15 (minimal_matmul, conv3d)
    // =========================================================================
    else if constexpr (test_mode == 5) {
        constexpr uint32_t num_blocks_inner = Kt;
        constexpr uint32_t num_blocks_h = 1;
        constexpr uint32_t num_blocks_w = 1;

        MatmulOpConfig cfg{
            .in0_cb_id = cb_in0,
            .in1_cb_id = cb_in1,
            .out_cb_id = cb_out,
            .ct_dim = out_subblock_w,
            .rt_dim = out_subblock_h,
            .kt_dim = in0_block_w,
            .partials_cb_id = (num_blocks_inner > 1) ? cb_partials : 0u,
        };
        BlockMatmulOp mm(cfg);
        mm.init();
        mm.run(
            batch,
            num_blocks_h,
            num_blocks_w,
            num_blocks_inner,
            in0_num_subblocks,
            in1_num_subblocks,
            in0_block_num_tiles,
            in1_block_num_tiles,
            in1_per_core_w);
    }
}
