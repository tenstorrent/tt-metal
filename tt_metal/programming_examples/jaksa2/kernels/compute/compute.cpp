// SPDX-FileCopyrightText: © 2025 Tenstorre fAI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t dbg_block = 8;

    uint32_t num_blocks = get_compile_time_arg_val(0);
    uint32_t num_subblocks_h = get_compile_time_arg_val(1);
    uint32_t num_subblocks_w = get_compile_time_arg_val(2);
    uint32_t out_subblock_h = get_compile_time_arg_val(3);
    uint32_t out_subblock_w = get_compile_time_arg_val(4);
    uint32_t in0_block_w = get_compile_time_arg_val(5);

    uint32_t full_block_tiles = out_subblock_h * out_subblock_w * num_subblocks_h * num_subblocks_w;
    uint32_t out_subblock_tiles = out_subblock_h * out_subblock_w;

    constexpr auto cb_in_a = tt::CBIndex::c_0;
    constexpr auto cb_in_b = tt::CBIndex::c_1;
    constexpr auto cb_in_c = tt::CBIndex::c_2;

    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_interm = tt::CBIndex::c_24;

    mm_init(cb_in_a, cb_in_b, cb_out);

    for (uint32_t block = 0; block < num_blocks; block++) {
        bool last_block = block == (num_blocks - 1);

        // DPRINT_MATH(DPRINT << "compute - block: " << block << " - waiting A and B data " << ENDL());
        cb_wait_front(cb_in_a, in0_block_w * out_subblock_h * num_subblocks_h);
        cb_wait_front(cb_in_b, in0_block_w * out_subblock_w * num_subblocks_w);

        // if (block == dbg_block) {
        //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - got A and B data " << ENDL());
        // }
        for (uint32_t subblock_h = 0; subblock_h < num_subblocks_h; subblock_h++) {
            for (uint32_t subblock_w = 0; subblock_w < num_subblocks_w; subblock_w++) {
                if (block == dbg_block && subblock_w == 1) {
                    DPRINT_MATH(DPRINT << "compute - block: " << block << " - starting subblock " << ENDL());
                }
                tile_regs_acquire();

                if (block == dbg_block && subblock_w == 1) {
                    DPRINT_MATH(DPRINT << "compute - block: " << block << " - waiting interm data " << ENDL());
                }
                if (block > 0) {
                    copy_tile_to_dst_init_short(cb_interm);
                    cb_wait_front(cb_interm, out_subblock_tiles);
                    for (uint32_t i = 0; i < out_subblock_tiles; ++i) {
                        copy_tile(cb_interm, i, i);
                    }
                    cb_pop_front(cb_interm, out_subblock_tiles);
                    mm_init_short(cb_in_a, cb_in_b);
                }

                if (block == dbg_block && subblock_w == 1) {
                    DPRINT_MATH(DPRINT << "compute - block: " << block << " - starting " << ENDL());
                }
                uint32_t dst_index = 0;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                            // if (block == dbg_block) {
                            //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - matmul_tiles: CB0 tile: "
                            //     << in0_block_w * (subblock_h * out_subblock_h + h) + inner_dim << ", CB1 tile: " <<
                            //     inner_dim * out_subblock_w * num_subblocks_w + subblock_w * out_subblock_w + w <<
                            //     ENDL());
                            // }
                            matmul_tiles(
                                cb_in_a,
                                cb_in_b,
                                in0_block_w * (subblock_h * out_subblock_h + h) + inner_dim,  // one row -> in0_block_w
                                inner_dim * out_subblock_w * num_subblocks_w + subblock_w * out_subblock_w +
                                    w,  // one row -> out_subblock_w * num_subblocks_w
                                dst_index);
                        }
                        dst_index++;
                    }
                }
                // if (block == dbg_block) {
                //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - calcualted " << dst_index << "tiles
                //     into DST" << ENDL());
                // }

                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(cb_interm, out_subblock_tiles);
                for (uint32_t i = 0; i < out_subblock_tiles; i++) {
                    pack_tile(i, cb_interm);
                }
                cb_push_back(cb_interm, out_subblock_tiles);

                if (block == dbg_block) {
                    DPRINT_MATH(
                        DPRINT << "compute - block: " << block << " - sent " << dst_index << "tiles from DST to CB24"
                               << ENDL());
                }

                if (last_block) {
                    tile_regs_release();

                    binary_op_init_common(cb_interm, cb_in_c, cb_out);
                    add_tiles_init(cb_interm, cb_in_c);

                    cb_wait_front(cb_interm, full_block_tiles);  // per_core_M x per_core_N
                    cb_wait_front(cb_in_c, full_block_tiles);    // per_core_M x per_core_N

                    // if (block == dbg_block) {
                    //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - got INTERMC and C data " <<
                    //     ENDL());
                    // }

                    tile_regs_acquire();
                    for (uint32_t m = 0; m < out_subblock_h * num_subblocks_h; m++) {
                        for (uint32_t n = 0; n < out_subblock_w * num_subblocks_w; n++) {
                            add_tiles(
                                cb_interm,
                                cb_in_c,
                                m * out_subblock_w * num_subblocks_w +
                                    n,  // one row -> out_subblock_w * num_subblocks_w
                                m * out_subblock_w * num_subblocks_w +
                                    n,                                    // one row -> out_subblock_w * num_subblocks_w
                                m * out_subblock_w * num_subblocks_w + n  // one row -> out_subblock_w * num_subblocks_w
                            );
                        }
                    }
                    // if (block == dbg_block) {
                    //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - add_tiles done " << ENDL());
                    // }

                    tile_regs_commit();
                    tile_regs_wait();

                    cb_reserve_back(cb_out, full_block_tiles);
                    for (uint32_t i = 0; i < full_block_tiles; i++) {
                        pack_tile(i, cb_out);
                    }
                    cb_push_back(cb_out, full_block_tiles);

                    // if (block == dbg_block) {
                    //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - data packed to out CB " << ENDL());
                    // }

                    cb_pop_front(cb_interm, full_block_tiles);
                    cb_pop_front(cb_in_c, full_block_tiles);

                    // if (block == dbg_block) {
                    //     DPRINT_MATH(DPRINT << "compute - block: " << block << " - poped C and INTERM CBs " <<
                    //     ENDL());
                    // }
                }
                tile_regs_release();
                if (block == dbg_block) {
                    DPRINT_MATH(DPRINT << "compute - block: " << block << " - TILE REGS RELEASED " << ENDL());
                }
            }
            if (block == dbg_block) {
                DPRINT_MATH(DPRINT << "compute - block: " << block << " - finished w subblock " << ENDL());
            }
        }
        cb_pop_front(cb_in_a, in0_block_w * out_subblock_h * num_subblocks_h);
        cb_pop_front(cb_in_b, in0_block_w * out_subblock_w * num_subblocks_w);
        if (block == dbg_block) {
            DPRINT_MATH(DPRINT << "compute - block: " << block << " - poped A and B CBs " << ENDL());
        }
    }
    DPRINT << "COMPUTE - finished!" << ENDL();
}
