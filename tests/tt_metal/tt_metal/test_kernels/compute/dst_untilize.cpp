// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t num_faces = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows_per_face = get_compile_time_arg_val(3);
#ifndef ARCH_QUASAR
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
#else
    constexpr uint32_t dfb_in0 = get_compile_time_arg_val(4);
    constexpr uint32_t dfb_out0 = get_compile_time_arg_val(5);
#endif

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(per_core_block_tile_cnt);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

#ifndef ARCH_QUASAR
    compute_kernel_hw_startup(cb_in0, cb_out0);
    copy_tile_to_dst_init_short(cb_in0);
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(cb_out0, num_rows_per_face, num_faces);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        cb_reserve_back(cb_out0, full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(cb_in0, block_ct_dim);
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_ct_dim; ++i) {
                copy_tile(cb_in0, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(cb_out0, 1, b, num_rows_per_face, num_faces);
            tile_regs_release();
            cb_pop_front(cb_in0, block_ct_dim);
        }
        cb_push_back(cb_out0, full_ct_dim);
    }

    pack_untilize_uninit(cb_out0);
#else
    compute_kernel_hw_startup(dfb_in0, dfb_out0);
    copy_tile_to_dst_init_short(dfb_in0);
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(dfb_out0, num_rows_per_face, num_faces);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        cb_reserve_back(dfb_out0, full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(dfb_in0, block_ct_dim);
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_ct_dim; ++i) {
                copy_tile(dfb_in0, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(dfb_out0, 1, b, num_rows_per_face, num_faces);
            tile_regs_release();
            cb_pop_front(dfb_in0, block_ct_dim);
        }
        cb_push_back(dfb_out0, full_ct_dim);
    }

    pack_untilize_uninit(dfb_out0);
#endif
}
