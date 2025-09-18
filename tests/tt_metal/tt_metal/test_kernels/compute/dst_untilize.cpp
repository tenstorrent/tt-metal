// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {

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

void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t num_faces = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows_per_face = get_compile_time_arg_val(3);

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(per_core_block_tile_cnt);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_to_dst_init_short(tt::CBIndex::c_0);
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(tt::CBIndex::c_16, num_rows_per_face, num_faces);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        cb_reserve_back(tt::CBIndex::c_16, full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(tt::CBIndex::c_0, block_ct_dim);
            tile_regs_acquire();
            for (uint32_t i = 0; i < block_ct_dim; ++i) {
                copy_tile(tt::CBIndex::c_0, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(tt::CBIndex::c_16, 1, b, num_rows_per_face, num_faces);
            tile_regs_release();
            cb_pop_front(tt::CBIndex::c_0, block_ct_dim);
        }
        cb_push_back(tt::CBIndex::c_16, full_ct_dim);
    }

    pack_untilize_uninit(tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
