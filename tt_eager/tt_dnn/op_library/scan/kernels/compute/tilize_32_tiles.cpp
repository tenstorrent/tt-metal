
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);

constexpr auto cb_reshape = get_compile_time_arg_val(2);
constexpr auto cb_block = get_compile_time_arg_val(3);

constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

ALWI void tilize_to_dst(uint32_t block_cb, uint32_t dst_cb) {
    tilize_init_short(block_cb, tiles_per_block);
    for (uint32_t block = 0; block < blocks_per_full_reshape; ++block) {
        cb_wait_front(block_cb, tiles_per_block);
        cb_reserve_back(dst_cb, tiles_per_block);

        tilize_block(block_cb, tiles_per_block, dst_cb);

        cb_push_back(dst_cb, tiles_per_block);
        cb_pop_front(block_cb, tiles_per_block);
    }
    tilize_uninit(block_cb);
}

namespace NAMESPACE {
void MAIN {
    uint32_t tiles_per_col = get_arg_val<uint32_t>(0);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);

    tilize_init(cb_block, tiles_per_block, cb_dst);

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            // block -> reshape 8 tiles at a time, 4 times
            tilize_to_dst(cb_block, cb_reshape);

            cb_wait_front(cb_reshape, tiles_per_reshape);

            // copy reshape to dst
            for (uint32_t tile = 0; tile < tiles_per_reshape; ++tile) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_reshape);
                copy_tile(cb_reshape, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_dst);
                tile_regs_release();
                cb_push_back(cb_dst, 1);
            }

            cb_pop_front(cb_reshape, tiles_per_reshape);
        }
    }
}
}  // namespace NAMESPACE
