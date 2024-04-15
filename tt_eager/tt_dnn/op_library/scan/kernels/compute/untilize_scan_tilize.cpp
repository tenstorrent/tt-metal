// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);

constexpr auto cb_reshape = get_compile_time_arg_val(2);

constexpr auto cb_scanned = get_compile_time_arg_val(3);
constexpr auto cb_aux = get_compile_time_arg_val(4);
constexpr auto cb_scanned2 = get_compile_time_arg_val(5);

constexpr auto cb_block = get_compile_time_arg_val(6);
constexpr auto cb_block2 = get_compile_time_arg_val(7);

constexpr auto cb_factors = get_compile_time_arg_val(8);
constexpr auto cb_factors2 = get_compile_time_arg_val(9);

constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

ALWI void untilize_from_src(uint32_t src_cb, uint32_t block_cb) {
    pack_untilize_init_short<tiles_per_block>(src_cb, block_cb);
    for (uint32_t block = 0; block < blocks_per_full_reshape; ++block) {
        cb_wait_front(src_cb, tiles_per_block);
        cb_reserve_back(block_cb, tiles_per_block);

        pack_untilize_block<tiles_per_block>(src_cb, 1, block_cb);

        cb_push_back(block_cb, tiles_per_block);
        cb_pop_front(src_cb, tiles_per_block);
    }
    pack_untilize_uninit(block_cb);
}

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
    uint32_t tiles_per_row = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(1);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(2);
    uint32_t total_tiles = get_arg_val<uint32_t>(3);

    pack_untilize_init<tiles_per_block>(cb_src, cb_block);

    cb_push_back(cb_factors, reshapes_per_row);

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        cb_wait_front(cb_factors, reshapes_per_row);
        cb_reserve_back(cb_factors2, reshapes_per_row);

        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            untilize_from_src(cb_src, cb_block);

            cb_wait_front(cb_reshape, tiles_per_reshape);
            cb_reserve_back(cb_scanned, tiles_per_reshape);
            cb_reserve_back(cb_aux, tiles_per_reshape);

            // multiply the first tile of cb_reshape by the factor and push to cb_scanned
            mul_tiles_init();
            tile_regs_acquire();
            mul_tiles(cb_factors, cb_reshape, 0, 0, 0);
            cb_pop_front(cb_factors, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_aux);
            tile_regs_release();
            cb_push_back(cb_aux, 1);

            for (uint32_t tile = 1; tile < tiles_per_reshape; ++tile) {
                mul_tiles_init();
                cb_wait_front(cb_aux, 1);
                tile_regs_acquire();
                mul_tiles(cb_aux, cb_reshape, 0, tile, 0);
                cb_pop_front(cb_aux, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_aux);
                tile_regs_release();
                cb_push_back(cb_aux, 1);
            }

            cb_push_back(cb_scanned, tiles_per_reshape);

            // push this reshape to the destination
            tilize_to_dst(cb_block2, cb_dst);

            cb_wait_front(cb_scanned2, tiles_per_reshape);  // we are blocked here - why?

            // copy the last tile from cb_scanned back to cb_factors
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_scanned);
            copy_tile(cb_scanned2, tiles_per_reshape - 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_factors2);
            tile_regs_release();
            cb_push_back(cb_factors2, 1);

            cb_pop_front(cb_reshape, tiles_per_reshape);
            cb_pop_front(cb_scanned2, tiles_per_reshape);
            cb_pop_front(cb_aux, 1);
        }
        cb_pop_front(cb_factors2, reshapes_per_row);
        cb_push_back(cb_factors, reshapes_per_row);
    }
}
}  // namespace NAMESPACE
