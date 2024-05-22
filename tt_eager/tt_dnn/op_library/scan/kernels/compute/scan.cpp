
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "debug/dprint.h"

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);

constexpr auto cb_factors = get_compile_time_arg_val(2);
constexpr auto cb_factors2 = get_compile_time_arg_val(3);

constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK(DPRINT << "======" << ENDL());
    for (uint16_t r = 0; r < 32; ++r) {
        uint16_t h1 = r + 1;
        SliceRange sr = SliceRange{.h0 = r, .h1 = h1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL());
    }
    PACK(DPRINT << "++++++" << ENDL());
}

namespace NAMESPACE {
void MAIN {
    uint32_t tiles_per_col = get_arg_val<uint32_t>(0);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);

    cb_push_back(cb_factors, reshapes_per_row);

    uint32_t tile_count = 0;
    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        cb_wait_front(cb_factors, reshapes_per_row);
        cb_reserve_back(cb_factors2, reshapes_per_row);

        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            cb_wait_front(cb_src, tiles_per_reshape);
            cb_reserve_back(cb_dst, 1);

            // multiply the first tile of cb_src by the factor and push to cb_dst
            mul_tiles_init();
            tile_regs_acquire();
            mul_tiles(cb_factors, cb_src, 0, 0, 0);
            cb_pop_front(cb_factors, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dst);
            tile_regs_release();
            cb_push_back(cb_dst, 1);

            for (uint32_t tile = 1; tile < tiles_per_reshape; ++tile) {
                mul_tiles_init();
                cb_wait_front(cb_dst, 1);
                tile_regs_acquire();
                mul_tiles(cb_dst, cb_src, 0, tile, 0);
                cb_pop_front(cb_dst, 1);
                tile_regs_commit();
                cb_reserve_back(cb_dst, 1);
                tile_regs_wait();
                pack_tile(0, cb_dst);
                tile_regs_release();
                cb_push_back(cb_dst, 1);
            }

            cb_wait_front(cb_dst, 1);

            // copy the last tile from cb_dst back to cb_factors via cb_factors2
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_dst);
            copy_tile(cb_dst, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_factors2);
            tile_regs_release();
            cb_push_back(cb_factors2, 1);

            cb_pop_front(cb_src, tiles_per_reshape);
            cb_pop_front(cb_dst, 1);
        }
        cb_pop_front(cb_factors2, reshapes_per_row);
        cb_push_back(cb_factors, reshapes_per_row);
    }
}
}  // namespace NAMESPACE
