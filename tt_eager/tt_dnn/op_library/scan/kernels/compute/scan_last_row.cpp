

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "debug/dprint.h"

#define PRINTER UNPACK

inline void print_full_tile_unpack(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    UNPACK(DPRINT << "======" << ENDL());
    for (uint16_t r = 0; r < 32; ++r) {
        uint16_t h1 = r + 1;
        SliceRange sr = SliceRange{.h0 = r, .h1 = h1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        UNPACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL());
    }
    UNPACK(DPRINT << "++++++" << ENDL());
}

inline void print_full_tile_pack(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK(DPRINT << "======" << ENDL());
    for (uint16_t r = 0; r < 32; ++r) {
        uint16_t h1 = r + 1;
        SliceRange sr = SliceRange{.h0 = r, .h1 = h1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PACK(DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL());
    }
    PACK(DPRINT << "++++++" << ENDL());
}

void print_cb_details(uint32_t cb_id) {
    PRINTER(
        DPRINT << "cb_id " << cb_id << ": { " << "size: " << cb_interface[cb_id].fifo_size << ", " << "limit: "
               << cb_interface[cb_id].fifo_limit << ", " << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
               << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
               << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", " << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr
               << ", " << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL());
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t dst0 = 0;

    constexpr auto cb_last_row = get_compile_time_arg_val(2);
    constexpr auto cb_aux = get_compile_time_arg_val(3);
    constexpr auto cb_scanned = get_compile_time_arg_val(4);

    constexpr uint32_t tiles_per_block = 8;
    constexpr uint32_t blocks_per_full_reshape = 4;
    constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;

    uint32_t tiles_per_col = get_arg_val<uint32_t>(0);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);
    uint32_t ncores = get_arg_val<uint32_t>(3);

    for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
        cb_wait_front(cb_last_row, ncores);
        cb_reserve_back(cb_scanned, ncores);
//        cb_reserve_back(cb_aux, 1);
/*
        // copy first tile as is
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_last_row);
        copy_tile(cb_last_row, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_aux);
        tile_regs_release();
        cb_push_back(cb_aux, 1);

        for (uint32_t tile = 1; tile < ncores; ++tile) {
            mul_tiles_init();
            tile_regs_acquire();
            cb_wait_front(cb_aux, 1);
            mul_tiles(cb_aux, cb_last_row, 0, tile, dst0);
            cb_pop_front(cb_aux, 1);
            tile_regs_commit();
            cb_reserve_back(cb_aux, 1);
            tile_regs_wait();
            pack_tile(dst0, cb_aux);
            tile_regs_release();
            cb_push_back(cb_aux, 1);
        }
*/
        cb_push_back(cb_scanned, ncores);
        cb_pop_front(cb_last_row, ncores);
//        cb_pop_front(cb_aux, 1);
    }
}
}  // namespace NAMESPACE
