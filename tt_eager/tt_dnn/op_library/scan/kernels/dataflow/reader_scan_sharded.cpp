// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_HW = 1024;

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);

constexpr auto cb_reshape = get_compile_time_arg_val(2);

constexpr auto cb_scanned = get_compile_time_arg_val(3);
constexpr auto cb_scanned2 = get_compile_time_arg_val(5);

constexpr auto cb_block = get_compile_time_arg_val(6);
constexpr auto cb_block2 = get_compile_time_arg_val(7);

constexpr auto cb_factors = get_compile_time_arg_val(8);

constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(10);

constexpr uint32_t tile_size = get_tile_size(cb_src);
constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;
constexpr uint32_t quarter_tile_size = tile_size / 4;
constexpr uint32_t reshape_size = tiles_per_reshape * tile_size;

#define ALWI inline __attribute__((always_inline))

ALWI void fill_with_val(uint32_t begin_addr, uint32_t n_tiles, uint16_t val) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        for (uint32_t j = 0; j < TILE_HW; ++j) {
            ptr[j] = val;
        }
        ptr += TILE_HW;
    }
}

ALWI void reshape_from_blocks(uint32_t block_cb, uint32_t reshape_cb) {
    for (uint32_t block = 0, offset = 0; block < blocks_per_full_reshape; ++block, offset += quarter_tile_size) {
        cb_wait_front(block_cb, tiles_per_block);

        uint64_t src_noc_addr = get_noc_addr(get_read_ptr(block_cb));
        uint32_t dst_addr = get_write_ptr(reshape_cb) + offset;
        for (uint32_t row = 0; row < TILE_HEIGHT; ++row, src_noc_addr += quarter_tile_size, dst_addr += tile_size) {
            noc_async_read(src_noc_addr, dst_addr, quarter_tile_size);
        }
        noc_async_read_barrier();

        cb_pop_front(block_cb, tiles_per_block);
    }
}

ALWI void undo_reshape_to_blocks(uint32_t scanned_cb, uint32_t block_cb) {
    for (uint32_t block = 0, offset = 0; block < blocks_per_full_reshape; ++block, offset += quarter_tile_size) {
        cb_reserve_back(block_cb, tiles_per_block);

        uint32_t src_addr = get_read_ptr(scanned_cb) + offset;
        uint64_t dst_noc_addr = get_noc_addr(get_write_ptr(block_cb));
        for (uint32_t row = 0; row < TILE_HEIGHT; ++row, src_addr += tile_size, dst_noc_addr += quarter_tile_size) {
            noc_async_write(src_addr, dst_noc_addr, quarter_tile_size);
        }
        noc_async_write_barrier();

        cb_push_back(block_cb, tiles_per_block);
    }
}

void kernel_main() {
    uint32_t tiles_per_row = get_arg_val<uint32_t>(0);
    uint32_t tiles_per_col = get_arg_val<uint32_t>(1);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(2);
    uint32_t total_tiles = get_arg_val<uint32_t>(3);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    fill_with_val(get_write_ptr(cb_factors), reshapes_per_row, bf16_one_u16);

    cb_push_back(cb_src, total_tiles);  // signal to compute kernel that the src CB is ready

    for (uint32_t row = 0; row < tiles_per_col; ++row) {
        for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
            // from src to reshape CB
            cb_reserve_back(cb_reshape, tiles_per_reshape);
            reshape_from_blocks(cb_block, cb_reshape);
            cb_push_back(cb_reshape, tiles_per_reshape);

            // Compute kernel fills scanned CB with cumulative product of tiles in reshape CB

            cb_reserve_back(cb_scanned2, tiles_per_reshape);
            cb_wait_front(cb_scanned, tiles_per_reshape);
            undo_reshape_to_blocks(cb_scanned, cb_block2);
            cb_pop_front(cb_scanned, tiles_per_reshape);
            cb_push_back(cb_scanned2, tiles_per_reshape);
        }
    }
}
