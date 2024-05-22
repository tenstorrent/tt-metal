
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_HW = 1024;

constexpr auto cb_src = get_compile_time_arg_val(0);
constexpr auto cb_dst = get_compile_time_arg_val(1);

constexpr auto cb_last_row = get_compile_time_arg_val(2);

constexpr auto core0_ready_to_receive_sem = get_compile_time_arg_val(5);
constexpr auto core0_ready_to_send_sem = get_compile_time_arg_val(6);
constexpr auto work_done_sem = get_compile_time_arg_val(7);

constexpr uint32_t tile_size = get_tile_size(cb_src);
constexpr uint32_t tiles_per_block = 8;
constexpr uint32_t blocks_per_full_reshape = 4;
constexpr uint32_t tiles_per_reshape = tiles_per_block * blocks_per_full_reshape;
constexpr uint32_t quarter_tile_size = tile_size / 4;
constexpr uint32_t reshape_size = tiles_per_reshape * tile_size;

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << SETW(2) << (uint)r << ": " << TileSlice(cb_id, tile_id, sr, true, untilize);
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    uint32_t tiles_per_col = get_arg_val<uint32_t>(0);
    uint32_t reshapes_per_row = get_arg_val<uint32_t>(1);
    uint32_t total_tiles = get_arg_val<uint32_t>(2);
    uint32_t first_tile_offset = get_arg_val<uint32_t>(3);

    uint32_t first_core_x = get_arg_val<uint32_t>(4);
    uint32_t first_core_y = get_arg_val<uint32_t>(5);
    uint32_t reshape_offset = get_arg_val<uint32_t>(6);
    uint32_t ntiles_last_row_cb = get_arg_val<uint32_t>(7);
    uint32_t core_offset = get_arg_val<uint32_t>(8);

    // wait for the core0 to be ready to receive last row data
    auto* core0_ready_to_receive_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(core0_ready_to_receive_sem);
    noc_semaphore_wait(core0_ready_to_receive_sem_ptr, VALID);

    uint32_t src_addr = get_read_ptr(cb_src) + first_tile_offset;
    uint64_t dst_noc_addr = get_noc_addr(first_core_x, first_core_y, get_write_ptr(cb_last_row)) + core_offset;

    for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
        noc_async_write(src_addr, dst_noc_addr, tile_size);
        src_addr += reshape_size;
        dst_noc_addr += reshape_offset;
    }

    noc_async_write_barrier();

    uint64_t work_done_sem_addr = get_noc_addr(first_core_x, first_core_y, work_done_sem);
    noc_semaphore_inc(work_done_sem_addr, 1);

    // wait for the core0 to be ready to send last row data
    auto* core0_ready_to_send_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(core0_ready_to_send_sem);
    noc_semaphore_wait(core0_ready_to_send_sem_ptr, VALID);

    uint64_t src_noc_addr =
        get_noc_addr(first_core_x, first_core_y, get_read_ptr(cb_last_row)) + core_offset - tile_size;
    uint32_t dst_addr = get_write_ptr(cb_last_row);

    for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
        noc_async_read(src_noc_addr, dst_addr, tile_size);
        src_noc_addr += reshape_offset;
        dst_addr += reshape_offset;
    }

    noc_async_read_barrier();
    noc_semaphore_inc(work_done_sem_addr, 1);

    print_full_tile(cb_last_row, 0);

    cb_push_back(cb_last_row, ntiles_last_row_cb);
    cb_push_back(cb_src, total_tiles);
    cb_wait_front(cb_dst, total_tiles);
}
