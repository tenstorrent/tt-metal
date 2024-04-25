

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
constexpr auto cb_scanned = get_compile_time_arg_val(4);

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

    uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(4);
    uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(5);
    uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(6);
    uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(7);

    uint32_t reshape_offset = get_arg_val<uint32_t>(8);

    uint32_t cores_to_wait_for = get_arg_val<uint32_t>(9);
    uint32_t ntiles_last_row_cb = get_arg_val<uint32_t>(10);

    // place its own last row data to the last row buffer
    uint32_t src_addr = get_read_ptr(cb_src) + first_tile_offset;
    uint64_t dst_noc_addr = get_noc_addr(get_write_ptr(cb_last_row));

    for (uint32_t reshape = 0; reshape < reshapes_per_row; ++reshape) {
        noc_async_write(src_addr, dst_noc_addr, tile_size);
        src_addr += reshape_size;
        dst_noc_addr += reshape_offset;
    }
    noc_async_write_barrier();

    // communicate to other cores that it's ready to receive data
    const uint64_t core0_ready_to_receive_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        core0_ready_to_receive_sem);

    auto* core0_ready_to_receive_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(core0_ready_to_receive_sem);
    noc_semaphore_set(core0_ready_to_receive_sem_ptr, VALID);
    noc_semaphore_set_multicast(core0_ready_to_receive_sem, core0_ready_to_receive_noc_addr, VALID);

    auto* work_done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(work_done_sem);
    noc_semaphore_wait(work_done_sem_ptr, cores_to_wait_for);

    // all cores have placed their last row data to the last row buffer
    // now this core runs the prefix scan on this data in compute kernel
    cb_push_back(cb_last_row, ntiles_last_row_cb);

    // compute signals to the reader that it's done
    cb_wait_front(cb_scanned, ntiles_last_row_cb);

    for (uint32_t i = 0; i < ntiles_last_row_cb; i++) {
        print_full_tile(cb_scanned, i);
    }

    // prepare for other cores to read the last row data back
    noc_semaphore_set(work_done_sem_ptr, 0);

    const uint64_t core0_ready_to_send_noc_addr = get_noc_multicast_addr(
        mcast_dest_noc_start_x,
        mcast_dest_noc_start_y,
        mcast_dest_noc_end_x,
        mcast_dest_noc_end_y,
        core0_ready_to_send_sem);

    auto* core0_ready_to_send_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(core0_ready_to_send_sem);
    noc_semaphore_set(core0_ready_to_send_sem_ptr, VALID);
    noc_semaphore_set_multicast(core0_ready_to_send_sem, core0_ready_to_send_noc_addr, VALID);

    noc_semaphore_wait(work_done_sem_ptr, cores_to_wait_for);
}
