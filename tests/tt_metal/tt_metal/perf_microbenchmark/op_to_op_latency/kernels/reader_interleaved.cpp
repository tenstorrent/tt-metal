// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads `n_tiles` tiles from interleaved DRAM into CB_in.
//
// Runtime args:
//   0: src_addr
//   1: n_tiles
//   2: start_tile_id
//   3: program_id
//   4: push_tile_count  (e.g. 2 — reserve/read/push this many tiles per chunk)
//   5: reader_mode      (0 = reserve N, read+push one-by-one; 1 = reserve N, read all, push N)
//
// Profiler on first tile of the slice only: READ_BEFORE_BARRIER / READ_AFTER_BARRIER.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t program_id = get_arg_val<uint32_t>(3);
    const uint32_t push_tile_count = get_arg_val<uint32_t>(4);
    const uint32_t reader_mode = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const auto src = TensorAccessor(src_args, src_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    DeviceTimestampedData("NCRISC_GO", program_id);

    const uint32_t chunk_size = push_tile_count > 0 ? push_tile_count : 1;
    const uint32_t end_tile_id = start_tile_id + n_tiles;
    const uint32_t tile_bytes = get_local_cb_interface(cb_in).fifo_page_size;

    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id;) {
        const uint32_t tiles_left = end_tile_id - tile_id;
        const uint32_t chunk = tiles_left < chunk_size ? tiles_left : chunk_size;

        cb_reserve_back(cb_in, chunk);
        const uint32_t cb_base = get_write_ptr(cb_in);

        if (reader_mode == 1) {
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tid = tile_id + i;
                const uint32_t l1_write_addr = cb_base + i * tile_bytes;
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_BEFORE_BARRIER", tid);
                }
                noc_async_read_tile(tid, src, l1_write_addr);
                noc_async_read_barrier();
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_AFTER_BARRIER", tid);
                }
            }
            cb_push_back(cb_in, chunk);
        } else {
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tid = tile_id + i;
                const uint32_t l1_write_addr = get_write_ptr(cb_in);
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_BEFORE_BARRIER", tid);
                }
                noc_async_read_tile(tid, src, l1_write_addr);
                noc_async_read_barrier();
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_AFTER_BARRIER", tid);
                }
                cb_push_back(cb_in, 1);
            }
        }

        tile_id += chunk;
    }

    DeviceTimestampedData("NCRISC_DONE", program_id);
}
