// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pops `n_tiles` tiles from the output circular buffer (CB_out) and writes
// them to the corresponding tile slots in an interleaved DRAM buffer. One
// instance of this kernel runs on every Tensix core; each core gets its own
// [start_tile_id, start_tile_id + n_tiles) slice via runtime args.
//
// Synchronization:
//   - noc_async_writes_flushed() once per CB-worth of pages (batch up to
//     output_cb_depth/tiles_per_page - 1 writes before a flush) so the source L1 slot is
//     safe to recycle on cb_pop_front. Per-page flushing serializes the writer and ~halves
//     write BW when not DRAM-bound, with no benefit when saturated, so it is not an option.
//   - End of kernel: END_BARRIER_MODE selects barrier / flush / none (op-end safety).
//
// Runtime args:
//   0: dst_addr
//   1: n_tiles
//   2: start_tile_id
//   3: program_id
//
// Compile-time args (see TensorAccessorArgs<6> for dst accessor):
//   0: cb_out
//   1: TILES_PER_PAGE
//   2: READ_ONLY        (1 = skip DRAM writes, just pop from output CB)
//   3: OUTPUT_CB_DEPTH_TILES
//   4: CROSS_PROGRAM_OFFSET_TILES (0 = every program writes same slice; >0 = program k
//                                  writes pages starting at start_tile_id + k*OFFSET)
//   5: END_BARRIER_MODE
//                          0 = noc_async_write_barrier (DEFAULT; wait for DRAM ACK; current)
//                          1 = noc_async_writes_flushed (just flush local L1; writes left
//                              the source but next op may see in-flight committed-at-NoC)
//                          2 = no barrier, no flush (UNSAFE for correctness in real
//                              workloads; simulates "HW gives us this for free")

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t program_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t READ_ONLY = get_compile_time_arg_val(2);
    constexpr uint32_t OUTPUT_CB_DEPTH_TILES = get_compile_time_arg_val(3);
    constexpr uint32_t CROSS_PROGRAM_OFFSET_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t END_BARRIER_MODE = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_page = TILES_PER_PAGE > 0 ? TILES_PER_PAGE : 1;
    constexpr uint32_t output_cb_depth_tiles = OUTPUT_CB_DEPTH_TILES > 0 ? OUTPUT_CB_DEPTH_TILES : tiles_per_page;
    constexpr uint32_t cb_page_slots = output_cb_depth_tiles / tiles_per_page;
    // Flush once per CB-worth of pages (never per page -- see header).
    constexpr uint32_t max_writes_before_flush = cb_page_slots > 1 ? cb_page_slots - 1 : 1;
    constexpr auto dst_args = TensorAccessorArgs<6>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    DeviceTimestampedData("BRISC_GO", program_id);

    // When CROSS_PROGRAM_OFFSET_TILES > 0, each program writes a disjoint slice
    // (mirrors reader); host allocates a buffer large enough for all slices.
    const uint32_t effective_start_tile_id = start_tile_id + program_id * CROSS_PROGRAM_OFFSET_TILES;
    // DRAM page = tiles_per_page tiles. Compute pushes 1 CB page (1 tile) at
    // a time; we accumulate tiles_per_page in the CB then issue one
    // noc_async_write_tile that pushes the whole page in a single NoC txn.
    const uint32_t start_page_id = effective_start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;
    const uint32_t end_page_id = start_page_id + n_pages;
    const uint32_t last_page_id = (n_pages > 0) ? (end_page_id - 1) : start_page_id;

    DeviceTimestampedData("WRITE_BEFORE_BARRIER", last_page_id);

    uint32_t writes_since_flush = 0;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_wait_front(cb_out, tiles_per_page);
        if constexpr (READ_ONLY == 0) {
            const uint32_t l1_read_addr = get_read_ptr(cb_out);
            noc_async_write_tile(page_id, dst, l1_read_addr);
            // Flush once per CB-worth of pages so the source slot is safe to recycle.
            if (++writes_since_flush >= max_writes_before_flush) {
                noc_async_writes_flushed();
                writes_since_flush = 0;
            }
        }
        cb_pop_front(cb_out, tiles_per_page);
    }

    if constexpr (READ_ONLY == 0) {
        if constexpr (END_BARRIER_MODE == 0) {
            noc_async_write_barrier();
        } else if constexpr (END_BARRIER_MODE == 1) {
            noc_async_writes_flushed();
        }
    }
    DeviceTimestampedData("WRITE_AFTER_BARRIER", last_page_id);

    DeviceTimestampedData("BRISC_DONE", program_id);
}
