// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pops `n_tiles` tiles from the output circular buffer (CB_out) and writes
// them to the corresponding tile slots in an interleaved DRAM buffer. One
// instance of this kernel runs on every Tensix core; each core gets its own
// [start_tile_id, start_tile_id + n_tiles) slice via runtime args.
//
// Synchronization (per kernel review feedback):
//   - noc_async_writes_flushed() after each tile so the source L1 slot is safe
//     to recycle on the next cb_pop_front (write has at least left L1).
//   - noc_async_write_barrier() ONCE at kernel exit so the next op sees fully
//     committed DRAM writes (op-end safety).
//
// Runtime args:
//   0: dst_addr
//   1: n_tiles
//   2: start_tile_id
//   3: program_id
//
// Compile-time args (see TensorAccessorArgs<1> for dst accessor):
//   0: cb_out
//   1: TILES_PER_PAGE

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
    constexpr uint32_t tiles_per_page = TILES_PER_PAGE > 0 ? TILES_PER_PAGE : 1;
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    DeviceTimestampedData("BRISC_GO", program_id);

    // DRAM page = tiles_per_page tiles. Compute pushes 1 CB page (1 tile) at
    // a time; we accumulate tiles_per_page in the CB then issue one
    // noc_async_write_tile that pushes the whole page in a single NoC txn.
    const uint32_t start_page_id = start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;
    const uint32_t end_page_id = start_page_id + n_pages;
    const uint32_t last_page_id = (n_pages > 0) ? (end_page_id - 1) : start_page_id;

    DeviceTimestampedData("WRITE_BEFORE_BARRIER", last_page_id);

    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_wait_front(cb_out, tiles_per_page);
        const uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(page_id, dst, l1_read_addr);
        // Cheap: ensures write left source L1, so the next cb_pop_front
        // can safely recycle the slot without corrupting an in-flight write.
        noc_async_writes_flushed();
        cb_pop_front(cb_out, tiles_per_page);
    }

    // Op-end safety: ONE barrier at kernel exit so the next op sees consistent
    // DRAM (writes have not just left L1, they have landed at the destination).
    noc_async_write_barrier();
    DeviceTimestampedData("WRITE_AFTER_BARRIER", last_page_id);

    DeviceTimestampedData("BRISC_DONE", program_id);
}
