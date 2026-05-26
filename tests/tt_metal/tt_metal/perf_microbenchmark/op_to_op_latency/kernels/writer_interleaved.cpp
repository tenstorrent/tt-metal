// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pops `n_tiles` tiles from the output circular buffer (CB_out) and writes
// them to the corresponding tile slots in an interleaved DRAM buffer. One
// instance of this kernel runs on every Tensix core; each core gets its own
// [start_tile_id, start_tile_id + n_tiles) slice via runtime args.
//
// The per-tile noc_async_write_barrier is intentional: it matches what real
// ops do today (full barrier at op end before the next op can safely begin)
// and is part of what the op-to-op latency benchmark is meant to measure.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t program_id = get_arg_val<uint32_t>(3);
    const uint32_t tiles_per_page_rt = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_page = tiles_per_page_rt > 0 ? tiles_per_page_rt : 1;
    // 0 = barrier after every write (default, mirrors current ops); 1 = single
    // barrier after all writes (experiment: shrinks pack_finish -> brisc_done).
    const uint32_t write_barrier_mode = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    // BRISC writer kernel entry (proxy for dataflow go on this RISC).
    DeviceTimestampedData("BRISC_GO", program_id);

    // DRAM page = tiles_per_page tiles. Compute pushes 1 CB page (1 tile) at
    // a time; we accumulate tiles_per_page in the CB then issue one
    // noc_async_write_tile that pushes the whole page in a single NoC txn.
    const uint32_t start_page_id = start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;
    const uint32_t end_page_id = start_page_id + n_pages;
    const uint32_t last_page_id = (n_pages > 0) ? (end_page_id - 1) : start_page_id;

    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_wait_front(cb_out, tiles_per_page);
        const uint32_t l1_read_addr = get_read_ptr(cb_out);

        if (page_id == last_page_id && write_barrier_mode == 0) {
            DeviceTimestampedData("WRITE_BEFORE_BARRIER", page_id);
        }

        noc_async_write_tile(page_id, dst, l1_read_addr);

        if (write_barrier_mode == 0) {
            noc_async_write_barrier();
            if (page_id == last_page_id) {
                DeviceTimestampedData("WRITE_AFTER_BARRIER", page_id);
            }
        }

        cb_pop_front(cb_out, tiles_per_page);
    }

    if (write_barrier_mode == 1) {
        DeviceTimestampedData("WRITE_BEFORE_BARRIER", last_page_id);
        noc_async_write_barrier();
        DeviceTimestampedData("WRITE_AFTER_BARRIER", last_page_id);
    }

    DeviceTimestampedData("BRISC_DONE", program_id);
}
