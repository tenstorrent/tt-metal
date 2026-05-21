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

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    // BRISC writer kernel entry (proxy for dataflow go on this RISC).
    DeviceTimestampedData("BRISC_GO", program_id);

    const uint32_t end_tile_id = start_tile_id + n_tiles;
    const uint32_t last_tile_id = (n_tiles > 0) ? (end_tile_id - 1) : start_tile_id;

    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
        cb_wait_front(cb_out, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_out);

        if (tile_id == last_tile_id) {
            DeviceTimestampedData("WRITE_BEFORE_BARRIER", tile_id);
        }

        noc_async_write_tile(tile_id, dst, l1_read_addr);
        noc_async_write_barrier();

        if (tile_id == last_tile_id) {
            DeviceTimestampedData("WRITE_AFTER_BARRIER", tile_id);
        }

        cb_pop_front(cb_out, 1);
    }

    DeviceTimestampedData("BRISC_DONE", program_id);
}
