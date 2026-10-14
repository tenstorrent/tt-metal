// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mixes large DRAM reads with non-posted DRAM writes on one NoC. For each task in this core's share:
//   - when the task starts a new table row, read the whole 16 KiB row (read_bytes per request) and wait for it;
//   - write four 1088 B pieces of a fixed payload to the cache pages the row selects, and wait for the acks.
// On Blackhole, 16 KiB DRAM reads issued alongside these writes have stalled the NoC within a fraction of a second.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ckernel.h"  // ckernel::load_blocking

void kernel_main() {
    constexpr uint32_t noc = get_compile_time_arg_val(0);
    constexpr uint32_t read_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = 16384;
    constexpr uint32_t piece_bytes = 1088;
    constexpr uint32_t pieces = 4;
    constexpr uint32_t tasks_per_row = 256;
    constexpr auto table_args = TensorAccessorArgs<2>();
    constexpr auto cache_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(table_args, get_arg_val<uint32_t>(0));
    const auto cache = TensorAccessor(cache_args, get_arg_val<uint32_t>(1));
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t seed = get_arg_val<uint32_t>(4);

    cb_reserve_back(0, 1);
    cb_reserve_back(1, 1);
    const uint32_t row_addr = get_write_ptr(0);
    const uint32_t payload_addr = get_write_ptr(1);
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_addr);
    volatile tt_l1_ptr uint32_t* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr);

    constexpr uint32_t piece_words = piece_bytes / 4;
    for (uint32_t i = 0; i < pieces * piece_words; ++i) {
        payload[i] = (seed * 65537u + (i % piece_words) * 37u + 11u) & 0x7fffffffu;
    }
    // The payload is the source of the NoC writes below, so its stores must reach L1 before the first write is issued.
    // A load of the last word that an instruction then consumes stalls until the load completes; L1 stores complete
    // in order, so all of them have landed by then.
    asm volatile("fence" ::: "memory");
    (void)ckernel::load_blocking(&payload[pieces * piece_words - 1]);

    uint32_t current_row = 0xffffffffu;
    for (uint32_t task = first; task < first + count; ++task) {
        const uint32_t row_id = task / tasks_per_row;
        const uint32_t head = (task % tasks_per_row) / 32u;
        const uint32_t slot = task % 32u;
        if (row_id != current_row) {
            for (uint32_t offset = 0; offset < row_bytes; offset += read_bytes) {
                noc_async_read(table.get_noc_addr(row_id, offset, noc), row_addr + offset, read_bytes, noc);
            }
            noc_async_read_barrier(noc);
            current_row = row_id;
        }
        const uint32_t page = row[slot] * 32u + head * pieces;
        for (uint32_t piece = 0; piece < pieces; ++piece) {
            noc_async_write(
                payload_addr + piece * piece_bytes, cache.get_noc_addr(page + piece, 0, noc), piece_bytes, noc);
        }
        noc_async_write_barrier(noc);
    }
}
