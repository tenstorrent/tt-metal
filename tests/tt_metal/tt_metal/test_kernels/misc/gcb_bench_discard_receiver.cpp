// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bandwidth-bench worker receiver for the DRAM-core vs worker-core prefetcher
// BW comparison. Per-page wait_front + pop_front in a loop, discards data.
//
// Unlike gcb_smoke_receiver.cpp (one big wait_front(num_pages) + pop_front(num_pages)),
// this kernel drains pages one-at-a-time so the GCB fifo (which only holds a few
// pages of in-flight data) keeps refilling as the sender pushes through num_iters
// total pages.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/remote_circular_buffer.h"
#include "risc_common.h"

void kernel_main() {
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_iters = get_compile_time_arg_val(1);
    constexpr uint32_t ordinary_read_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t ordinary_read_scratch_cb = get_compile_time_arg_val(3);

    uint64_t ordinary_dram_noc_addr = 0;
    uint32_t ordinary_read_scratch_addr = 0;
    volatile tt_l1_ptr uint64_t* timing = nullptr;
    if constexpr (ordinary_read_bytes > 0) {
        const uint32_t bank_id = get_arg_val<uint32_t>(0);
        const uint32_t ordinary_dram_addr = get_arg_val<uint32_t>(1);
        const uint32_t timing_l1_addr = get_arg_val<uint32_t>(2);
        ordinary_dram_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, ordinary_dram_addr);
        ordinary_read_scratch_addr = get_write_ptr(ordinary_read_scratch_cb);
        timing = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(timing_l1_addr);
    }

    uint64_t prefetch_wait_cycles = 0;
    uint64_t ordinary_read_cycles = 0;
    const uint64_t total_start = get_timestamp();
    for (uint32_t i = 0; i < num_iters; ++i) {
        const uint64_t wait_start = get_timestamp();
        experimental::remote_cb_wait_front(remote_cb_id, 1);
        prefetch_wait_cycles += get_timestamp() - wait_start;
        if constexpr (ordinary_read_bytes > 0) {
            const uint64_t read_start = get_timestamp();
            noc_async_read(ordinary_dram_noc_addr, ordinary_read_scratch_addr, ordinary_read_bytes);
            noc_async_read_barrier();
            ordinary_read_cycles += get_timestamp() - read_start;
        }
        experimental::remote_cb_pop_front(remote_cb_id, 1);
    }
    if constexpr (ordinary_read_bytes > 0) {
        timing[0] = prefetch_wait_cycles;
        timing[1] = ordinary_read_cycles;
        timing[2] = get_timestamp() - total_start;
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timing)[6] = num_iters;
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timing)[7] = ordinary_read_bytes;
    }
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_async_atomic_barrier();
}
