// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Bounded persistent ready/ack worker for the D2HStreamService benchmark.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

constexpr uint32_t transfer_done_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t total_iters = get_compile_time_arg_val(1);
constexpr uint32_t ungated_iters = get_compile_time_arg_val(2);
constexpr uint32_t latency_gate_sem_addr = get_compile_time_arg_val(3);

void kernel_main() {
    const uint32_t service_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(1);
    const uint32_t write_ack_counter_addr = get_arg_val<uint32_t>(2);

    Noc noc;
    volatile tt_l1_ptr uint32_t* transfer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(transfer_done_sem_addr);
    volatile tt_l1_ptr uint32_t* latency_gate = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(latency_gate_sem_addr);
    const uint64_t write_ack_noc = get_noc_addr(service_noc_x, service_noc_y, write_ack_counter_addr);

    for (uint32_t iter = 0; iter < total_iters; ++iter) {
        while (true) {
            invalidate_l1_cache();
            if (*transfer_done > 0) {
                *transfer_done = 0;
                break;
            }
        }

        if (iter >= ungated_iters) {
            while (true) {
                invalidate_l1_cache();
                if (*latency_gate > 0) {
                    *latency_gate = 0;
                    break;
                }
            }
        }

        // Device 2.0 migration: legacy primitive retained because write_ack_counter_addr is a raw
        // service-core L1 word allocated outside this program's semaphore namespace.
        noc_semaphore_inc(write_ack_noc, 1);
        noc.async_atomic_barrier();
    }
}
