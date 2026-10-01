// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);

void kernel_main() {
    Noc noc;
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_x = get_arg_val<uint32_t>(1);
    const uint32_t service_y = get_arg_val<uint32_t>(2);
    const uint32_t total_iters = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* data_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    // Device 2.0 migration: legacy address helper retained because noc_semaphore_inc
    // still takes a raw NoC address for this arbitrary service-core L1 counter.
    const uint64_t consumed_remote_addr = get_noc_addr(service_x, service_y, consumed_counter_addr);

    for (uint32_t iter = 0; iter < total_iters; ++iter) {
        while (true) {
            invalidate_l1_cache();
            if (*data_ready_sem > 0) {
                *data_ready_sem = 0;
                break;
            }
        }

        // Device 2.0 migration: legacy primitive retained. consumed_counter_addr is an arbitrary
        // service-core L1 counter, not a semaphore id that Semaphore<> can target.
        noc_semaphore_inc(consumed_remote_addr, 1);
        noc.async_atomic_barrier();
    }
}
