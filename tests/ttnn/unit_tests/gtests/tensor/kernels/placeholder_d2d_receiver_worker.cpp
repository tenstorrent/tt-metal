// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder receiver-side worker kernel for D2DStreamService tests. Stands in
// for a real consumer op: it runs the receiver-side handshake against the
// persistent receiver service kernel.
//
// Per iteration:
//   1. spin on the local data_ready_sem until the service multicast-incs it
//      (the transfer has landed in the receiver backing tensor), then reset it,
//   2. (a real consumer would read the backing-tensor slice + run compute here),
//   3. atomic-inc consumed_counter on the receiver service core (the service
//      kernel waits for num_workers of these before draining the next transfer).
//
// Runs a fixed num_iters then exits, so a test can Finish() the worker workload;
// because the service mcasts data_ready_sem only after the backing-tensor write
// barrier, the data is durable in DRAM by the time the final ack lands.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

constexpr uint32_t data_ready_sem_addr = get_compile_time_arg_val(0);
constexpr uint32_t num_iters = get_compile_time_arg_val(1);

void kernel_main() {
    const uint32_t consumed_counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t service_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t service_noc_y = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* data_ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_ready_sem_addr);
    const uint64_t consumed_counter_noc = get_noc_addr(service_noc_x, service_noc_y, consumed_counter_addr);
    Noc noc;

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        // 1. Wait for the service to signal the transfer landed, then reset.
        while (*data_ready_sem == 0) {
            invalidate_l1_cache();
        }
        *data_ready_sem = 0;

        // 2. (real consumer: read the backing-tensor slice + run downstream compute)

        // 3. Ack into consumed_counter — the service kernel waits for num_workers.
        noc_semaphore_inc(consumed_counter_noc, 1);
        noc.async_atomic_barrier();
    }
}
