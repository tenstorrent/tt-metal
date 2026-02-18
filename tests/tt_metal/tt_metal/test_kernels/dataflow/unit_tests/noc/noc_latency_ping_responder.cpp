// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Responder kernel for NOC hop latency ping-pong benchmark.
// Waits for sender to inc our semaphore, then incs sender's semaphore back.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    set_l1_data_cache<false>();
    uint32_t arg_idx = 0;
    const uint32_t sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t total_pings = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sender_ready_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    volatile tt_l1_ptr uint32_t* local_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_sem_addr);

    *local_sem = 0;

    // If sender_ready_sem_addr is nonzero, signal the sender that we're ready
    if (sender_ready_sem_addr != 0) {
        uint64_t sender_ready_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_ready_sem_addr);
        noc_semaphore_inc(sender_ready_noc_addr, 1);
    }

    uint64_t sender_sem_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sender_sem_addr);

    for (uint32_t i = 0; i < total_pings; i++) {
        noc_semaphore_wait(local_sem, 1);
        noc_semaphore_set(local_sem, 0);
        noc_semaphore_inc(sender_sem_noc_addr, 1);
    }
}
