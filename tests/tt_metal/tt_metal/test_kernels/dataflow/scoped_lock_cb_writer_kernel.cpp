// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"

// Writes into a CB on the locker core, at the address the locker publishes.
void kernel_main() {
    uint32_t local_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t write_size = get_arg_val<uint32_t>(1);
    uint32_t target_noc_x = get_arg_val<uint32_t>(2);
    uint32_t target_noc_y = get_arg_val<uint32_t>(3);
    // Local L1 word where the locker published the CB base to target.
    uint32_t inbox = get_arg_val<uint32_t>(4);
    uint32_t my_sem_id = get_arg_val<uint32_t>(5);
    uint32_t other_sem_id = get_arg_val<uint32_t>(6);
    uint32_t other_noc_x = get_arg_val<uint32_t>(7);
    uint32_t other_noc_y = get_arg_val<uint32_t>(8);

    Semaphore my_sem(my_sem_id);
    Semaphore other_sem(other_sem_id);
    Noc noc;
    UnicastEndpoint unicast_endpoint;

    CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);

    const auto write_to_cb = [&](uint32_t target_addr) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr});
        noc.async_write_barrier();
    };

    my_sem.down(1);  // the locker has published the target address (and taken the lock, if any)
    const uint32_t target_addr = *(volatile tt_l1_ptr uint32_t*)(uintptr_t)inbox;

    write_to_cb(target_addr);
    other_sem.up(noc, other_noc_x, other_noc_y, 1);  // ack: the locker may now release the lock

    my_sem.down(1);  // lock is released
    write_to_cb(target_addr);
    other_sem.up(noc, other_noc_x, other_noc_y, 1);
}
