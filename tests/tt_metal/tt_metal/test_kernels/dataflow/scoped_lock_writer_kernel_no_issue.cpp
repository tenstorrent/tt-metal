// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

// No-issue flow: write -> signal "first batch done" -> wait locked -> wait unlocked -> write (no writes while locked)
void kernel_main() {
    uint32_t local_buffer_addr = get_arg(args::local_buffer_addr);
    uint32_t num_elements = get_arg(args::num_elements);
    uint32_t target_noc_x = get_arg(args::target_noc_x);  // the locker core
    uint32_t target_noc_y = get_arg(args::target_noc_y);
    uint32_t target_addr = get_arg(args::target_addr);

    Semaphore first_batch_written(sem::first_batch_written);
    Semaphore lock_state_changed(sem::lock_state_changed);
    Noc noc;
    UnicastEndpoint unicast_endpoint;

    CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);
    uint32_t write_size = num_elements * sizeof(uint32_t);

    // First batch: write while locker has not locked (no issue)
    for (uint32_t i = 0; i < 5; ++i) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr});
        noc.async_write_barrier();
    }

    first_batch_written.up(noc, target_noc_x, target_noc_y, 1);

    // locked
    lock_state_changed.down(1);

    // unlocked
    lock_state_changed.down(1);

    // Second batch: write after locker has unlocked (no issue)
    for (uint32_t i = 0; i < 5; ++i) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr});
        noc.async_write_barrier();
    }
}
