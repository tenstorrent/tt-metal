// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t local_buffer_addr = get_arg(args::local_buffer_addr);
    uint32_t num_elements = get_arg(args::num_elements);
    uint32_t target_noc_x = get_arg(args::target_noc_x);  // the locker core
    uint32_t target_noc_y = get_arg(args::target_noc_y);
    uint32_t target_addr = get_arg(args::target_addr);

    Semaphore lock_state_changed(sem::lock_state_changed);
    Semaphore written(sem::written);
    Noc noc;
    UnicastEndpoint unicast_endpoint;

    CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);
    uint32_t write_size = num_elements * sizeof(uint32_t);

    // Wait for the locked core to signal it is locked
    lock_state_changed.down(1);

    // Do writes to the locked core
    for (uint32_t i = 0; i < 5; ++i) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr});
        noc.async_write_barrier();
    }

    // Signal the locked core to unlock
    written.up(noc, target_noc_x, target_noc_y, 1);

    // Now unlocked
    lock_state_changed.down(1);

    // Write to the other core's memory which is not locked
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
