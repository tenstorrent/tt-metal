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
    uint32_t write_size_a = get_arg(args::write_size_a);
    uint32_t write_size_b = get_arg(args::write_size_b);
    uint32_t target_noc_x = get_arg(args::target_noc_x);  // the locker core
    uint32_t target_noc_y = get_arg(args::target_noc_y);
    uint32_t target_addr_a = get_arg(args::target_addr_a);
    uint32_t target_addr_b = get_arg(args::target_addr_b);

    Semaphore lock_state_changed(sem::lock_state_changed);
    Semaphore written(sem::written);
    Noc noc;
    UnicastEndpoint unicast_endpoint;

    CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);

    lock_state_changed.down(1);

    noc.async_write(
        local_buffer,
        unicast_endpoint,
        write_size_a,
        {},
        {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr_a});
    noc.async_write_barrier();

    noc.async_write(
        local_buffer,
        unicast_endpoint,
        write_size_b,
        {},
        {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = target_addr_b});
    noc.async_write_barrier();

    written.up(noc, target_noc_x, target_noc_y, 1);

    lock_state_changed.down(1);
}
