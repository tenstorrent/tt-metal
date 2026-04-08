// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    uint32_t local_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t write_size = get_arg_val<uint32_t>(1);
    uint32_t target_noc_x = get_arg_val<uint32_t>(2);
    uint32_t target_noc_y = get_arg_val<uint32_t>(3);
    uint32_t target_start_addr = get_arg_val<uint32_t>(4);
    uint32_t target_end_addr = get_arg_val<uint32_t>(5);
    uint32_t stride = get_arg_val<uint32_t>(6);
    uint32_t my_sem_id = get_arg_val<uint32_t>(7);
    uint32_t other_sem_id = get_arg_val<uint32_t>(8);
    uint32_t other_noc_x = get_arg_val<uint32_t>(9);
    uint32_t other_noc_y = get_arg_val<uint32_t>(10);

    experimental::Semaphore my_sem(my_sem_id);
    experimental::Semaphore other_sem(other_sem_id);
    experimental::Noc noc;
    experimental::UnicastEndpoint unicast_endpoint;

    experimental::CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);

    my_sem.down(1);

    for (uint32_t addr = target_start_addr; addr + write_size <= target_end_addr; addr += stride) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = addr});
        noc.async_write_barrier();
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);

    my_sem.down(1);

    for (uint32_t addr = target_start_addr; addr + write_size <= target_end_addr; addr += stride) {
        noc.async_write(
            local_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = target_noc_x, .noc_y = target_noc_y, .addr = addr});
        noc.async_write_barrier();
    }
}
