// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "hw/inc/tt-1xx/risc_common.h"

void kernel_main() {
    uint32_t local_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    uint32_t target_noc_x = get_arg_val<uint32_t>(2);
    uint32_t target_noc_y = get_arg_val<uint32_t>(3);
    uint32_t target_addr = get_arg_val<uint32_t>(4);
    uint32_t my_sem_id = get_arg_val<uint32_t>(5);
    uint32_t other_sem_id = get_arg_val<uint32_t>(6);
    uint32_t other_noc_x = get_arg_val<uint32_t>(7);
    uint32_t other_noc_y = get_arg_val<uint32_t>(8);

    experimental::Semaphore my_sem(my_sem_id);
    experimental::Semaphore other_sem(other_sem_id);
    experimental::Noc noc;

    experimental::CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);

    for (uint32_t i = 0; i < num_elements; i++) {
        local_buffer[i] = 0x1000 + i;
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);
    my_sem.wait(1);

    {
        auto lock = local_buffer.scoped_lock(num_elements);

        // Spam some events for the purpose of testing
        for (uint32_t i = 0; i < 25; ++i) {
            uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_addr);
            noc_async_write(local_buffer_addr, target_noc_addr, num_elements * sizeof(uint32_t));
            noc_async_write_barrier();
        }
    }

    for (uint32_t i = 0; i < 50; ++i) {
        uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_addr);
        noc_async_write(local_buffer_addr, target_noc_addr, num_elements * sizeof(uint32_t));
        noc_async_write_barrier();
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);
    my_sem.wait(2);

    // Unlocked period
    for (uint32_t i = 0; i < 25; ++i) {
        uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_addr);
        noc_async_write(local_buffer_addr, target_noc_addr, num_elements * sizeof(uint32_t));
        noc_async_write_barrier();
    }
}
