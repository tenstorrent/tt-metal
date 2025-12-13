// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    uint32_t my_sem_id = get_arg_val<uint32_t>(2);
    uint32_t other_sem_id = get_arg_val<uint32_t>(3);
    uint32_t other_noc_x = get_arg_val<uint32_t>(4);
    uint32_t other_noc_y = get_arg_val<uint32_t>(5);

    experimental::Semaphore my_sem(my_sem_id);
    experimental::Semaphore other_sem(other_sem_id);
    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> buffer(l1_buffer_addr);

    {
        auto lock = buffer.scoped_lock(num_elements);

        my_sem.wait(1);
        other_sem.up(noc, other_noc_x, other_noc_y, 1);

        for (uint32_t i = 0; i < num_elements; i++) {
            buffer[i] = i;
        }

        my_sem.wait(2);
        other_sem.up(noc, other_noc_x, other_noc_y, 1);
    }
}
