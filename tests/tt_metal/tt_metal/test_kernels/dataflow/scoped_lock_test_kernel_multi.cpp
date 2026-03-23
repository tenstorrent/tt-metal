// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    uint32_t buffer_addr_a = get_arg_val<uint32_t>(0);
    uint32_t num_elements_a = get_arg_val<uint32_t>(1);
    uint32_t buffer_addr_b = get_arg_val<uint32_t>(2);
    uint32_t num_elements_b = get_arg_val<uint32_t>(3);
    uint32_t my_sem_id = get_arg_val<uint32_t>(4);
    uint32_t other_sem_id = get_arg_val<uint32_t>(5);
    uint32_t other_noc_x = get_arg_val<uint32_t>(6);
    uint32_t other_noc_y = get_arg_val<uint32_t>(7);

    experimental::Semaphore my_sem(my_sem_id);
    experimental::Semaphore other_sem(other_sem_id);
    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> buf_a(buffer_addr_a);
    experimental::CoreLocalMem<uint32_t> buf_b(buffer_addr_b);

    {
        auto lock_a = buf_a.scoped_lock(num_elements_a);
        auto lock_b = buf_b.scoped_lock(num_elements_b);
        other_sem.up(noc, other_noc_x, other_noc_y, 1);

        my_sem.down(1);
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);
}
