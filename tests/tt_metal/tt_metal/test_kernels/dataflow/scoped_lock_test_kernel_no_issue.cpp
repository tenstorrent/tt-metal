// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"
#include "experimental/core_local_mem.h"

// No-issue flow: wait for writer to finish first batch (no lock) -> lock -> signal locked -> unlock -> signal unlocked
void kernel_main() {
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    uint32_t my_sem_id = get_arg_val<uint32_t>(2);     // locker_sem: wait for "writer first batch done"
    uint32_t other_sem_id = get_arg_val<uint32_t>(3);  // writer_sem: signal "locked" then "unlocked"
    uint32_t other_noc_x = get_arg_val<uint32_t>(4);
    uint32_t other_noc_y = get_arg_val<uint32_t>(5);

    experimental::Semaphore my_sem(my_sem_id);
    experimental::Semaphore other_sem(other_sem_id);
    experimental::Noc noc;
    experimental::CoreLocalMem<uint32_t> buffer(l1_buffer_addr);

    // Wait for writer to complete first batch of writes (while buffer was not locked)
    my_sem.down(1);

    {
        // Lock this buffer for num_elements
        auto lock = buffer.scoped_lock(num_elements);
        other_sem.up(noc, other_noc_x, other_noc_y, 1);  // signal "locked"
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);
}
