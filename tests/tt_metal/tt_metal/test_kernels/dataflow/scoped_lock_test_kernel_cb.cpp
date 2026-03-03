// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t cb_id = get_arg_val<uint32_t>(0);
    uint32_t my_sem_id = get_arg_val<uint32_t>(1);
    uint32_t other_sem_id = get_arg_val<uint32_t>(2);
    uint32_t other_noc_x = get_arg_val<uint32_t>(3);
    uint32_t other_noc_y = get_arg_val<uint32_t>(4);

    experimental::Semaphore my_sem(my_sem_id);
    experimental::Semaphore other_sem(other_sem_id);
    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id);

    {
        auto lock = cb.scoped_lock();
        other_sem.up(noc, other_noc_x, other_noc_y, 1);
        my_sem.down(1);
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 2);
}
