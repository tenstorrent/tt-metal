// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t cb_id = get_arg_val<uint32_t>(0);
    uint32_t my_sem_id = get_arg_val<uint32_t>(1);
    uint32_t other_sem_id = get_arg_val<uint32_t>(2);
    uint32_t other_noc_x = get_arg_val<uint32_t>(3);
    uint32_t other_noc_y = get_arg_val<uint32_t>(4);
    uint32_t local_scratch = get_arg_val<uint32_t>(5);
    uint32_t writer_inbox = get_arg_val<uint32_t>(6);

    Semaphore my_sem(my_sem_id);
    Semaphore other_sem(other_sem_id);
    Noc noc;
    UnicastEndpoint unicast_endpoint;
    CircularBuffer cb(cb_id);

    {
        auto lock = cb.scoped_lock();

        // Publish the locked base (stage in local L1, then NOC it across).
        volatile tt_l1_ptr uint32_t* staged = (volatile tt_l1_ptr uint32_t*)(uintptr_t)local_scratch;
        *staged = cb.get_write_ptr();
        CoreLocalMem<uint32_t> addr_src(local_scratch);
        noc.async_write(
            addr_src,
            unicast_endpoint,
            sizeof(uint32_t),
            {},
            {.noc_x = other_noc_x, .noc_y = other_noc_y, .addr = writer_inbox});
        noc.async_write_barrier();

        // Release the writer: the lock is held and the target address is published.
        other_sem.up(noc, other_noc_x, other_noc_y, 1);
        my_sem.down(1);  // hold the lock until the writer's in-lock write is done
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);  // lock released -> writer may write again
    my_sem.down(1);                                  // stay alive until that write has landed
}
