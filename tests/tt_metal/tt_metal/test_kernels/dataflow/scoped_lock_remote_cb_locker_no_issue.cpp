// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// No-issue counterpart to scoped_lock_remote_cb_locker.cpp: the RemoteCircularBuffer lock is only held
// after the writer's pass has completed, so the NOC debug tool must NOT report a WRITE_TO_LOCKED_CB
// issue. Guards against false positives from the RemoteCircularBuffer lock events.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    uint32_t remote_cb_id = get_arg_val<uint32_t>(0);
    uint32_t my_sem_id = get_arg_val<uint32_t>(1);
    uint32_t other_sem_id = get_arg_val<uint32_t>(2);
    uint32_t other_noc_x = get_arg_val<uint32_t>(3);
    uint32_t other_noc_y = get_arg_val<uint32_t>(4);

    Semaphore my_sem(my_sem_id);
    Semaphore other_sem(other_sem_id);
    Noc noc;
    experimental::RemoteCircularBuffer rcb(remote_cb_id);

    // Wait for the writer to finish writing before taking the lock.
    my_sem.down(1);

    {
        auto lock = rcb.scoped_lock();
        other_sem.up(noc, other_noc_x, other_noc_y, 1);
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 2);
}
