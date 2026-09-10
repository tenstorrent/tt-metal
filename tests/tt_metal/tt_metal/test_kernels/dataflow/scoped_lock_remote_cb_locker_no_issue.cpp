// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// No-issue counterpart to scoped_lock_remote_cb_locker.cpp: the RemoteCircularBuffer lock is only taken while the
// writer is idle, so the NOC debug tool must NOT report a WRITE_TO_LOCKED_CB issue. Guards against false positives
// from the RemoteCircularBuffer lock events.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/remote_circular_buffer.h"

void kernel_main() {
    uint32_t remote_cb_id = get_arg_val<uint32_t>(0);
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
    experimental::RemoteCircularBuffer rcb(remote_cb_id);

    // Publish the remote CB base with no lock held.
    volatile tt_l1_ptr uint32_t* staged = (volatile tt_l1_ptr uint32_t*)(uintptr_t)local_scratch;
    *staged = get_remote_receiver_cb_interface(remote_cb_id).fifo_start_addr;
    CoreLocalMem<uint32_t> addr_src(local_scratch);
    noc.async_write(
        addr_src,
        unicast_endpoint,
        sizeof(uint32_t),
        {},
        {.noc_x = other_noc_x, .noc_y = other_noc_y, .addr = writer_inbox});
    noc.async_write_barrier();

    other_sem.up(noc, other_noc_x, other_noc_y, 1);  // addr published, CB unlocked -> writer may write
    my_sem.down(1);                                  // that write has landed

    {
        auto lock = rcb.scoped_lock();  // lock and release with no writer activity inside
    }

    other_sem.up(noc, other_noc_x, other_noc_y, 1);  // lock released -> writer may write again
    my_sem.down(1);                                  // stay alive until that write has landed
}
