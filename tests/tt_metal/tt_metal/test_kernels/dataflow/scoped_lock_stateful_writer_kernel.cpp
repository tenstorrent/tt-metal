// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

// Writes into another core's locked region using STATEFUL writes. The destination core is programmed once by the
// set-state call (the hardware keeps it in the command buffer) and each write then supplies only the destination
// address word, so this kernel is what exercises the host-side destination reconstruction: the recorded
// WRITE_WITH_STATE / WRITE_WITH_TRID_WITH_STATE events carry a placeholder (0,0) core, and the host has to recover
// the real destination by correlating with the preceding set-state event.
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

    Semaphore my_sem(my_sem_id);
    Semaphore other_sem(other_sem_id);
    Noc noc;

    const uint32_t write_size = num_elements * sizeof(uint32_t);
    const uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_addr);

    // Wait for the other core to signal that it holds the lock.
    my_sem.down(1);

#if defined(USE_TRID)
    constexpr uint32_t trid = 1;
    // The trid set-state programs no length; the size is supplied per write instead.
    noc_async_write_one_packet_with_trid_set_state(target_noc_addr);
    for (uint32_t i = 0; i < 5; ++i) {
        noc_async_write_one_packet_with_trid_with_state(local_buffer_addr, target_addr, write_size, trid);
        noc_async_write_barrier();
    }
#elif defined(USE_POSTED)
    // Posted variant: the write needs no acknowledgement, so it is safe to aim at a destination address that may
    // not correspond to anything real -- a non-posted write there would block forever waiting for an ack that never
    // comes. Used by the large-destination-address test, where only the recorded address matters, not the write
    // landing anywhere meaningful.
    noc_async_write_one_packet_set_state<true /* posted */>(target_noc_addr, write_size);
    for (uint32_t i = 0; i < 5; ++i) {
        noc_async_write_one_packet_with_state<true /* posted */>(local_buffer_addr, target_addr);
        noc_async_posted_writes_flushed();
    }
#else
    // The non-trid set-state programs the transfer size as well, so the writes carry no size of their own.
    noc_async_write_one_packet_set_state(target_noc_addr, write_size);
    for (uint32_t i = 0; i < 5; ++i) {
        noc_async_write_one_packet_with_state(local_buffer_addr, target_addr);
        noc_async_write_barrier();
    }
#endif

    // Release the other core, then wait for it to confirm it has unlocked.
    other_sem.up(noc, other_noc_x, other_noc_y, 1);
    my_sem.down(1);
}
