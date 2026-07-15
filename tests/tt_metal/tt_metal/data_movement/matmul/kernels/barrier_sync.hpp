// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Coordinator-broadcast barrier: each core increments a counter on the coordinator,
// which then multicasts a done signal once the count reaches num_cores. Avoids the
// NOC congestion of polling-read barriers and atomic wrap edge cases.
FORCE_INLINE void barrier_sync(
    uint32_t barrier_sem_id,
    uint32_t barrier_done_sem_id,
    uint32_t barrier_coord_x,
    uint32_t barrier_coord_y,
    uint32_t num_cores,
    uint32_t local_scratch_addr,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y) {
    if (num_cores <= 1) {
        return;
    }

    uint32_t barrier_sem_addr = get_semaphore(barrier_sem_id);
    uint32_t done_sem_addr = get_semaphore(barrier_done_sem_id);
    uint64_t barrier_noc_addr = get_noc_addr(barrier_coord_x, barrier_coord_y, barrier_sem_addr);

    noc_semaphore_inc(barrier_noc_addr, 1);
    noc_async_atomic_barrier();

    bool is_coordinator = (my_x[0] == barrier_coord_x && my_y[0] == barrier_coord_y);

    if (is_coordinator) {
        volatile tt_l1_ptr uint32_t* barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);
        noc_semaphore_wait_min(barrier_sem_ptr, num_cores);
        noc_semaphore_set(barrier_sem_ptr, 0);

        volatile tt_l1_ptr uint32_t* scratch_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_addr);
        *scratch_ptr = 1;

        // NOC0/NOC1 have opposite routing, so swap start/end for NOC1.
        uint64_t dst_done_sem_mcast_addr =
            noc_index == 0
                ? get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, done_sem_addr)
                : get_noc_multicast_addr(mcast_end_x, mcast_end_y, mcast_start_x, mcast_start_y, done_sem_addr);

        noc_semaphore_set_multicast_loopback_src(local_scratch_addr, dst_done_sem_mcast_addr, num_cores, false);
        noc_async_write_barrier();
    }

    volatile tt_l1_ptr uint32_t* done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);
    noc_semaphore_wait(done_sem_ptr, 1);
    noc_semaphore_set(done_sem_ptr, 0);
}
