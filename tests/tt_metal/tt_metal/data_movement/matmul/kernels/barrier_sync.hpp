// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Wait for all cores to reach the barrier before proceeding.
 *
 * This implements a coordinator-broadcast barrier synchronization:
 * 1. Each core increments a semaphore on the coordinator core
 * 2. The coordinator waits locally until its semaphore reaches num_cores
 * 3. The coordinator multicasts a "done" signal to all participating cores
 * 4. All cores wait for the "done" signal on their local semaphore
 *
 * This avoids the prior polling-read pattern that caused NOC congestion
 * with large core counts and potential issues with atomic wrap semantics.
 *
 * @param barrier_sem_id      Semaphore ID for arrival counting (on coordinator)
 * @param barrier_done_sem_id Semaphore ID for done broadcast (on all cores)
 * @param barrier_coord_x     Physical X coordinate of the coordinator core
 * @param barrier_coord_y     Physical Y coordinate of the coordinator core
 * @param num_cores           Total number of cores participating in the barrier
 * @param local_scratch_addr  Local L1 address for temporary storage (4 bytes)
 * @param mcast_start_x       Multicast rectangle start X (translated coords)
 * @param mcast_start_y       Multicast rectangle start Y (translated coords)
 * @param mcast_end_x         Multicast rectangle end X (translated coords)
 * @param mcast_end_y         Multicast rectangle end Y (translated coords)
 */
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
    // Single core: nothing to synchronize.
    if (num_cores <= 1) {
        return;
    }

    // Get L1 addresses for both semaphores
    uint32_t barrier_sem_addr = get_semaphore(barrier_sem_id);
    uint32_t done_sem_addr = get_semaphore(barrier_done_sem_id);

    // Get NOC address of coordinator's barrier semaphore
    uint64_t barrier_noc_addr = get_noc_addr(barrier_coord_x, barrier_coord_y, barrier_sem_addr);

    // Step 1: Signal arrival by incrementing the coordinator's semaphore
    noc_semaphore_inc(barrier_noc_addr, 1);
    noc_async_atomic_barrier();

    // Check if this core is the coordinator
    bool is_coordinator = (my_x[0] == barrier_coord_x && my_y[0] == barrier_coord_y);

    if (is_coordinator) {
        // Step 2: Coordinator waits locally until all cores have arrived
        volatile tt_l1_ptr uint32_t* barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);
        noc_semaphore_wait_min(barrier_sem_ptr, num_cores);
        noc_semaphore_set(barrier_sem_ptr, 0);  // Reset for potential reuse

        // Step 3: Multicast "done" signal to all cores (including self)
        // Write 1 to local scratch as the source value for the multicast
        volatile tt_l1_ptr uint32_t* scratch_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_addr);
        *scratch_ptr = 1;

        // Build the multicast address for the done semaphore on all cores
        // NOC0 and NOC1 have opposite routing directions, so swap start/end for NOC1
        uint64_t dst_done_sem_mcast_addr =
            noc_index == 0
                ? get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, done_sem_addr)
                : get_noc_multicast_addr(mcast_end_x, mcast_end_y, mcast_start_x, mcast_start_y, done_sem_addr);

        noc_semaphore_set_multicast_loopback_src(local_scratch_addr, dst_done_sem_mcast_addr, num_cores, false);
        noc_async_write_barrier();
    }

    // Step 4: All cores (including coordinator) wait for the done signal
    volatile tt_l1_ptr uint32_t* done_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_sem_addr);
    noc_semaphore_wait(done_sem_ptr, 1);
    noc_semaphore_set(done_sem_ptr, 0);  // Reset for potential reuse
}
