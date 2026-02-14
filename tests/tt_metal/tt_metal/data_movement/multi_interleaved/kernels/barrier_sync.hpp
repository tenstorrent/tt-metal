// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Wait for all cores to reach the barrier before proceeding.
 *
 * This implements a global barrier synchronization where:
 * 1. Each core increments a semaphore on the coordinator core
 * 2. Each core polls the coordinator's semaphore until it equals num_cores
 * 3. Once all cores have arrived, they all proceed simultaneously
 *
 * @param barrier_sem_id      Semaphore ID (call get_semaphore() to get L1 address)
 * @param barrier_coord_x     Physical X coordinate of the coordinator core
 * @param barrier_coord_y     Physical Y coordinate of the coordinator core
 * @param num_cores           Total number of cores participating in the barrier
 * @param local_scratch_addr  Local L1 address for polling scratch space (4 bytes)
 */
FORCE_INLINE void barrier_sync(
    uint32_t barrier_sem_id,
    uint32_t barrier_coord_x,
    uint32_t barrier_coord_y,
    uint32_t num_cores,
    uint32_t local_scratch_addr) {
    // Get the L1 address of the barrier semaphore (same logical address on all cores)
    uint32_t barrier_sem_addr = get_semaphore(barrier_sem_id);

    // Get NOC address of coordinator's barrier semaphore
    uint64_t barrier_noc_addr = get_noc_addr(barrier_coord_x, barrier_coord_y, barrier_sem_addr);

    // Increment the coordinator's semaphore to signal this core has arrived
    noc_semaphore_inc(barrier_noc_addr, 1);
    noc_async_atomic_barrier();

    // Poll the coordinator's semaphore until all cores have arrived
    // We read from the coordinator's L1 into our local scratch space
    volatile tt_l1_ptr uint32_t* local_poll_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_addr);
    *local_poll_ptr = 0;

    while (*local_poll_ptr < num_cores) {
        // Read coordinator's semaphore value into local memory
        noc_async_read(barrier_noc_addr, local_scratch_addr, sizeof(uint32_t));
        noc_async_read_barrier();
    }
}
