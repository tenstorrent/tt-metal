// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
 * @param arrival_threshold   Cumulative arrival count to wait for. The coordinator semaphore is
 *                            never reset, so to reuse this barrier N times pass g*num_cores on the
 *                            g-th call (1-based). 0 (default) means a single-use barrier (num_cores).
 */
FORCE_INLINE void barrier_sync(
    uint32_t barrier_sem_id,
    uint32_t barrier_coord_x,
    uint32_t barrier_coord_y,
    uint32_t num_cores,
    uint32_t local_scratch_addr,
    uint32_t arrival_threshold = 0) {
    // Get the L1 address of the barrier semaphore (same logical address on all cores)
    uint32_t barrier_sem_addr = get_semaphore(barrier_sem_id);

    // Get NOC address of coordinator's barrier semaphore
    uint64_t barrier_noc_addr = get_noc_addr(barrier_coord_x, barrier_coord_y, barrier_sem_addr);

    const uint32_t threshold = (arrival_threshold == 0) ? num_cores : arrival_threshold;

    // Increment the coordinator's semaphore to signal this core has arrived
    noc_semaphore_inc(barrier_noc_addr, 1);
    noc_async_atomic_barrier();

    // Poll the coordinator's semaphore until all cores have arrived
    // We read from the coordinator's L1 into our local scratch space
    volatile tt_l1_ptr uint32_t* local_poll_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_addr);
    *local_poll_ptr = 0;

    while (*local_poll_ptr < threshold) {
        // Read coordinator's semaphore value into local memory
        noc_async_read(barrier_noc_addr, local_scratch_addr, sizeof(uint32_t));
        noc_async_read_barrier();
    }
}

/**
 * @brief Wait until a coordinator "progress" semaphore reaches at least `threshold`, to release core
 *        groups in a staggered order (e.g. one column/row of the grid at a time).
 *
 * Same poll mechanism as barrier_sync, but the threshold is a per-core PREFIX count: the number of
 * cores in all groups that must finish before this core's group is allowed to start. threshold==0
 * returns immediately (the first group). Uses a DIFFERENT semaphore than the global barrier so the
 * two counters do not interfere.
 *
 * @param sem_id            Progress-semaphore ID (same L1 offset on all cores).
 * @param coord_x           Physical X of the coordinator core holding the counter.
 * @param coord_y           Physical Y of the coordinator core holding the counter.
 * @param threshold         Cores that must complete before this group starts (prefix sum; 0 = go now).
 * @param local_scratch_addr Local L1 scratch for polling (4 bytes).
 */
FORCE_INLINE void stagger_wait(
    uint32_t sem_id, uint32_t coord_x, uint32_t coord_y, uint32_t threshold, uint32_t local_scratch_addr) {
    if (threshold == 0) {
        return;
    }
    uint32_t sem_addr = get_semaphore(sem_id);
    uint64_t sem_noc_addr = get_noc_addr(coord_x, coord_y, sem_addr);
    volatile tt_l1_ptr uint32_t* local_poll_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_addr);
    *local_poll_ptr = 0;
    while (*local_poll_ptr < threshold) {
        noc_async_read(sem_noc_addr, local_scratch_addr, sizeof(uint32_t));
        noc_async_read_barrier();
    }
}

/**
 * @brief Increment the coordinator's staggered-progress semaphore by 1 to signal this core has
 *        finished its issue phase, releasing the next group once all of this group has signaled.
 */
FORCE_INLINE void stagger_signal(uint32_t sem_id, uint32_t coord_x, uint32_t coord_y) {
    uint32_t sem_addr = get_semaphore(sem_id);
    uint64_t sem_noc_addr = get_noc_addr(coord_x, coord_y, sem_addr);
    noc_semaphore_inc(sem_noc_addr, 1);
    noc_async_atomic_barrier();
}
