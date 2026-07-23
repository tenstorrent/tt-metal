// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

/**
 * @brief Wait for all cores to reach the barrier before proceeding.
 *
 * Global barrier synchronization (mirrors the multi_interleaved helper):
 * 1. Each core increments a semaphore on the coordinator core.
 * 2. Each core polls the coordinator's semaphore until it equals num_cores.
 * 3. Once all cores have arrived, they all proceed simultaneously.
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
    uint32_t barrier_sem_addr = get_semaphore(barrier_sem_id);
    uint64_t barrier_noc_addr = get_noc_addr(barrier_coord_x, barrier_coord_y, barrier_sem_addr);

    // Signal arrival on the coordinator's semaphore.
    noc_semaphore_inc(barrier_noc_addr, 1);
    noc_async_atomic_barrier();

    // Spin until all cores have arrived.
    volatile tt_l1_ptr uint32_t* local_poll_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_addr);
    *local_poll_ptr = 0;
    while (*local_poll_ptr < num_cores) {
        noc_async_read(barrier_noc_addr, local_scratch_addr, sizeof(uint32_t));
        noc_async_read_barrier();
    }
}
