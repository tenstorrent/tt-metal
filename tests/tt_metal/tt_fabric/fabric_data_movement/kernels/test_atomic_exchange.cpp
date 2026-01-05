// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Acquire lock - spin until we get it
inline void acquire_lock(std::atomic<uint32_t>* lock) {
    while (std::atomic_exchange(lock, uint32_t(1)) != 0) {
        // Spin waiting for lock to become available
    }
}

// Release lock
inline void release_lock(std::atomic<uint32_t>* lock) { std::atomic_exchange(lock, uint32_t(0)); }

void kernel_main() {
    // Memory layout (from runtime args):
    // arg0: lock address
    // arg1: counter address
    // arg2: status address (set to 1 when complete)
    // arg3: num_iterations

    uint32_t lock_addr = get_arg_val<uint32_t>(0);
    uint32_t counter_addr = get_arg_val<uint32_t>(1);
    uint32_t status_addr = get_arg_val<uint32_t>(2);
    uint32_t num_iterations = get_arg_val<uint32_t>(3);

    std::atomic<uint32_t>* lock = reinterpret_cast<std::atomic<uint32_t>*>(lock_addr);
    volatile uint32_t* counter = reinterpret_cast<volatile uint32_t*>(counter_addr);
    volatile uint32_t* status = reinterpret_cast<volatile uint32_t*>(status_addr);

    // Initialize status to 0 (in progress)
    *status = 0;

    // Perform multiple lock/unlock cycles to stress test
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Acquire the lock
        acquire_lock(lock);

        // Critical section: increment counter
        uint32_t val = *counter;
        val += 1;
        *counter = val;

        // Release the lock
        release_lock(lock);
    }

    // Mark completion
    *status = 1;
}
