// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

// Receiver-side completion wait.
// Runs on the destination device and blocks until the receiver's global
// semaphore reaches `expected_value`. In this test, the sender atomically
// increments the semaphore after sending all pages, so reaching the target
// value implies "all data has arrived". Fabric guarantees the semaphore
// signal is delivered after payload data.
//
// CT (compile-time) args: none
// RT (runtime) args:
//   0: completion_sem_addr   (u32)  // L1 address of the global semaphore on receiver
//   1: expected_value        (u32)  // e.g. number of pages, or just 1

void kernel_main() {
    size_t idx = 0;
    const uint32_t sem_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t expected_value = get_arg_val<uint32_t>(idx++);

    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    noc_semaphore_wait(sem_ptr, expected_value);

    // Reset for next iteration so we always observe a fresh 0→1 transition.
    // (Prevents the next run from passing the wait immediately on a stale '1'.)
    noc_semaphore_set(sem_ptr, 0);
}
