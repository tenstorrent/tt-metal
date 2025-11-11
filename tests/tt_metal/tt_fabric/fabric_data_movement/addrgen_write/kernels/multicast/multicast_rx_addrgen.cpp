// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// Receiver kernel for multicast test
// Just waits on a global semaphore to signal completion
void kernel_main() {
    // RT args: 0: sem_l1_addr, 1: sem_wait_value
    size_t idx = 0;
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_wait_value = get_arg_val<uint32_t>(idx++);

    // Wait for semaphore (fabric sender will increment it)
    noc_semaphore_wait(reinterpret_cast<volatile uint32_t*>(sem_l1_addr), sem_wait_value);
}
