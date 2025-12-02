// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * This kernel tests the scoped_lock functionality of CoreLocalMem.
 * It locks a memory region, writes to it, and then unlocks it.
 * The scoped_lock records MEM_LOCK and MEM_UNLOCK events to the NOC event profiler.
 */
void kernel_main() {
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);

    experimental::CoreLocalMem<uint32_t> buffer(l1_buffer_addr);

    // Acquire a scoped lock on the buffer region
    // This records a MEM_LOCK event to the NOC event profiler
    auto lock = buffer.scoped_lock(num_elements);

    // Write to the locked region
    for (uint32_t i = 0; i < num_elements; i++) {
        buffer[i] = write_value + i;
    }
}
