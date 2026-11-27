// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "risc_common.h"

// Spends a while on self-addressed NOC atomics, re-reading L1 each step as a polling kernel does, then writes a
// value to its own L1. A read queued after the program sees the value only if dispatch waited for this core.
void kernel_main() {
    const uint32_t address = get_arg_val<uint32_t>(0);
    const uint32_t scratch = get_arg_val<uint32_t>(1);
    const uint32_t core_x = get_arg_val<uint32_t>(2);
    const uint32_t core_y = get_arg_val<uint32_t>(3);
    const uint32_t iterations = get_arg_val<uint32_t>(4);
    const uint32_t value = get_arg_val<uint32_t>(5);

    const uint64_t scratch_noc_addr = get_noc_addr(core_x, core_y, scratch);
    for (uint32_t i = 0; i < iterations; ++i) {
        noc_semaphore_inc(scratch_noc_addr, 1);
        noc_async_atomic_barrier();
        invalidate_l1_cache();
    }

    CoreLocalMem<uint32_t> output(address);
    output[0] = value;
    flush_l2_cache_line(address);
}
