// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/*
 * This kernel attempts to write to a memory region on another core.
 * It is used in conjunction with scoped_lock_test_kernel to test the
 * scoped lock profiler events by performing writes to a "locked" region.
 */
void kernel_main() {
    uint32_t local_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t num_elements = get_arg_val<uint32_t>(1);
    uint32_t write_value = get_arg_val<uint32_t>(2);
    uint32_t target_noc_x = get_arg_val<uint32_t>(3);
    uint32_t target_noc_y = get_arg_val<uint32_t>(4);
    uint32_t target_addr = get_arg_val<uint32_t>(5);

    experimental::CoreLocalMem<uint32_t> local_buffer(local_buffer_addr);

    // Prepare data in local buffer
    for (uint32_t i = 0; i < num_elements; i++) {
        local_buffer[i] = write_value + i;
    }

    experimental::CoreLocalMem<uint32_t> buffer(local_buffer_addr);

    auto lock = buffer.scoped_lock(num_elements);

    // Write to the target core's memory region via NoC
    // This write targets a region that may be "locked" by another kernel
    uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, target_addr);
    noc_async_write(local_buffer_addr, target_noc_addr, num_elements * sizeof(uint32_t));
    noc_async_write_barrier();
}
