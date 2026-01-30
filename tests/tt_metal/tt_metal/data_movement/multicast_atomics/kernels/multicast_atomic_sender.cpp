// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time arguments
    const uint32_t semaphore_addr = get_compile_time_arg_val(0);
    const uint32_t num_of_transactions = get_compile_time_arg_val(1);
    const uint32_t atomic_inc_value = get_compile_time_arg_val(2);
    const uint32_t num_dests = get_compile_time_arg_val(3);
    const uint32_t dst_start_x = get_compile_time_arg_val(4);
    const uint32_t dst_start_y = get_compile_time_arg_val(5);
    const uint32_t dst_end_x = get_compile_time_arg_val(6);
    const uint32_t dst_end_y = get_compile_time_arg_val(7);
    const uint32_t test_id = get_compile_time_arg_val(8);

    // Get multicast NOC address for the destination grid
    // Note: For NOC_1, the coordinate system is inverted, so start/end need to be swapped
    uint64_t dst_multicast_noc_addr =
        noc_index == 0 ? get_noc_multicast_addr(dst_start_x, dst_start_y, dst_end_x, dst_end_y, semaphore_addr)
                       : get_noc_multicast_addr(dst_end_x, dst_end_y, dst_start_x, dst_start_y, semaphore_addr);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Send multicast atomic increment to all destination cores
            noc_semaphore_inc_multicast(dst_multicast_noc_addr, atomic_inc_value, num_dests);
        }
        // Wait for atomic operation to complete before proceeding
        noc_async_atomic_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", 4);
    DeviceTimestampedData("Test id", test_id);
}
