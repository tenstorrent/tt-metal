// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define PROFILE_KERNEL 1

#include "api/dataflow/dataflow_api.h"

// Atomic semaphore operations test kernel
void kernel_main() {
    // Compile-time arguments
    const uint32_t l1_local_addr = get_compile_time_arg_val(0);
    const uint32_t num_of_transactions = get_compile_time_arg_val(1);
    const uint32_t atomic_inc_value = get_compile_time_arg_val(2);
    const uint32_t test_id = get_compile_time_arg_val(3);
    const uint32_t packed_subordinate_core_coordinates = get_compile_time_arg_val(4);
    const uint32_t semaphore_addr_offset = get_compile_time_arg_val(5);

    // Unpack coordinates
    uint32_t receiver_x_coord = packed_subordinate_core_coordinates >> 16;
    uint32_t receiver_y_coord = packed_subordinate_core_coordinates & 0xFFFF;

    // Set up NOC addresses
    uint64_t dst_semaphore_noc_addr =
        get_noc_addr(receiver_x_coord, receiver_y_coord, l1_local_addr + semaphore_addr_offset);

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Send atomic increment to remote semaphore
            noc_semaphore_inc(dst_semaphore_noc_addr, atomic_inc_value);

            // Wait for atomic operation to complete before proceeding
            noc_async_atomic_barrier();
        }
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    // Even though this is the atomic increment value, printing "Transaction size in bytes" is required for the existing parsing scripts.
    DeviceTimestampedData("Transaction size in bytes", atomic_inc_value);
}
