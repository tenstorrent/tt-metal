// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define PROFILE_KERNEL 1

#include "dataflow_api.h"

// Atomic semaphore operations test kernel
void kernel_main() {
    // Compile-time arguments
    const uint32_t l1_local_addr = get_compile_time_arg_val(0);
    const uint32_t num_of_transactions = get_compile_time_arg_val(1);
    const uint32_t bytes_per_transaction = get_compile_time_arg_val(2);
    const uint32_t test_id = get_compile_time_arg_val(3);
    const uint32_t packed_subordinate_core_coordinates = get_compile_time_arg_val(4);
    const uint32_t semaphore_addr_offset = get_compile_time_arg_val(5);
    const uint32_t atomic_inc_value = get_compile_time_arg_val(6);

    // Runtime arguments - unpack coordinates
    uint32_t receiver_x_coord = packed_subordinate_core_coordinates >> 16;
    uint32_t receiver_y_coord = packed_subordinate_core_coordinates & 0xFFFF;

    // Set up NOC addresses
    //    uint64_t dst_data_noc_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, l1_local_addr);
    uint64_t dst_semaphore_noc_addr =
        get_noc_addr(receiver_x_coord, receiver_y_coord, l1_local_addr + semaphore_addr_offset);

    // Initialize local semaphore value for synchronization
    volatile uint32_t* local_semaphore = reinterpret_cast<volatile uint32_t*>(l1_local_addr + semaphore_addr_offset);
    *local_semaphore = 0;

    DeviceZoneScopedN("AtomicSender");

    for (uint32_t i = 0; i < num_of_transactions; i++) {
        // Send data first using regular NOC write
        //        noc_async_write(l1_local_addr, dst_data_noc_addr, bytes_per_transaction, noc_index, 0);

        // Wait for data write to complete
        //        noc_async_write_barrier();

        // Now atomically increment the remote semaphore to signal completion
        noc_semaphore_inc(dst_semaphore_noc_addr, atomic_inc_value);

        // Wait for atomic operation to complete before proceeding
        noc_async_atomic_barrier();

        // For bandwidth measurement, also increment local counter
        //        noc_semaphore_inc(get_noc_addr(0) + l1_local_addr + semaphore_addr_offset, 1);
        //        noc_async_atomic_barrier();
    }

    DPRINT << "Sender Test ID" << test_id << ENDL();
    DeviceTimestampedData("Sender Test ID", test_id);
    DeviceTimestampedData("Sender NOC Index", noc_index);
    DeviceTimestampedData("Sender Number of transactions", num_of_transactions);
    DeviceTimestampedData("Sender Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Sender Atomic increment value", atomic_inc_value);
    DeviceTimestampedData("Sender Local semaphore final value", *local_semaphore);
}
