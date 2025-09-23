// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define PROFILE_KERNEL 1

#include "dataflow_api.h"

// Atomic semaphore receiver kernel that waits for atomic notifications
void kernel_main() {
    // Compile-time arguments
    const uint32_t l1_local_addr = get_compile_time_arg_val(0);
    const uint32_t num_of_transactions = get_compile_time_arg_val(1);
    const uint32_t bytes_per_transaction = get_compile_time_arg_val(2);
    const uint32_t test_id = get_compile_time_arg_val(3);
    const uint32_t semaphore_addr_offset = get_compile_time_arg_val(4);
    const uint32_t expected_atomic_inc_value = get_compile_time_arg_val(5);

    // Set up local semaphore for receiving atomic notifications
    volatile uint32_t* local_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_local_addr + semaphore_addr_offset);
    *local_semaphore = 0;

    DeviceZoneScopedN("AtomicReceiver");

    uint32_t expected_semaphore_value = 0;

    /*    for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Calculate expected semaphore value after this transaction
            expected_semaphore_value += expected_atomic_inc_value;

            // Wait for the semaphore to reach the expected value
            // This indicates that the sender has completed the atomic increment
            while (*local_semaphore < expected_semaphore_value) {
                // Poll the semaphore value
                // In a real scenario, this could be doing other work
                asm volatile("nop" : : : "memory");
            }

            // Validate data received (simple check - verify non-zero data)
            volatile uint32_t* data_ptr = reinterpret_cast<volatile uint32_t*>(l1_local_addr);
            uint32_t data_check = 0;
            for (uint32_t j = 0; j < bytes_per_transaction / sizeof(uint32_t); j++) {
                data_check |= data_ptr[j];
            }

            // If no data received, this could indicate a problem
            if (data_check == 0 && i == 0) {
                DeviceTimestampedData("Warning: No data detected", i);
            }
        }
        */
    noc_semaphore_wait(local_semaphore, expected_semaphore_value);

    DPRINT << "Receiver Test ID" << test_id << ENDL();
    DeviceTimestampedData("Receiver Test ID", test_id);
    DeviceTimestampedData("Receiver Number of transactions processed", num_of_transactions);
    DeviceTimestampedData("Receiver Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Receiver Final semaphore value", *local_semaphore);
    DeviceTimestampedData("Receiver Expected final value", expected_semaphore_value);

    //    DeviceTimestampedData("Test ID", test_id);
    //    DeviceTimestampedData("Number of transactions processed", num_of_transactions);
    //    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    //    DeviceTimestampedData("Final semaphore value", *local_semaphore);
    //    DeviceTimestampedData("Expected final value", expected_semaphore_value);
}
