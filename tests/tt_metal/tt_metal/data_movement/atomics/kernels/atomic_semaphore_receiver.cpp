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
    const uint32_t expected_atomic_inc_value = get_compile_time_arg_val(2);
    const uint32_t test_id = get_compile_time_arg_val(3);
    const uint32_t semaphore_addr_offset = get_compile_time_arg_val(4);

    // Set up local semaphore for receiving atomic notifications
    // Note that semaphore is initialized by the host, to guarantee that it is ready when
    // sender starts without impacting the performance.
    volatile uint32_t* local_semaphore =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_local_addr + semaphore_addr_offset);

    DeviceTimestampedData("Receiver Initial semaphore value", *local_semaphore);
    uint32_t expected_semaphore_value = num_of_transactions * expected_atomic_inc_value;
    {
        DeviceZoneScopedN("RISCV0");

        noc_semaphore_wait(local_semaphore, expected_semaphore_value);
    }

    //    DeviceTimestampedData("Test id", test_id);
    //    DeviceTimestampedData("NoC Index", noc_index);
    //    DeviceTimestampedData("Number of transactions processed", num_of_transactions);
    //    DeviceTimestampedData("Transaction size in bytes", expected_atomic_inc_value);
    //    DeviceTimestampedData("Receiver Final semaphore value", *local_semaphore);
}
