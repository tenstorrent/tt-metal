// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time arguments
    const uint32_t semaphore_addr = get_compile_time_arg_val(0);
    const uint32_t expected_value = get_compile_time_arg_val(1);
    const uint32_t test_id = get_compile_time_arg_val(2);

    // Note: semaphore is initialized by the host before kernel launch
    volatile uint32_t* local_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    {
        DeviceZoneScopedN("RISCV1");
        // Wait for semaphore to reach expected value
        noc_semaphore_wait(local_semaphore, expected_value);
    }

    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", 1);
    DeviceTimestampedData("Test id", test_id);
}
