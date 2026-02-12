// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t expected_value = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);

    // Get semaphore L1 address from semaphore ID
    uint32_t semaphore_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* local_semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    {
        DeviceZoneScopedN("RISCV1");
        noc_semaphore_wait(local_semaphore, expected_value);
    }

    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", 1);
    DeviceTimestampedData("Test id", test_id);
}
