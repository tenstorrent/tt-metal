// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t expected_value = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);

    // Create NOC 2.0 API objects
    experimental::Noc noc(noc_index);
    experimental::Semaphore semaphore(sem_id);

    {
        DeviceZoneScopedN("RISCV1");
        semaphore.wait(expected_value);
    }

    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", 1);
    DeviceTimestampedData("Test id", test_id);
}
