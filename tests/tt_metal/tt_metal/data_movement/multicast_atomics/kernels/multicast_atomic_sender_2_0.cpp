// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t atomic_inc_value = get_compile_time_arg_val(2);
    constexpr uint32_t num_dests = get_compile_time_arg_val(3);
    uint32_t dst_start_x = get_compile_time_arg_val(4);
    uint32_t dst_start_y = get_compile_time_arg_val(5);
    uint32_t dst_end_x = get_compile_time_arg_val(6);
    uint32_t dst_end_y = get_compile_time_arg_val(7);
    constexpr uint32_t test_id = get_compile_time_arg_val(8);

    // For NOC_1, the coordinate system is inverted, so start/end need to be swapped
    if (noc_index == 1) {
        std::swap(dst_start_x, dst_end_x);
        std::swap(dst_start_y, dst_end_y);
    }

    experimental::Noc noc(noc_index);
    experimental::Semaphore semaphore(sem_id);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            semaphore.inc_multicast(noc, dst_start_x, dst_start_y, dst_end_x, dst_end_y, atomic_inc_value, num_dests);
        }
        noc.async_atomic_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", 4);
    DeviceTimestampedData("Test id", test_id);
}
