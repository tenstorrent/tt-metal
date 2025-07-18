// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t packed_subordinate_core_coordinates = get_compile_time_arg_val(5);
    constexpr uint32_t virtual_channel = get_compile_time_arg_val(6);

    // Runtime arguments
    uint32_t receiver_x_coord = packed_subordinate_core_coordinates >> 16;
    uint32_t receiver_y_coord = packed_subordinate_core_coordinates & 0xFFFF;

    uint64_t dst_noc_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, l1_local_addr);

    // Derivative values
    uint32_t sem_addr = get_semaphore(sem_id);
    uint64_t sem_noc_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, sem_addr);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc_async_write(l1_local_addr, dst_noc_addr, bytes_per_transaction, virtual_channel);
        }
        noc_async_write_barrier();
    }
    noc_semaphore_inc(sem_noc_addr, 1);

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("Virtual Channel", virtual_channel);
}
