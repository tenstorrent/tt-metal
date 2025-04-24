// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    uint32_t src_addr = get_compile_time_arg_val(0);
    uint32_t dst_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);

    uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(0));
    uint32_t receiver_x_coord = get_arg_val<uint32_t>(1);
    uint32_t receiver_y_coord = get_arg_val<uint32_t>(2);

    uint64_t sem_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, semaphore);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            uint64_t dst_noc_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, dst_addr);

            noc_async_write(src_addr, dst_noc_addr, transaction_size_bytes);

            src_addr += transaction_size_bytes;
            dst_addr += transaction_size_bytes;
        }
        noc_async_write_barrier();
    }
    noc_semaphore_inc(sem_addr, 1);
}
