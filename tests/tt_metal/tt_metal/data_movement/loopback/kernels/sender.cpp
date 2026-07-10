// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    constexpr uint32_t src_addr = get_named_compile_time_arg_val("src_addr");
    constexpr uint32_t dst_addr = get_named_compile_time_arg_val("dst_addr");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t transaction_num_pages = get_named_compile_time_arg_val("tx_num_pages");
    constexpr uint32_t page_size_bytes = get_named_compile_time_arg_val("page_size");
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");

    uint32_t semaphore = get_semaphore(get_arg_val<uint32_t>(0));
    uint32_t dest_x = get_arg_val<uint32_t>(1);
    uint32_t dest_y = get_arg_val<uint32_t>(2);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV0");
        uint64_t dst_noc_addr = get_noc_addr(dest_x, dest_y, dst_addr);
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc_async_write(src_addr, dst_noc_addr, transaction_size_bytes);
        }
        noc_async_write_barrier();
    }

    uint64_t sem_addr = get_noc_addr(dest_x, dest_y, semaphore);
    noc_semaphore_inc(sem_addr, 1);
}
