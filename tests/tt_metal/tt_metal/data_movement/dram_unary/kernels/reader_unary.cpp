// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// DRAM to L1 read
void kernel_main() {
    uint32_t src_addr = get_compile_time_arg_val(0);
    constexpr uint32_t bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(5);
    constexpr uint32_t test_id = get_compile_time_arg_val(6);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;
    constexpr uint32_t total_num_pages = num_of_transactions * transaction_num_pages;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    cb_reserve_back(cb_id_in0, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

            noc_async_read(src_noc_addr, l1_write_addr, transaction_size_bytes);

            src_addr += transaction_size_bytes;
            l1_write_addr += transaction_size_bytes;
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_id_in0, 1);
}
