// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// L1 to DRAM write
void kernel_main() {
    uint32_t dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(5);
    constexpr uint32_t test_id = get_compile_time_arg_val(6);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;
    constexpr uint32_t total_num_pages = num_of_transactions * transaction_num_pages;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    cb_wait_front(cb_id_out0, 1);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    {
        DeviceZoneScopedN("RISCV0");
        uint64_t dst_base_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, 0);
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            /* The 64-bit NOC addresses consists of a 32-bit local address and a NOC XY coordinate. The local address
             * occupies the lower 32 bits and the NOC XY coordinate occupies the next 12 (unicast) to 24 (multicast)
             * bits. In the get_noc_addr call, we set the local address to 0 to get the base address. Then, we OR it
             * with the local address (dst_addr) in each iteration to get the full NOC address. */
            uint64_t dst_noc_addr = dst_base_noc_addr | dst_addr;

            noc_async_write(l1_read_addr, dst_noc_addr, transaction_size_bytes);

            l1_read_addr += transaction_size_bytes;
            dst_addr += transaction_size_bytes;
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_id_out0, 1);
}
