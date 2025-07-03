// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    // Compile-time arguments
    uint32_t mst_base_addr = get_compile_time_arg_val(0);
    uint32_t sub_base_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(6);
    constexpr bool is_linked = get_compile_time_arg_val(7);
    constexpr bool loopback = get_compile_time_arg_val(8);
    constexpr uint32_t start_x = get_compile_time_arg_val(9);
    constexpr uint32_t start_y = get_compile_time_arg_val(10);
    constexpr uint32_t end_x = get_compile_time_arg_val(11);
    constexpr uint32_t end_y = get_compile_time_arg_val(12);

    // Derivative values
    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;
    constexpr uint32_t bytes = bytes_per_transaction * num_of_transactions;

    uint64_t dst_noc_addr_multicast = get_noc_multicast_addr(start_x, start_y, end_x, end_y, sub_base_addr);

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
            if constexpr (loopback) {
                noc_async_write_multicast_loopback_src(
                    mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, is_linked);
            } else {
                noc_async_write_multicast(
                    mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, is_linked);
            }
        }
        // Last packet is sent separately to unlink the transaction,
        // so the next one can use the VC and do its own path reservation
        if constexpr (loopback) {
            noc_async_write_multicast_loopback_src(
                mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, false);
        } else {
            noc_async_write_multicast(
                mst_base_addr, dst_noc_addr_multicast, bytes_per_transaction, num_subordinates, false);
        }
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Test id", test_id);
}
