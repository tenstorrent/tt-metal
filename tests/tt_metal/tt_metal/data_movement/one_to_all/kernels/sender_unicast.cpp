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

    // Derivative values
    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;
    constexpr uint32_t bytes = bytes_per_transaction * num_of_transactions;
    constexpr uint32_t bytes_transferred = bytes * num_subordinates;

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t subordinate_num = 0; subordinate_num < num_subordinates; subordinate_num++) {
            uint32_t dest_coord_packed = get_arg_val<uint32_t>(subordinate_num);
            uint32_t dest_coord_x = dest_coord_packed >> 16;
            uint32_t dest_coord_y = dest_coord_packed & 0xFFFF;

            uint64_t dst_noc_addr = get_noc_addr(dest_coord_x, dest_coord_y, sub_base_addr);

            for (uint32_t i = 0; i < num_of_transactions; i++) {
                noc_async_write(mst_base_addr, dst_noc_addr, bytes_per_transaction);
            }
        }

        noc_async_write_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction * num_subordinates);
    DeviceTimestampedData("Test id", test_id);
}
