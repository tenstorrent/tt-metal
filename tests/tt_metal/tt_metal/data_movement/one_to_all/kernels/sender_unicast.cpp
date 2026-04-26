// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// L1 to L1 send
void kernel_main() {
#ifdef ARCH_QUASAR
    // Quasar: use named compile-time args (Metal 2.0 API)
    uint32_t mst_base_addr = get_named_compile_time_arg_val("mst_base_addr");
    uint32_t sub_base_addr = get_named_compile_time_arg_val("sub_base_addr");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t pages_per_transaction = get_named_compile_time_arg_val("pages_per_tx");
    constexpr uint32_t bytes_per_page = get_named_compile_time_arg_val("bytes_per_page");
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_subordinates = get_named_compile_time_arg_val("num_subordinates");
    constexpr uint32_t num_virtual_channels = get_named_compile_time_arg_val("num_vc");
#else
    // WH/BH: use indexed compile-time args (legacy API)
    uint32_t mst_base_addr = get_compile_time_arg_val(0);
    uint32_t sub_base_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(4);
    constexpr uint32_t test_id = get_compile_time_arg_val(5);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(6);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(7);
#endif

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
                // Cycle through virtual channels 0 to (num_virtual_channels - 1)
                uint32_t current_virtual_channel = i % num_virtual_channels;
                noc_async_write(mst_base_addr, dst_noc_addr, bytes_per_transaction, noc_index, current_virtual_channel);
            }
        }

        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_subordinates);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Number of subordinates", num_subordinates);
}
