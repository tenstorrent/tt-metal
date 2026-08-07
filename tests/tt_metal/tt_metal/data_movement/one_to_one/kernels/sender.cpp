// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t l1_local_addr = get_named_compile_time_arg_val("l1_addr");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_tx");
    constexpr uint32_t bytes_per_transaction = get_named_compile_time_arg_val("tx_size");
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t packed_subordinate_core_coordinates = get_named_compile_time_arg_val("dest_coords");
    constexpr uint32_t num_virtual_channels = get_named_compile_time_arg_val("num_vc");

    // Runtime arguments
    uint32_t receiver_x_coord = packed_subordinate_core_coordinates >> 16;
    uint32_t receiver_y_coord = packed_subordinate_core_coordinates & 0xFFFF;

    uint64_t dst_noc_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, l1_local_addr);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Cycle through virtual channels 0 to (num_virtual_channels - 1)
            uint32_t current_virtual_channel = i % num_virtual_channels;
            noc_async_write(l1_local_addr, dst_noc_addr, bytes_per_transaction, noc_index, current_virtual_channel);
        }
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("NoC Index", noc_index);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
}
