// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "api/debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t packed_subordinate_core_coordinates = get_compile_time_arg_val(4);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(5);

    // Runtime arguments
    uint32_t receiver_x_coord = packed_subordinate_core_coordinates >> 16;
    uint32_t receiver_y_coord = packed_subordinate_core_coordinates & 0xFFFF;

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Cycle through virtual channels 0 to (num_virtual_channels - 1)
            uint32_t current_virtual_channel = i % num_virtual_channels;
            noc.async_write(
                unicast_endpoint,
                unicast_endpoint,
                bytes_per_transaction,
                {
                    .addr = l1_local_addr,
                },
                {
                    .noc_x = receiver_x_coord,
                    .noc_y = receiver_y_coord,
                    .addr = l1_local_addr,
                },
                current_virtual_channel);
        }
        noc.async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("NoC Index", noc.get_noc_id());

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);

    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
}
