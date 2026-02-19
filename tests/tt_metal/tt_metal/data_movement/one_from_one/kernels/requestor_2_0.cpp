// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "api/debug/dprint.h"

// L1 to L1 request
void kernel_main() {
    constexpr uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(4);

    uint32_t responder_x_coord = get_arg_val<uint32_t>(0);
    uint32_t responder_y_coord = get_arg_val<uint32_t>(1);

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Cycle through virtual channels 0 to (num_virtual_channels - 1)
            uint32_t current_virtual_channel = i % num_virtual_channels;
            noc.async_read(
                unicast_endpoint,
                unicast_endpoint,
                transaction_size_bytes,
                {
                    .noc_x = responder_x_coord,
                    .noc_y = responder_y_coord,
                    .addr = l1_local_addr,
                },
                {
                    .addr = l1_local_addr,
                },
                current_virtual_channel);
        }
        noc.async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);

    DeviceTimestampedData("NoC Index", noc.get_noc_id());
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
}
