// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// L1 to L1 request
void kernel_main() {
    constexpr uint32_t l1_local_addr = get_named_compile_time_arg_val("l1_addr");
    constexpr uint32_t num_of_transactions = get_named_compile_time_arg_val("num_transactions");
    constexpr uint32_t transaction_size_bytes = get_named_compile_time_arg_val("tx_size");
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_virtual_channels = get_named_compile_time_arg_val("num_vc");

    uint32_t responder_x_coord = get_arg_val<uint32_t>(0);
    uint32_t responder_y_coord = get_arg_val<uint32_t>(1);

    {
        DeviceZoneScopedN("RISCV1");
        uint64_t src_noc_addr = get_noc_addr(responder_x_coord, responder_y_coord, l1_local_addr);
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            // Cycle through virtual channels 0 to (num_virtual_channels - 1)
            uint32_t current_virtual_channel = i % num_virtual_channels;
            noc_async_read(src_noc_addr, l1_local_addr, transaction_size_bytes, noc_index, current_virtual_channel);
        }
        noc_async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);

    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
}
