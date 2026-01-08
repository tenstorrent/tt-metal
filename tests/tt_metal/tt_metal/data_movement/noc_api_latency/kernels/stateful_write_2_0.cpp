// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_size = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t packed_dest_core_coordinates = get_compile_time_arg_val(4);

    uint32_t dest_x_coord = packed_dest_core_coordinates >> 16;
    uint32_t dest_y_coord = packed_dest_core_coordinates & 0xFFFF;

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;

    noc.set_async_write_state(
        unicast_endpoint, transaction_size, {.noc_x = dest_x_coord, .noc_y = dest_y_coord, .addr = l1_local_addr});

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_transactions; i++) {
            noc.async_write_with_state(
                unicast_endpoint,
                unicast_endpoint,
                transaction_size,
                {.addr = l1_local_addr},
                {.noc_x = dest_x_coord, .noc_y = dest_y_coord, .addr = l1_local_addr});
        }
    }

    noc.async_write_barrier();

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc.get_noc_id());
    DeviceTimestampedData("Number of transactions", num_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size);
}
