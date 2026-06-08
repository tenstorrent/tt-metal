// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t l1_local_addr = get_arg(args::l1_addr);
    constexpr uint32_t num_transactions = get_arg(args::num_tx);
    constexpr uint32_t transaction_size = get_arg(args::tx_size);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t packed_dest_core_coordinates = get_arg(args::dest_coords);

    uint32_t dest_x_coord = packed_dest_core_coordinates >> 16;
    uint32_t dest_y_coord = packed_dest_core_coordinates & 0xFFFF;

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_transactions; i++) {
            noc.async_write(
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
