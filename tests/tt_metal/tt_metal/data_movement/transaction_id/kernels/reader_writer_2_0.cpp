// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    constexpr uint32_t l1_local_addr = get_arg(args::l1_addr);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t packed_sub0_core_coordinates = get_arg(args::sub0_coords);
    constexpr uint32_t packed_sub1_core_coordinates = get_arg(args::sub1_coords);

    const uint32_t num_transactions_raw = get_arg(args::num_transactions);
    const uint32_t num_of_transactions = num_transactions_raw < 16 ? num_transactions_raw : 15;
    const uint32_t bytes_per_transaction = get_arg(args::bytes_per_transaction);

    const uint32_t sub0_receiver_x_coord = packed_sub0_core_coordinates >> 16;
    const uint32_t sub0_receiver_y_coord = packed_sub0_core_coordinates & 0xFFFF;
    const uint32_t sub1_sender_x_coord = packed_sub1_core_coordinates >> 16;
    const uint32_t sub1_sender_y_coord = packed_sub1_core_coordinates & 0xFFFF;

    Noc noc(noc_index);
    UnicastEndpoint endpoint;

    {
        DeviceZoneScopedN("RISCV0");

        uint32_t local_addr = l1_local_addr;
        uint32_t sub1_offset = 0;

        // Send out reads with transaction ids
        // Avoid using transaction id 0 in case fast dispatch breaks it in the future
        for (uint32_t i = 1; i <= num_of_transactions; i++) {
            noc.async_read<NocOptions::TXN_ID>(
                endpoint,
                endpoint,
                bytes_per_transaction,
                {.noc_x = sub1_sender_x_coord, .noc_y = sub1_sender_y_coord, .addr = l1_local_addr + sub1_offset},
                {.addr = local_addr},
                NocOptVals{.trid = i});
            local_addr += bytes_per_transaction;
            sub1_offset += bytes_per_transaction;
        }

        local_addr = l1_local_addr;
        uint32_t sub0_offset = 0;

        // Wait for reads with transaction ids to finish, then write
        for (uint32_t i = 1; i <= num_of_transactions; i++) {
            noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = i});
            noc.async_write(
                endpoint,
                endpoint,
                bytes_per_transaction,
                {.addr = local_addr},
                {.noc_x = sub0_receiver_x_coord, .noc_y = sub0_receiver_y_coord, .addr = l1_local_addr + sub0_offset});
            local_addr += bytes_per_transaction;
            sub0_offset += bytes_per_transaction;
        }
        noc.async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions * 2);  // 2 because of the write and read
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}
