// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

// L1 to L1 send (one packet, stateful + transaction-ID).
//
// NOTE: This kernel intentionally stays on the legacy NOC free functions
// because the new Noc/UnicastEndpoint API does not yet expose a combined
// "with_state + with_trid" form. The new API provides:
//   - stateful only:        set_async_write_state + async_write_with_state
//   - transaction-id only:  async_write<TxnIdMode::ENABLED>
// but not the union of the two that this test exercises. Once the new API
// surface adds a stateful+trid combined call, migrate this kernel to it.
void kernel_main() {
    constexpr uint32_t l1_local_addr = get_arg(args::l1_addr);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t packed_sub0_core_coordinates = get_arg(args::sub0_coords);
    constexpr uint32_t packed_sub1_core_coordinates = get_arg(args::sub1_coords);

    const uint32_t num_trids_raw = get_arg(args::num_transactions);
    const uint32_t num_of_trids = num_trids_raw < 16 ? num_trids_raw : 15;
    const uint32_t bytes_per_transaction = get_arg(args::bytes_per_transaction);

    // Runtime arguments
    uint32_t sub0_receiver_x_coord = packed_sub0_core_coordinates >> 16;
    uint32_t sub0_receiver_y_coord = packed_sub0_core_coordinates & 0xFFFF;
    uint32_t sub1_sender_x_coord = packed_sub1_core_coordinates >> 16;
    uint32_t sub1_sender_y_coord = packed_sub1_core_coordinates & 0xFFFF;

    uint64_t sub0_dst_noc_addr = get_noc_addr(sub0_receiver_x_coord, sub0_receiver_y_coord, l1_local_addr);
    uint64_t sub1_src_noc_addr = get_noc_addr(sub1_sender_x_coord, sub1_sender_y_coord, l1_local_addr);

    {
        DeviceZoneScopedN("RISCV0");

        uint32_t tmp_local_addr = l1_local_addr;

        // Send out writes with transaction ids
        // Avoid using transaction id 0 in case fast dispatch breaks it in the future
        noc_async_write_one_packet_with_trid_set_state(sub0_dst_noc_addr);
        for (uint32_t i = 1; i <= num_of_trids; i++) {
            noc_async_write_one_packet_with_trid_with_state(tmp_local_addr, tmp_local_addr, bytes_per_transaction, i);
            tmp_local_addr += bytes_per_transaction;
        }

        tmp_local_addr = l1_local_addr;

        // Wait for writes with transaction ids to depart
        noc_async_read_one_packet_set_state(sub1_src_noc_addr, bytes_per_transaction);
        for (uint32_t i = 1; i <= num_of_trids; i++) {
            noc_async_write_flushed_with_trid(i);
            noc_async_read_one_packet_with_state(tmp_local_addr, tmp_local_addr);
            tmp_local_addr += bytes_per_transaction;
        }
        noc_async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_trids * 2);  // 2 because of the write and read
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
}
