// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// L1 to L1 send
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_trids = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t packed_sub0_core_coordinates = get_compile_time_arg_val(4);
    constexpr uint32_t packed_sub1_core_coordinates = get_compile_time_arg_val(5);

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
        noc_async_write_one_packet_with_trid_set_state(sub0_dst_noc_addr);
        for (uint32_t i = 0; i < num_of_trids; i++) {
            noc_async_write_one_packet_with_trid_with_state(tmp_local_addr, tmp_local_addr, bytes_per_transaction, i);
            tmp_local_addr += bytes_per_transaction;
        }

        tmp_local_addr = l1_local_addr;

        // Wait for writes with transaction ids to depart
        noc_async_read_one_packet_set_state(sub1_src_noc_addr, bytes_per_transaction);
        for (uint32_t i = 0; i < num_of_trids; i++) {
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
