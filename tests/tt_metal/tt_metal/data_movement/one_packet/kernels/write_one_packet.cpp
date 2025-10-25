// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint_pages.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t num_packets = get_compile_time_arg_val(0);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);

    uint32_t master_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t subordinate_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t responder_x_coord = get_arg_val<uint32_t>(2);
    uint32_t responder_y_coord = get_arg_val<uint32_t>(3);

    DeviceTimestampedData("Number of transactions", num_packets);
    DeviceTimestampedData("Transaction size in bytes", packet_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV0");
        uint64_t dst_noc_addr = get_noc_addr(responder_x_coord, responder_y_coord, subordinate_l1_addr);
        noc_async_write_one_packet_set_state(dst_noc_addr, packet_size_bytes);
        for (uint32_t i = 0; i < num_packets; i++) {
            noc_async_write_one_packet_with_state(master_l1_addr, subordinate_l1_addr);
        }
        noc_async_write_barrier();
    }
}
