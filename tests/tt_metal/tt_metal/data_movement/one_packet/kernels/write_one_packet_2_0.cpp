// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint.h"
#include "experimental/endpoints.h"

void kernel_main() {
    constexpr uint32_t num_packets = get_compile_time_arg_val(0);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);

    uint32_t master_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t subordinate_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t responder_x_coord = get_arg_val<uint32_t>(2);
    uint32_t responder_y_coord = get_arg_val<uint32_t>(3);

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;

    DeviceTimestampedData("Number of transactions", num_packets);
    DeviceTimestampedData("Transaction size in bytes", packet_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV0");
        noc.set_async_write_state(
            unicast_endpoint, packet_size_bytes, {.noc_x = responder_x_coord, .noc_y = responder_y_coord, .addr = subordinate_l1_addr});

        for (uint32_t i = 0; i < num_packets; i++) {
            noc.async_write_with_state(
                unicast_endpoint,
                unicast_endpoint,
                packet_size_bytes,
                {.addr = master_l1_addr},
                {.noc_x = responder_x_coord, .noc_y = responder_y_coord, .addr = subordinate_l1_addr});
        }
        noc.async_write_barrier();
    }
}
