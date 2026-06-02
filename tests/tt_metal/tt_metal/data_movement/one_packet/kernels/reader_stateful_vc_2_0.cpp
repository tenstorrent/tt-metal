// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Verifies set_async_read_state + async_read_with_state with NocOptions::CUSTOM_VC and NocOptVals{.vc}.
// Mirrors read_one_packet_2_0.cpp but explicitly selects a non-default virtual channel via the new
// NocOptVals struct.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t num_packets = get_compile_time_arg_val(0);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t test_id = get_compile_time_arg_val(2);

    uint32_t master_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t subordinate_l1_addr = get_arg_val<uint32_t>(1);
    uint32_t responder_x_coord = get_arg_val<uint32_t>(2);
    uint32_t responder_y_coord = get_arg_val<uint32_t>(3);

    constexpr uint32_t custom_vc = (NOC_UNICAST_WRITE_VC + 1) % 4;

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    DeviceTimestampedData("Number of transactions", num_packets);
    DeviceTimestampedData("Transaction size in bytes", packet_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        noc.set_async_read_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
            unicast_endpoint,
            packet_size_bytes,
            {.noc_x = responder_x_coord, .noc_y = responder_y_coord, .addr = subordinate_l1_addr},
            NocOptVals{.vc = custom_vc});

        for (uint32_t i = 0; i < num_packets; i++) {
            noc.async_read_with_state<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                unicast_endpoint,
                unicast_endpoint,
                packet_size_bytes,
                {.noc_x = responder_x_coord, .noc_y = responder_y_coord, .addr = subordinate_l1_addr},
                {.addr = master_l1_addr},
                NocOptVals{.vc = custom_vc});
        }
        noc.async_read_barrier();
    }
}
