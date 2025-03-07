// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

bool terminate_fabric_endpoints_farthest_to_nearest(
    tt::tt_fabric::WorkerToFabricEdmSender& sender, size_t a_packet_header_addr, size_t arg_idx) {
    bool closed = false;
    size_t num_endpoints_to_terminate = get_arg_val<uint32_t>(arg_idx++);
    for (size_t i = 0; i < num_endpoints_to_terminate; i++) {
        size_t edm_noc_x = get_arg_val<uint32_t>(arg_idx++);
        size_t edm_noc_y = get_arg_val<uint32_t>(arg_idx++);
        size_t distance = get_arg_val<uint32_t>(arg_idx++);
        size_t termination_addr = get_arg_val<uint32_t>(arg_idx++);

        if (!closed && distance == 0) {
            closed = true;
            sender.close();
        }
        uint64_t termination_sig_noc_addr = get_noc_addr(edm_noc_x, edm_noc_y, termination_addr);
        if (distance == 0) {
            noc_inline_dw_write(
                get_noc_addr(edm_noc_x, edm_noc_y, termination_addr),
                tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE);
        } else {
            auto& packet_header = *reinterpret_cast<PACKET_HEADER_TYPE*>(a_packet_header_addr);
            reinterpret_cast<volatile uint32_t*>(a_packet_header_addr)[sizeof(PACKET_HEADER_TYPE) >> 2] =
                tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE;
            sender.wait_for_empty_write_slot();
            packet_header.to_chip_unicast(static_cast<uint8_t>(distance))
                .to_noc_unicast_write(
                    tt::tt_fabric::NocUnicastCommandHeader{termination_sig_noc_addr},
                    sizeof(PACKET_HEADER_TYPE) + sizeof(uint32_t));
            sender.send_payload_blocking_from_address(a_packet_header_addr, packet_header.get_payload_size_including_header());
            noc_async_writes_flushed();
        }
    }

    return closed;
}
