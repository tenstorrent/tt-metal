// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

// To each destination in turn over one connection: the cable's, or (behind) a router's toward the exit chip, whose
// route ends out the cable.
void kernel_main() {
    size_t i = 0;
    const uint32_t src_addr = get_arg_val<uint32_t>(i++);
    const uint32_t header_addr = get_arg_val<uint32_t>(i++);
    const uint32_t num_packets = get_arg_val<uint32_t>(i++);
    const uint32_t packet_size = get_arg_val<uint32_t>(i++);
    const uint32_t behind = get_arg_val<uint32_t>(i++);
    const uint32_t exit_chip = get_arg_val<uint32_t>(i++);
    const uint32_t exit_mesh = get_arg_val<uint32_t>(i++);
    const uint32_t out_cable = get_arg_val<uint32_t>(i++);  // the perimeter direction's action bit
    auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
    for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t); w++) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header_addr)[w] = 0;
    }
    if (behind) {
        fabric_set_unicast_route(header, exit_chip, exit_mesh);
        for (uint32_t b = 0; b < sizeof(header->route_buffer); b++) {
            if (header->route_buffer[b] & Routing2DCodec::ACTION_LOCAL_DELIVER) {
                header->route_buffer[b] = out_cable;
            }
        }
    }
    const uint32_t num_dsts = get_arg_val<uint32_t>(i++);
    const size_t dsts = i;
    i += num_dsts * 5;
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i);

    sender.open();
    for (uint32_t d = 0; d < num_dsts; d++) {
        const size_t a = dsts + d * 5;
        const uint32_t dst_x = get_arg_val<uint32_t>(a);
        const uint32_t dst_y = get_arg_val<uint32_t>(a + 1);
        const uint32_t dst_addr = get_arg_val<uint32_t>(a + 2);
        header->dst_start_mesh_id = get_arg_val<uint32_t>(a + 3);
        header->dst_start_chip_id = get_arg_val<uint32_t>(a + 4);
        for (uint32_t p = 0; p < num_packets; p++) {
            const uint32_t offset = p * packet_size;
            header->to_noc_unicast_write(
                tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(dst_x, dst_y, dst_addr + offset)}, packet_size);
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(src_addr + offset, packet_size);
            sender.send_payload_flush_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
        }
    }
    sender.close();
}
