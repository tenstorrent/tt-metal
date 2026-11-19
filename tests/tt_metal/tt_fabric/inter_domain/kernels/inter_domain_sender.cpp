// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sends a buffer into another fabric domain as unicast writes: straight into an inter-domain router, or through a
// router of this domain with a route that leaves by an inter-domain router further on. The peer's inter-domain router
// lands the packet and routes it to its destination chip in that domain.

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

#include <climits>
#include <type_traits>

// one route entry per hop, packed into runtime-arg words (the host packs the same way)
using RouteEntry = std::remove_extent_t<decltype(PACKET_HEADER_TYPE::route_buffer)>;
constexpr uint32_t kEntriesPerArg = sizeof(uint32_t) / sizeof(RouteEntry);
constexpr uint32_t kEntryBits = sizeof(RouteEntry) * CHAR_BIT;

void kernel_main() {
    size_t i = 0;
    const uint32_t dst_x = get_arg_val<uint32_t>(i++);
    const uint32_t dst_y = get_arg_val<uint32_t>(i++);
    const uint32_t dst_addr = get_arg_val<uint32_t>(i++);
    const uint32_t src_addr = get_arg_val<uint32_t>(i++);
    const uint32_t header_addr = get_arg_val<uint32_t>(i++);
    const uint32_t num_packets = get_arg_val<uint32_t>(i++);
    const uint32_t packet_size = get_arg_val<uint32_t>(i++);
    const uint32_t dst_mesh = get_arg_val<uint32_t>(i++);  // the destination in its own domain
    const uint32_t dst_chip = get_arg_val<uint32_t>(i++);
    const uint32_t route_len = get_arg_val<uint32_t>(i++);  // route entries, kEntriesPerArg per arg

    auto* header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_addr);
    for (uint32_t w = 0; w < sizeof(PACKET_HEADER_TYPE) / sizeof(uint32_t); w++) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(header_addr)[w] = 0;
    }
    header->dst_start_mesh_id = dst_mesh;  // the landing router routes on it in the destination domain
    header->dst_start_chip_id = dst_chip;
    uint32_t word = 0;
    for (uint32_t b = 0; b < route_len; b++) {
        if (b % kEntriesPerArg == 0) {
            word = get_arg_val<uint32_t>(i++);
        }
        header->route_buffer[b] = static_cast<RouteEntry>(word >> (kEntryBits * (b % kEntriesPerArg)));
    }
    auto sender = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i);

    sender.open();
    for (uint32_t p = 0; p < num_packets; p++) {
        const uint32_t offset = p * packet_size;
        header->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(dst_x, dst_y, dst_addr + offset)}, packet_size);
        sender.wait_for_empty_write_slot();
        sender.send_payload_without_header_non_blocking_from_address(src_addr + offset, packet_size);
        sender.send_payload_flush_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
    }
    sender.close();
}
