// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Opens a D2D stage gate over fabric using only the receiver's D2DStageGateDescriptor.
// mode 0: inline write of open_value. mode 1: flushed atomic +1. mode 2: 16 B payload write
// covering the gate's slot, open_value first.
//
// Fabric payload writes must be at least 16 B (smaller writes go as inline writes), so
// mode 2 writes the gate's whole slot.
//
// RT args: mode, gate_noc_x, gate_noc_y, gate_addr, dst_dev_id, dst_mesh_id, open_value,
// then the fabric connection args.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    size_t rt_idx = 0;
    const uint32_t mode = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t gate_noc_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t gate_noc_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t gate_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dst_dev_id = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t open_value = get_arg_val<uint32_t>(rt_idx++);

    auto connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_idx);
    connection.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    tt::tt_fabric::fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
    const uint64_t gate_noc_addr = get_noc_addr(gate_noc_x, gate_noc_y, gate_addr);
    connection.wait_for_empty_write_slot();
    if (mode == 0) {
        header->to_noc_unicast_inline_write(
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{gate_noc_addr, open_value});
    } else if (mode == 1) {
        header->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{gate_noc_addr, 1, /*flush=*/true});
    } else {
        // A spare header slot is aligned L1, so it serves as the payload.
        constexpr uint32_t payload_bytes = 16;
        volatile tt_l1_ptr uint32_t* payload =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PacketHeaderPool::allocate_header());
        payload[0] = open_value;
        for (uint32_t i = 1; i < payload_bytes / sizeof(uint32_t); ++i) {
            payload[i] = 0;
        }
        header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{gate_noc_addr}, payload_bytes);
        connection.send_payload_without_header_non_blocking_from_address(
            reinterpret_cast<uint32_t>(payload), payload_bytes);
    }
    connection.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
    connection.close();
}
