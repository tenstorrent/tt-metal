// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Opens a D2D stage gate over fabric using only the receiver's D2DStageGateDescriptor.
// mode 0: inline write of open_value. mode 1: flushed atomic +1.
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
    if (mode == 0) {
        header->to_noc_unicast_inline_write(
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{gate_noc_addr, open_value});
    } else {
        header->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{gate_noc_addr, 1, /*flush=*/true});
    }
    connection.wait_for_empty_write_slot();
    connection.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
    connection.close();
}
