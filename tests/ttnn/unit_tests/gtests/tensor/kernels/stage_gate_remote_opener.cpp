// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stand-in for a remote KVM opening a D2D stage gate over fabric. Everything it needs comes
// from the receiver's D2DStageGateDescriptor (fabric node, service-core NoC coords, gate word
// address), so it exercises the same contract a KVM on another chip/host would use.
//
// Two opener flavours (both are valid per the gate contract: the opener only ever writes a
// CLOSED gate, so +1 and "store OPEN" are equivalent):
//   * mode 0: inline write of kStageGateOpen (no flush; delivery is ordered behind nothing).
//   * mode 1: atomic +1 with flush (the write is acked before the kernel returns).
//
// RT layout:
//   [0] mode
//   [1] gate_noc_x   (descriptor.service_core_noc.x)
//   [2] gate_noc_y
//   [3] gate_addr    (descriptor.gate_base_addr + gate * descriptor.gate_stride_bytes)
//   [4] dst_dev_id   (descriptor.chip_id)
//   [5] dst_mesh_id  (descriptor.mesh_id)
//   then the append_fabric_connection_rt_args block.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"  // tt::tt_fabric::WorkerToFabricEdmSender
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "ttnn/api/ttnn/tensor/d2d_stage_gate.hpp"

void kernel_main() {
    size_t rt_idx = 0;
    const uint32_t mode = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t gate_noc_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t gate_noc_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t gate_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dst_dev_id = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(rt_idx++);

    auto connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_idx);
    connection.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    tt::tt_fabric::fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
    const uint64_t gate_noc_addr = get_noc_addr(gate_noc_x, gate_noc_y, gate_addr);
    if (mode == 0) {
        header->to_noc_unicast_inline_write(
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{gate_noc_addr, ttnn::kStageGateOpen});
    } else {
        header->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{gate_noc_addr, 1, /*flush=*/true});
    }
    connection.wait_for_empty_write_slot();
    connection.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
    connection.close();
}
