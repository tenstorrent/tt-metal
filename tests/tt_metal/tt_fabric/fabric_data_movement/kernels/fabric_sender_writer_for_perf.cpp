// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt;
using namespace tt::tt_fabric;

// Compile-time args (CT):
//   0: TOTAL_PAGES   (u32)  -- how many pages to send (should match reader’s NUM_PAGES)
//   1: PAGE_SIZE     (u32)  -- bytes per page (should match reader’s PAGE_SIZE)
//
// Runtime args (RT) in this order:
//   0: dest_noc_addr_lo     (u32)
//   1: dest_noc_addr_hi     (u32)
//   2: dst_mesh_id          (u32)  logical mesh id (will be truncated to u16)
//   3: dst_dev_id           (u32)  logical device id in that mesh (u16)
//   [then]: WorkerToFabricEdmSender RT args (append_fabric_connection_rt_args adds these)

void kernel_main() {
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    size_t idx = 0;
    const uint32_t dest_noc_lo = get_arg_val<uint32_t>(idx++);
    const uint32_t dest_noc_hi = get_arg_val<uint32_t>(idx++);
    const uint64_t dest_noc_addr = (uint64_t(dest_noc_hi) << 32) | uint64_t(dest_noc_lo);

    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    // Build fabric connection from remaining RT args
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // Reusable packet header in L1
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    zero_l1_buf((uint32_t*)header, sizeof(PACKET_HEADER_TYPE));

    // NOC header: unicast write of PAGE_SIZE bytes to the destination NOC address
    header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);

    // Fabric header (2D dynamic routing): set destination mesh/device on the header.
    // This call ignores outgoing dir / my_dev / ew_dim under dynamic routing.
    // Cast to the MeshPacketHeader form expected by the 2D API helper.
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt_l1_ptr MeshPacketHeader*>(header),
        eth_chan_directions::EAST,  // ignored for dynamic routing
        /*my_dev_id*/ 0,            // ignored
        /*dst_dev_id*/ dst_dev_id,
        /*dst_mesh_id*/ dst_mesh_id,
        /*ew_dim*/ 0  // ignored
    );

    sender.open();

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        // Wait until there is at least 1 page available in the CB
        cb_wait_front(CB_ID, 1);

        // Pointer to the page the reader just filled
        uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Make sure TX path has space
        sender.wait_for_empty_write_slot();

        // 1) send payload (no header)
        sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);

        // 2) send header (completes packet)
        sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

        // Return the page to the CB
        cb_pop_front(CB_ID, 1);
    }

    sender.close();
}
