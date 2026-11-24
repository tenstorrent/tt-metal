// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified CM sender: stateless/stateful unicast write and scatter write.
// Dispatches at runtime on noc_pkt_type (0=unicast, 4=scatter) and with_state.
//
// Named RT args (indices 0-10):
//   src_l1_addr, dst_noc_x, dst_noc_y,
//   dst_l1_addr_a_0, dst_l1_addr_b_0,
//   dst_l1_addr_a_1, dst_l1_addr_b_1,
//   chunk_size_a, total_size, noc_pkt_type, with_state
//
// ConnectionManager RT args (indices 11-24):
//   Appended by host via append_routing_plane_connection_manager_rt_args (2 slots).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

static constexpr uint32_t HORIZON_NOC_UNICAST_SCATTER_WRITE = 4u;

void kernel_main() {
    const uint32_t src_l1_addr = get_arg(args::src_l1_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_l1_addr_a_0 = get_arg(args::dst_l1_addr_a_0);
    const uint32_t dst_l1_addr_b_0 = get_arg(args::dst_l1_addr_b_0);
    const uint32_t dst_l1_addr_a_1 = get_arg(args::dst_l1_addr_a_1);
    const uint32_t dst_l1_addr_b_1 = get_arg(args::dst_l1_addr_b_1);
    const uint32_t chunk_size_a = get_arg(args::chunk_size_a);
    const uint32_t total_size = get_arg(args::total_size);
    const uint32_t noc_pkt_type = get_arg(args::noc_pkt_type);
    const uint32_t with_state = get_arg(args::with_state);

    std::size_t cm_arg_idx = 11;
    auto cm = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args(cm_arg_idx, 2);
    PacketHeaderPool::allocate_header_n(1);  // route_id 0
    PacketHeaderPool::allocate_header_n(1);  // route_id 1

    using namespace tt::tt_fabric;
    using namespace tt::tt_fabric::mesh::experimental;

    const uint64_t noc_addr_a_0 =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_a_0);
    const uint64_t noc_addr_b_0 =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_b_0);
    const uint64_t noc_addr_a_1 =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_a_1);
    const uint64_t noc_addr_b_1 =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_b_1);

    if (noc_pkt_type == HORIZON_NOC_UNICAST_SCATTER_WRITE) {
        NocUnicastScatterCommandHeader hdr_0({noc_addr_a_0, noc_addr_b_0}, {static_cast<uint16_t>(chunk_size_a)});
        NocUnicastScatterCommandHeader hdr_1({noc_addr_a_1, noc_addr_b_1}, {static_cast<uint16_t>(chunk_size_a)});
        if (with_state) {
            NocUnicastScatterCommandHeader dummy(
                {static_cast<uint64_t>(0), static_cast<uint64_t>(0)}, {static_cast<uint16_t>(chunk_size_a)});
            fabric_unicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                cm, 0, dummy, static_cast<uint16_t>(total_size));
            fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                cm, 0, src_l1_addr, hdr_0);
            fabric_unicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::PayloadSize | UnicastScatterWriteUpdateMask::ChunkSizes>(
                cm, 1, dummy, static_cast<uint16_t>(total_size));
            fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                cm, 1, src_l1_addr, hdr_1);
        } else {
            fabric_unicast_noc_scatter_write<false>(cm, 0, src_l1_addr, total_size, hdr_0);
            fabric_unicast_noc_scatter_write<false>(cm, 1, src_l1_addr, total_size, hdr_1);
        }
    } else {
        if (with_state) {
            fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::All>(
                cm, 0, NocUnicastCommandHeader{noc_addr_a_0}, static_cast<uint16_t>(total_size));
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::None>(cm, 0, src_l1_addr);
            fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::All>(
                cm, 1, NocUnicastCommandHeader{noc_addr_a_1}, static_cast<uint16_t>(total_size));
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::None>(cm, 1, src_l1_addr);
        } else {
            fabric_unicast_noc_unicast_write<false>(
                cm, 0, src_l1_addr, total_size, NocUnicastCommandHeader{noc_addr_a_0});
            fabric_unicast_noc_unicast_write<false>(
                cm, 1, src_l1_addr, total_size, NocUnicastCommandHeader{noc_addr_a_1});
        }
    }

    noc_async_write_barrier();
}
