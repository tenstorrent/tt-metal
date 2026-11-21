// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tests the RoutingPlaneConnectionManager overload of fabric_unicast_noc_unicast_write.
//
// Fixed RT args (indices 0-5):
//   0: src_l1_addr    — source buffer on this chip
//   1: dst_noc_x      — NOC x-coord of target core on chip 1
//   2: dst_noc_y      — NOC y-coord of target core on chip 1
//   3: dst_l1_addr_0  — destination for route_id=0
//   4: dst_l1_addr_1  — destination for route_id=1
//   5: size           — payload size in bytes
//
// ConnectionManager RT args (indices 6+):
//   Appended by host via append_routing_plane_connection_manager_rt_args (2 slots).
//
// Writes src_l1_addr → dst_l1_addr_0 via route_id=0, then
//        src_l1_addr → dst_l1_addr_1 via route_id=1.

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

void kernel_main() {
    const uint32_t src_l1_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t dst_l1_addr_0 = get_arg_val<uint32_t>(3);
    const uint32_t dst_l1_addr_1 = get_arg_val<uint32_t>(4);
    const uint32_t size = get_arg_val<uint32_t>(5);

    std::size_t cm_arg_idx = 6;
    auto connection_manager = tt::tt_fabric::RoutingPlaneConnectionManager::build_from_args(cm_arg_idx, 2);

    using namespace tt::tt_fabric::mesh::experimental;

    const uint64_t noc_addr_0 =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_0);
    const uint64_t noc_addr_1 =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_1);

    fabric_unicast_noc_unicast_write<false>(
        connection_manager, 0, src_l1_addr, size, tt::tt_fabric::NocUnicastCommandHeader{noc_addr_0});

    fabric_unicast_noc_unicast_write<false>(
        connection_manager, 1, src_l1_addr, size, tt::tt_fabric::NocUnicastCommandHeader{noc_addr_1});

    noc_async_write_barrier();
}
