// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Experimental API; subject to change without notice.

#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

namespace tt::tt_metal::experimental::fabric {

// Worker-side view of the opaque arguments produced by get_equal_cost_unicast_routes.
class UnicastRouteView {
public:
    constexpr UnicastRouteView() = default;
    explicit constexpr UnicastRouteView(size_t arg_index) : arg_index_(arg_index), canonical_(false) {}
    uint32_t num_hops() const { return canonical_ ? 0 : get_arg_val<uint32_t>(arg_index_ + 1); }
    tt::tt_fabric::eth_chan_directions initial_direction() const {
        return static_cast<tt::tt_fabric::eth_chan_directions>(get_arg_val<uint32_t>(arg_index_));
    }

private:
    friend bool fabric_set_equal_cost_unicast_route(
        volatile tt_l1_ptr tt::tt_fabric::HybridMeshPacketHeader*, uint16_t, uint16_t, const UnicastRouteView&);

    uint32_t command(uint32_t hop) const {
        return (get_arg_val<uint32_t>(arg_index_ + 2 + hop / 8) >> (4 * (hop % 8))) & 0xf;
    }

    size_t arg_index_ = 0;
    bool canonical_ = true;
};

FORCE_INLINE UnicastRouteView get_unicast_route_from_args(size_t arg_index) { return UnicastRouteView(arg_index); }

FORCE_INLINE tt::tt_fabric::eth_chan_directions get_unicast_route_direction(
    const UnicastRouteView& route, uint16_t dst_mesh_id, uint16_t dst_dev_id) {
    return route.num_hops() == 0 ? tt::tt_fabric::get_next_hop_router_direction(dst_mesh_id, dst_dev_id)
                                 : route.initial_direction();
}

// Only Fabric writes packet routing fields. The host validates every physical hop and
// capacity; recheck the device header specialization before using a noncanonical route.
FORCE_INLINE bool fabric_set_equal_cost_unicast_route(
    volatile tt_l1_ptr tt::tt_fabric::HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    const UnicastRouteView& route) {
    const uint32_t hops = route.num_hops();
    if (hops == 0) {
        return tt::tt_fabric::fabric_set_unicast_route(packet_header, dst_dev_id, dst_mesh_id);
    }
    if (route.initial_direction() >= tt::tt_fabric::eth_chan_directions::Z ||
        hops > tt::tt_fabric::FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE) {
        return false;
    }
    packet_header->dst_start_node_id = (static_cast<uint32_t>(dst_mesh_id) << 16) | dst_dev_id;
    packet_header->mcast_params_64 = 0;
    packet_header->is_mcast_active = 0;
    packet_header->routing_fields.value = 0;
    for (uint32_t hop = 0; hop < tt::tt_fabric::FabricHeaderConfig::MESH_ROUTE_BUFFER_SIZE; ++hop) {
        packet_header->route_buffer[hop] = hop < hops ? route.command(hop) : 0;
    }
    return true;
}

}  // namespace tt::tt_metal::experimental::fabric
