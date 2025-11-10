// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "control_plane.hpp"
#include "fabric_host_utils.hpp"

#include <tt-metalium/fabric.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <tt-metalium/metal_soc_descriptor.h>
#include "impl/context/metal_context.hpp"
#include "erisc_datamover_builder.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include "fabric/hw/inc/fabric_routing_mode.h"
#include "fabric_context.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace tt::tt_fabric {

bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return is_1d_fabric_config(fabric_config) || is_2d_fabric_config(fabric_config);
}

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config) {
    switch (fabric_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: return FabricType::TORUS_XY;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X: return FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y: return FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return FabricType::TORUS_XY;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_X: return FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_Y: return FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY: return FabricType::TORUS_XY;
        default: return FabricType::MESH;
    }
}

bool requires_more_connectivity(FabricType requested_type, FabricType available_type, const MeshShape& mesh_shape) {
    // Requesting MESH is always valid (can restrict any topology to MESH)
    if (requested_type == FabricType::MESH) {
        return false;
    }

    // Check if available topology can satisfy the requested topology
    if (available_type == FabricType::MESH) {
        // Special case: 2-element dimensions make torus wrap-around equivalent to mesh neighbor connections
        // E.g., in a 2-row mesh, north/south wrap-around just connects to the adjacent row
        bool has_two_rows = (mesh_shape[0] == 2);
        bool has_two_cols = (mesh_shape[1] == 2);

        if (has_flag(requested_type, FabricType::TORUS_Y) && !has_two_rows) {
            return true;
        }
        if (has_flag(requested_type, FabricType::TORUS_X) && !has_two_cols) {
            return true;
        }
        return false;
    }

    // For non-MESH available types, check if requested features are present
    if (requested_type == FabricType::TORUS_XY) {
        return available_type != FabricType::TORUS_XY;
    }
    if (requested_type == FabricType::TORUS_X) {
        return !has_flag(available_type, FabricType::TORUS_X);
    }
    if (requested_type == FabricType::TORUS_Y) {
        return !has_flag(available_type, FabricType::TORUS_Y);
    }

    return false;
}

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id, RoutingDirection direction) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    const std::vector<chan_id_t>& fabric_channels =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);

    // the subset of routers that support forwarding b/w those chips
    std::vector<chan_id_t> forwarding_channels;
    forwarding_channels =
        control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, direction);

    std::vector<uint32_t> link_indices;
    for (uint32_t i = 0; i < fabric_channels.size(); i++) {
        if (std::find(forwarding_channels.begin(), forwarding_channels.end(), fabric_channels[i]) !=
            forwarding_channels.end()) {
            link_indices.push_back(i);
        }
    }

    return link_indices;
}

void set_routing_mode(uint16_t routing_mode) {
    // override for forced routing mode
    if (routing_mode == ROUTING_MODE_UNDEFINED) {
        return;
    }

    // Validate dimension flags are orthogonal (only one can be set)
    TT_FATAL(
        __builtin_popcount(routing_mode & (ROUTING_MODE_1D | ROUTING_MODE_2D | ROUTING_MODE_3D)) == 1,
        "Only one dimension mode (1D, 2D, 3D) can be active at once");

    // Validate topology flags are orthogonal
    TT_FATAL(
        __builtin_popcount(
            routing_mode & (ROUTING_MODE_RING | ROUTING_MODE_LINE | ROUTING_MODE_MESH | ROUTING_MODE_TORUS)) == 1,
        "Only one topology mode (RING, LINE, MESH, TORUS) can be active at once");

    // Validate 1D can't be used with MESH or TORUS
    TT_FATAL(
        !(routing_mode & ROUTING_MODE_1D) || !(routing_mode & (ROUTING_MODE_MESH | ROUTING_MODE_TORUS)),
        "1D routing mode cannot be combined with MESH or TORUS topology");

    // Validate 2D can't be used with LINE or RING
    TT_FATAL(
        !(routing_mode & ROUTING_MODE_2D) || !(routing_mode & (ROUTING_MODE_LINE | ROUTING_MODE_RING)),
        "2D routing mode cannot be combined with LINE or RING topology");

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    control_plane.set_routing_mode(routing_mode);
}

void set_routing_mode(Topology topology, tt::tt_fabric::FabricConfig fabric_config, uint32_t dimension /*, take more*/) {
    // TODO: take more parameters to set detail routing mode
    TT_FATAL(
        dimension == 1 || dimension == 2 || dimension == 3,
        "Invalid dimension {}. Supported dimensions are 1, 2, or 3",
        dimension);

    uint16_t mode = (dimension == 3 ? ROUTING_MODE_3D : 0);
    if (topology == Topology::Ring) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_RING);
    } else if (topology == Topology::Linear) {
        mode |= (ROUTING_MODE_1D | ROUTING_MODE_LINE);
    } else if (topology == Topology::Mesh) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_MESH);
    } else if (topology == Topology::Torus) {
        mode |= (ROUTING_MODE_2D | ROUTING_MODE_TORUS);
    }

    if (tt::tt_fabric::FabricContext::is_dynamic_routing_config(fabric_config)) {
        mode |= ROUTING_MODE_DYNAMIC;
    } else {
        mode |= ROUTING_MODE_LOW_LATENCY;
    }
    set_routing_mode(mode);
}

}  // namespace tt::tt_fabric
