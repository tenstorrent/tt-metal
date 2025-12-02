// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {

// Forward declaration
struct FabricEriscDatamoverConfig;

/**
 * RouterLocation
 *
 * Minimal per-router topological info - "where is this router?"
 * Common state (device, program, local_node) is cached at FabricBuilder level.
 */
struct RouterLocation {
    chan_id_t eth_chan;
    FabricNodeId remote_node;
    RoutingDirection direction;  // Use RoutingDirection, convert to eth_chan_directions when needed
    bool is_dispatch_link;

    /**
     * Factory method to create a RouterLocation
     *
     * @param eth_chan The ethernet channel ID
     * @param remote_node The remote fabric node ID
     * @param direction The routing direction (N/S/E/W)
     * @param is_dispatch_link Whether this is a dispatch link
     * @return RouterLocation instance
     */
    static RouterLocation create(
        chan_id_t eth_chan, FabricNodeId remote_node, RoutingDirection direction, bool is_dispatch_link) {
        return RouterLocation{
            .eth_chan = eth_chan,
            .remote_node = remote_node,
            .direction = direction,
            .is_dispatch_link = is_dispatch_link,
        };
    }
};

/**
 * RouterBuildSpec
 *
 * Computed configuration for a router - "how should it behave?"
 * Output of FabricBuilderContext::get_router_build_spec()
 */
struct RouterBuildSpec {
    const FabricEriscDatamoverConfig* edm_config;  // Pointer to config template
    Topology topology;                             // Linear/Ring/Mesh/Torus
    bool tensix_extension_enabled;                 // For compute mesh MUX mode
    bool is_switch_mesh;                           // Determines which builder type to create
};

}  // namespace tt::tt_fabric
