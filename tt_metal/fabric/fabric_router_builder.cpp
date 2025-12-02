// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_builder.hpp"
#include "tt_metal/fabric/compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric {

std::unique_ptr<FabricRouterBuilder> FabricRouterBuilder::create(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricNodeId local_node,
    const RouterLocation& location,
    const RouterBuildSpec& spec) {
    // Get SOC descriptor for eth core lookup
    const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    auto eth_logical_core = soc_desc.get_eth_core_for_channel(location.eth_chan, CoordSystem::LOGICAL);

    // Convert RoutingDirection to eth_chan_directions
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto eth_direction = control_plane.routing_direction_to_eth_direction(location.direction);

    if (spec.is_switch_mesh) {
        // TODO: Phase 6 - Create SwitchMeshRouterBuilder
        // For now, fall through to compute mesh (shouldn't reach here until switch mesh is enabled)
        TT_FATAL(false, "Switch mesh router builder not yet implemented");
    }

    // Create compute mesh router using ComputeMeshRouterBuilder::build()
    return ComputeMeshRouterBuilder::build(
        device,
        program,
        eth_logical_core,
        local_node,
        location.remote_node,
        *spec.edm_config,
        eth_direction,
        location.is_dispatch_link,
        location.eth_chan,
        spec.topology);
}

}  // namespace tt::tt_fabric
