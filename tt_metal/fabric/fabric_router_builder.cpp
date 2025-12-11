// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_builder.hpp"
#include "tt_metal/fabric/compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/switch_mesh_router_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>

namespace tt::tt_fabric {

std::unique_ptr<FabricRouterBuilder> FabricRouterBuilder::create(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricNodeId local_node,
    const RouterLocation& location) {
    // Query fabric context to determine router type
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    bool is_switch_mesh = fabric_context.is_switch_mesh(local_node.mesh_id);

    if (is_switch_mesh) {
        return SwitchMeshRouterBuilder::build(device, program, local_node, location);
    }

    // Create compute mesh router - it handles its own config lookup
    return ComputeMeshRouterBuilder::build(device, program, local_node, location);
}

}  // namespace tt::tt_fabric
