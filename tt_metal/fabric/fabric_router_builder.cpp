// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_router_builder.hpp"
#include "tt_metal/fabric/compute_mesh_router_builder.hpp"
#include "tt_metal/fabric/switch_mesh_router_builder.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

std::unique_ptr<FabricRouterBuilder> FabricRouterBuilder::create(
    const FabricContext& fabric_context,
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricNodeId local_node,
    const RouterLocation& location) {
    // Query fabric context to determine router type
    bool is_switch_mesh = fabric_context.is_switch_mesh(local_node.mesh_id);

    if (is_switch_mesh) {
        return SwitchMeshRouterBuilder::build(fabric_context, device, program, local_node, location);
    }

    // Create compute mesh router - it handles its own config lookup
    return ComputeMeshRouterBuilder::build(fabric_context, device, program, local_node, location);
}

}  // namespace tt::tt_fabric
