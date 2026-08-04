// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/internal/cluster.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal::internal {

AsicID get_chip_unique_id_from_fabric_node_id(const tt::tt_fabric::FabricNodeId& fabric_node_id) {
    // The control plane owns the canonical FabricNodeId -> physical ASIC mapping (the same one
    // fabric sockets route by). Resolve through it so the id matches what the UMD/worker keys on.
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
}

AsicID get_chip_unique_id_from_fabric_node_id(uint32_t mesh_id, uint32_t chip_id) {
    return get_chip_unique_id_from_fabric_node_id(tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{mesh_id}, chip_id));
}

}  // namespace tt::tt_metal::internal
