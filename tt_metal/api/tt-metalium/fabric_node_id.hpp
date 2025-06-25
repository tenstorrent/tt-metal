// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <cstdint>
#include <tt-metalium/fabric_types.hpp>

namespace tt::tt_fabric {

/**
 * @brief Represents a unique identifier for a node in the tt_fabric network.
 *
 * FabricNodeId combines a mesh identifier and a physical chip identifier to uniquely
 * identify a specific node within tt_fabric, regardless of the physical topology.
 */
class FabricNodeId {
public:
    /**
     * @brief Constructs a FabricNodeId with the specified mesh and chip identifiers.
     * @param mesh_id The identifier of the mesh this node belongs to
     * @param chip_id The physical identifier of the chip within the mesh
     */
    explicit FabricNodeId(MeshId mesh_id, std::uint32_t chip_id);

    MeshId mesh_id{0};          ///< The mesh this node belongs to
    std::uint32_t chip_id = 0;  ///< The physical chip identifier within the mesh
};

bool operator==(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator!=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>=(const FabricNodeId& lhs, const FabricNodeId& rhs);
std::ostream& operator<<(std::ostream& os, const FabricNodeId& fabric_node_id);

}  // namespace tt::tt_fabric

namespace std {
template <>
struct hash<tt::tt_fabric::FabricNodeId> {
    size_t operator()(const tt::tt_fabric::FabricNodeId& fabric_node_id) const noexcept {
        return tt::stl::hash::hash_objects_with_default_seed(fabric_node_id.mesh_id, fabric_node_id.chip_id);
    }
};
}  // namespace std
