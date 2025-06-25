// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "fabric_node_id.hpp"
#include <ostream>
namespace tt::tt_fabric {

FabricNodeId::FabricNodeId(MeshId mesh_id, std::uint32_t chip_id) {
    this->mesh_id = mesh_id;
    this->chip_id = chip_id;
}

bool operator==(const FabricNodeId& lhs, const FabricNodeId& rhs) {
    return lhs.mesh_id == rhs.mesh_id && lhs.chip_id == rhs.chip_id;
}
bool operator!=(const FabricNodeId& lhs, const FabricNodeId& rhs) { return !(lhs == rhs); }
bool operator<(const FabricNodeId& lhs, const FabricNodeId& rhs) {
    return lhs.mesh_id < rhs.mesh_id || (lhs.mesh_id == rhs.mesh_id && lhs.chip_id < rhs.chip_id);
}
bool operator>(const FabricNodeId& lhs, const FabricNodeId& rhs) { return rhs < lhs; }
bool operator<=(const FabricNodeId& lhs, const FabricNodeId& rhs) { return !(rhs > lhs); }
bool operator>=(const FabricNodeId& lhs, const FabricNodeId& rhs) { return !(lhs < rhs); }
std::ostream& operator<<(std::ostream& os, const FabricNodeId& fabric_node_id) {
    using ::operator<<;  // Enable ADL for StrongType operator<<
    os << "M" << fabric_node_id.mesh_id << "D" << fabric_node_id.chip_id;
    return os;
}
}  // namespace tt::tt_fabric
