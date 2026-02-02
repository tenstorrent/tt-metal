// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include <enchantum/enchantum.hpp>
#include <ostream>
#include <tt_stl/reflection.hpp>

namespace tt::tt_fabric {

FabricManagerMode operator|(FabricManagerMode lhs, FabricManagerMode rhs) {
    return static_cast<FabricManagerMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

FabricManagerMode operator&(FabricManagerMode lhs, FabricManagerMode rhs) {
    return static_cast<FabricManagerMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

bool has_flag(FabricManagerMode flags, FabricManagerMode test) { return (flags & test) == test; }

FabricType operator|(FabricType lhs, FabricType rhs) {
    return static_cast<FabricType>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

FabricType operator&(FabricType lhs, FabricType rhs) {
    return static_cast<FabricType>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

bool has_flag(FabricType flags, FabricType test) { return (flags & test) == test; }

FabricNodeId::FabricNodeId(MeshId mesh_id_val, std::uint32_t chip_id_val) :
    mesh_id(mesh_id_val), chip_id(chip_id_val) {}

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

namespace std {
size_t hash<tt::tt_fabric::FabricNodeId>::operator()(const tt::tt_fabric::FabricNodeId& fabric_node_id) const noexcept {
    return tt::stl::hash::hash_objects_with_default_seed(fabric_node_id.mesh_id, fabric_node_id.chip_id);
}
}  // namespace std

auto fmt::formatter<tt::tt_fabric::FabricNodeId>::format(
    const tt::tt_fabric::FabricNodeId& node_id, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(ctx.out(), "(M{}, D{})", *node_id.mesh_id, node_id.chip_id);
}
