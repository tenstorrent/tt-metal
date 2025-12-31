// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/core_coord.hpp>

#include <optional>
#include <unordered_map>
#include <vector>
namespace tt::tt_metal::experimental {

enum class FabricConnectionMode : uint8_t {
    Direct,
    RoutingPlane,
    MUX,
};

// Describes a single fabric connection from one mesh coordinate to another.
// Used for explicit point-to-point connections when user knows exact src->dst pairs.
struct FabricConnectionDescriptor {
    distributed::MeshCoordinate src_coord;
    distributed::MeshCoordinate dst_coord;
    std::vector<distributed::MeshCoordinate> dst_coords;

    size_t kernel_index = 0;
    CoreCoord worker_core;

    // Link index for multi-link scenarios (0 for single link)
    uint32_t link_idx = 0;
    CoreType core_type = CoreType::WORKER;

    // In case we want different fabric config for different devices
    FabricConnectionMode mode = FabricConnectionMode::Direct;

    // Only needed for RoutingPlane mode
    tt::tt_fabric::FabricApiType api_type = tt::tt_fabric::FabricApiType::Linear;
};

// Describes a fabric topology to auto-generate connections.
// Use this when you want standard patterns (ring, linear, mesh) without
// specifying each connection explicitly.
struct FabricTopologyDescriptor {
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    std::optional<uint32_t> cluster_axis;
    size_t kernel_index = 0;
    CoreCoord worker_core;
    CoreType core_type = CoreType::WORKER;
};

struct MeshProgramDescriptor {
    std::unordered_map<distributed::MeshCoordinateRange, ProgramDescriptor> mesh_programs;

    // Option A: Auto-generate fabric connections from topology
    // Option B: Explicit fabric connections specified by the user
    std::optional<FabricTopologyDescriptor> fabric_topology = std::nullopt;
    std::vector<FabricConnectionDescriptor> fabric_connections;
    FabricConnectionMode mode = FabricConnectionMode::Direct;

    static constexpr auto attribute_names =
        std::forward_as_tuple("num_mesh_programs", "has_fabric_topology", "num_fabric_connections");
    auto attribute_values() const {
        return std::forward_as_tuple(mesh_programs.size(), fabric_topology.has_value(), fabric_connections.size());
    }
};

}  // namespace tt::tt_metal::experimental
