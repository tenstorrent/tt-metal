// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"

namespace tt::tt_metal {

class TrayAsicToMeshMappingFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceExaboxFixture {};

// Key type for (TrayID, ASICLocation) pair
struct TrayAsicLocationKey {
    TrayID tray_id;
    ASICLocation asic_location;

    bool operator==(const TrayAsicLocationKey& other) const {
        return tray_id == other.tray_id && asic_location == other.asic_location;
    }
};

// Value type containing FabricNodeId and the rank that owns it
struct FabricNodeIdWithRank {
    tt_fabric::FabricNodeId fabric_node_id;
    uint32_t rank;
};

}  // namespace tt::tt_metal

// Hash specialization for the key
namespace std {
template <>
struct hash<tt::tt_metal::TrayAsicLocationKey> {
    std::size_t operator()(const tt::tt_metal::TrayAsicLocationKey& key) const noexcept {
        return std::hash<uint32_t>{}(*key.tray_id) ^ (std::hash<uint32_t>{}(*key.asic_location) << 1);
    }
};
}  // namespace std

namespace tt::tt_metal {

// Determine mapping from AsicID to FabricNodeId with rank information
std::unordered_map<AsicID, FabricNodeIdWithRank> generate_asic_id_to_fabric_node_id_map(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();

    std::unordered_map<tt::tt_metal::AsicID, FabricNodeIdWithRank> asic_id_to_fabric_node_id_map;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        tt_fabric::FabricNodeId fabric_node_id = mesh_device->get_fabric_node_id(coord);
        tt_metal::AsicID asic_id = control_plane.get_asic_id_from_fabric_node_id(fabric_node_id);
        asic_id_to_fabric_node_id_map.emplace(asic_id, FabricNodeIdWithRank{fabric_node_id, my_rank});
    }
    // Exchange this map across all hosts using distributed context
    // Follow MPI broadcast semantics for this (sender + receivers all call the broadcast API)
    for (auto rank = 0; rank < *(distributed_context->size()); rank++) {
        if (rank == *(distributed_context->rank())) {
            // Loop over all entries of the map and send them to the other hosts
            std::size_t num_entries = asic_id_to_fabric_node_id_map.size();
            distributed_context->broadcast(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto& [asic_id, fabric_node_id_with_rank] : asic_id_to_fabric_node_id_map) {
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(const_cast<tt_metal::AsicID*>(&asic_id)), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                uint32_t mesh_id_val = *fabric_node_id_with_rank.fabric_node_id.mesh_id;
                uint32_t chip_id_val = fabric_node_id_with_rank.fabric_node_id.chip_id;
                uint32_t rank_val = fabric_node_id_with_rank.rank;
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&mesh_id_val), sizeof(mesh_id_val)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&chip_id_val), sizeof(chip_id_val)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&rank_val), sizeof(rank_val)),
                    distributed::multihost::Rank{rank});
            }
        } else {
            // Receive the map from the other host
            std::size_t num_entries = 0;
            distributed_context->broadcast(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&num_entries), sizeof(num_entries)),
                distributed::multihost::Rank{rank});
            for (auto i = 0; i < num_entries; i++) {
                tt_metal::AsicID asic_id;
                uint32_t mesh_id_val = 0;
                uint32_t chip_id_val = 0;
                uint32_t rank_val = 0;
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&asic_id), sizeof(asic_id)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&mesh_id_val), sizeof(mesh_id_val)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&chip_id_val), sizeof(chip_id_val)),
                    distributed::multihost::Rank{rank});
                distributed_context->broadcast(
                    tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&rank_val), sizeof(rank_val)),
                    distributed::multihost::Rank{rank});
                asic_id_to_fabric_node_id_map.emplace(
                    asic_id,
                    FabricNodeIdWithRank{
                        tt_fabric::FabricNodeId(tt_fabric::MeshId{mesh_id_val}, chip_id_val), rank_val});
            }
        }
    }
    return asic_id_to_fabric_node_id_map;
}

PhysicalSystemDescriptor create_physical_system_descriptor() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    constexpr bool run_discovery = true;
    const auto& driver = cluster.get_driver();

    return tt::tt_metal::PhysicalSystemDescriptor(driver, distributed_context, &hal, rtoptions, run_discovery);
}

// Generate mapping from (TrayID, ASICLocation) → FabricNodeId with rank
std::unordered_map<TrayAsicLocationKey, FabricNodeIdWithRank> generate_tray_asic_to_fabric_node_id_map(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    // First get AsicID → FabricNodeId map (with multi-host exchange)
    auto asic_id_to_fabric_node_id = generate_asic_id_to_fabric_node_id_map(mesh_device);

    // Build (TrayID, ASICLocation) → FabricNodeId map
    std::unordered_map<TrayAsicLocationKey, FabricNodeIdWithRank> tray_asic_to_fabric_node_id;

    for (const auto& [asic_id, fabric_node_id_with_rank] : asic_id_to_fabric_node_id) {
        TrayID tray_id = physical_system_descriptor.get_tray_id(asic_id);
        ASICLocation asic_location = physical_system_descriptor.get_asic_location(asic_id);
        tray_asic_to_fabric_node_id.emplace(TrayAsicLocationKey{tray_id, asic_location}, fabric_node_id_with_rank);
    }

    return tray_asic_to_fabric_node_id;
}

TEST_F(TrayAsicToMeshMappingFixture, DetermineTrayAsicToMeshMapping) {
    const auto& distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto my_rank = *distributed_context->rank();

    auto physical_system_descriptor = create_physical_system_descriptor();
    auto tray_asic_to_fabric_node_id =
        generate_tray_asic_to_fabric_node_id_map(mesh_device_, physical_system_descriptor);

    // Only print from rank 0 to avoid interleaved output
    if (my_rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "TrayID + ASICLocation -> FabricNodeId Mapping" << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Collect and sort entries for readable output
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> sorted_entries;
        for (const auto& [key, fabric_node_id_with_rank] : tray_asic_to_fabric_node_id) {
            sorted_entries.emplace_back(
                *key.tray_id,
                *key.asic_location,
                *fabric_node_id_with_rank.fabric_node_id.mesh_id,
                fabric_node_id_with_rank.fabric_node_id.chip_id,
                fabric_node_id_with_rank.rank);
        }
        std::sort(sorted_entries.begin(), sorted_entries.end());

        std::cout << "| TrayID | ASICLocation | MeshId | ChipId | Rank |" << std::endl;
        std::cout << "|--------|--------------|--------|--------|------|" << std::endl;
        for (const auto& [tray_id, asic_loc, mesh_id, chip_id, rank] : sorted_entries) {
            std::cout << "|   " << std::setw(4) << tray_id << " |         " << std::setw(4) << asic_loc << " |   "
                      << std::setw(4) << mesh_id << " |   " << std::setw(4) << chip_id << " | " << std::setw(4) << rank
                      << " |" << std::endl;
        }
        std::cout << "\nTotal entries: " << tray_asic_to_fabric_node_id.size() << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    // Synchronize before finishing
    distributed_context->barrier();

    EXPECT_GT(tray_asic_to_fabric_node_id.size(), 0);
}

}  // namespace tt::tt_metal
