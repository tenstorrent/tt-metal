// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <vector>

#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_fabric {
namespace multi_host_tests {

TEST(MultiHost, TestDualGalaxyControlPlaneInit) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::GALAXY) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    const std::filesystem::path dual_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(dual_galaxy_mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestDualGalaxyFabric2DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::GALAXY) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    FabricNodeId src_node_id(MeshId{0}, 3);  // On host rank 0
    MeshCoordinate src_mesh_coord(0, 3);
    FabricNodeId dst_node_id(MeshId{0}, 4);  // On host rank 1
    MeshCoordinate dst_mesh_coord(0, 4);

    auto host_local_coord_range = control_plane.get_coord_range(MeshId{0}, MeshScope::LOCAL);
    if (host_local_coord_range.contains(src_mesh_coord)) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_EQ(direction, RoutingDirection::E);

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, RoutingDirection::E);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }

    if (host_local_coord_range.contains(dst_mesh_coord)) {
        const auto& reverse_direction = control_plane.get_forwarding_direction(dst_node_id, src_node_id);
        EXPECT_EQ(reverse_direction, RoutingDirection::W);

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id);
        EXPECT_TRUE(!eth_chans.empty());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id, RoutingDirection::W);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
    }
}

TEST(MultiHost, TestDualGalaxyFabric1DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::GALAXY) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    FabricNodeId src_node_id(MeshId{0}, 3);  // On host rank 0
    MeshCoordinate src_mesh_coord(0, 3);
    FabricNodeId dst_node_id(MeshId{0}, 4);  // On host rank 1
    MeshCoordinate dst_mesh_coord(0, 4);

    auto host_local_coord_range = control_plane.get_coord_range(MeshId{0}, MeshScope::LOCAL);
    if (host_local_coord_range.contains(src_mesh_coord)) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_EQ(direction, RoutingDirection::E);

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, RoutingDirection::E);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }

    if (host_local_coord_range.contains(dst_mesh_coord)) {
        const auto& reverse_direction = control_plane.get_forwarding_direction(dst_node_id, src_node_id);
        EXPECT_EQ(reverse_direction, RoutingDirection::W);

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id);
        EXPECT_TRUE(!eth_chans.empty());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id, RoutingDirection::W);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
    }
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
