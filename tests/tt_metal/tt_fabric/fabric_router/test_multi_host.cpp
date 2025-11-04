// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <vector>
#include <tt_stl/span.hpp>
#include <cstring>

#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed_context.hpp>

// FIXME: Remove after testing
#include <set>
#include <unordered_set>

namespace tt::tt_fabric {
namespace multi_host_tests {

std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_intermesh_connections(const ControlPlane& control_plane) {
    std::vector<std::pair<FabricNodeId, FabricNodeId>> all_intermesh_connections;
    const auto& inter_conn = control_plane.get_mesh_graph().get_inter_mesh_connectivity();
    for (size_t mesh_id_val = 0; mesh_id_val < inter_conn.size(); ++mesh_id_val) {
        const auto& mesh = inter_conn[mesh_id_val];
        for (size_t chip_id = 0; chip_id < mesh.size(); ++chip_id) {
            const auto& connections = mesh[chip_id];
            for (const auto& [dst_mesh_id, edge] : connections) {
                for (auto dst_chip_id : edge.connected_chip_ids) {
                    all_intermesh_connections.push_back(
                        {FabricNodeId(
                             MeshId(static_cast<unsigned int>(mesh_id_val)), static_cast<std::uint32_t>(chip_id)),
                         FabricNodeId(MeshId(*dst_mesh_id), dst_chip_id)});
                }
            }
        }
    }
    return all_intermesh_connections;
}

std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_intramesh_connections(const ControlPlane& control_plane) {
    std::vector<std::pair<FabricNodeId, FabricNodeId>> all_intramesh_connections;
    const auto& intra_mesh_connectivity = control_plane.get_mesh_graph().get_intra_mesh_connectivity();
    for (size_t mesh_id_val = 0; mesh_id_val < intra_mesh_connectivity.size(); ++mesh_id_val) {
        const auto& mesh = intra_mesh_connectivity[mesh_id_val];
        for (size_t chip_id = 0; chip_id < mesh.size(); ++chip_id) {
            const auto& connections = mesh[chip_id];
            for (const auto& [dst_chip_id, edge] : connections) {
                for (auto dst_chip_id : edge.connected_chip_ids) {
                    all_intramesh_connections.push_back(
                        {FabricNodeId(
                             MeshId(static_cast<unsigned int>(mesh_id_val)), static_cast<std::uint32_t>(chip_id)),
                         FabricNodeId(MeshId(static_cast<unsigned int>(mesh_id_val)), dst_chip_id)});
                }
            }
        }
    }
    return all_intramesh_connections;
}

TEST(MultiHost, TestDualGalaxyControlPlaneInit) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
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
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_EQ(
        intramesh_connections.size(),
        896);  // 56 (connections for 8x8) * 2 (bidirectional) * 4 (connections per direction)
    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestDualGalaxyFabric1DSanity) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_EQ(
        intramesh_connections.size(),
        896);  // 56 (connections for 8x8) * 2 (bidirectional) * 4 (connections per direction)
    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestDual2x4ControlPlaneInit) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        log_info(tt::LogTest, "This test is only for T3K");
        GTEST_SKIP();
    }
    const std::filesystem::path dual_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(dual_galaxy_mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestDual2x4Fabric2DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        log_info(tt::LogTest, "This test is only for T3K");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intermesh_connections = get_all_intermesh_connections(control_plane);
    EXPECT_EQ(intermesh_connections.size(), 16);  // Bidirectional
    for (const auto& [src_node_id, dst_node_id] : intermesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestDual2x4Fabric1DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        log_info(tt::LogTest, "This test is only for T3K");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intermesh_connections = get_all_intermesh_connections(control_plane);
    EXPECT_EQ(intermesh_connections.size(), 16);  // Bidirectional
    for (const auto& [src_node_id, dst_node_id] : intermesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestSplit2x2ControlPlaneInit) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() !=
        tt::tt_metal::ClusterType::N300_2x2) {
        log_info(tt::LogTest, "This test is only for N300 2x2");
        GTEST_SKIP();
    }

    const std::filesystem::path split_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(split_2x2_mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestSplit2x2Fabric2DSanity) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intermesh_connections = get_all_intermesh_connections(control_plane);
    EXPECT_EQ(intermesh_connections.size(), 8);  // Bidirectional
    for (const auto& [src_node_id, dst_node_id] : intermesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestSplit2x2Fabric1DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() !=
        tt::tt_metal::ClusterType::N300_2x2) {
        log_info(tt::LogTest, "This test is only for N300 2x2");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intermesh_connections = get_all_intermesh_connections(control_plane);
    EXPECT_EQ(intermesh_connections.size(), 8);  // Bidirectional
    for (const auto& [src_node_id, dst_node_id] : intermesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestBigMesh2x4ControlPlaneInit) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() !=
        tt::tt_metal::ClusterType::N300_2x2) {
        log_info(tt::LogTest, "This test is only for N300 2x2");
        GTEST_SKIP();
    }

    const std::filesystem::path big_mesh_2x4_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(big_mesh_2x4_mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestBigMesh2x4Fabric2DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() !=
        tt::tt_metal::ClusterType::N300_2x2) {
        log_info(tt::LogTest, "This test is only for N300 2x2");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_EQ(
        intramesh_connections.size(),
        40);  // 10 (connections for 2x4) * 2 (bidirectional) * 2 (connections per direction)
    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestBigMesh2x4Fabric1DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() !=
        tt::tt_metal::ClusterType::N300_2x2) {
        log_info(tt::LogTest, "This test is only for N300 2x2");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_EQ(
        intramesh_connections.size(),
        40);  // 10 (connections for 2x4) * 2 (bidirectional) * 2 (connections per direction)
    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestQuadGalaxyControlPlaneInit) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    const std::filesystem::path quad_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/quad_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(quad_galaxy_mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestQuadGalaxyFabric2DSanity) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_type = get_fabric_type(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY);

    FabricNodeId src_node_id(MeshId{0}, 3);  // On host rank 0
    MeshCoordinate src_mesh_coord(0, 3);
    FabricNodeId dst_node_id(MeshId{0}, 12);  // On host rank 3
    MeshCoordinate dst_mesh_coord(0, 12);

    RoutingDirection expected_direction;
    RoutingDirection expected_reverse_direction;
    if (fabric_type == tt::tt_fabric::FabricType::TORUS_XY) {
        expected_direction = RoutingDirection::W;
        expected_reverse_direction = RoutingDirection::E;
    } else {
        expected_direction = RoutingDirection::E;
        expected_reverse_direction = RoutingDirection::W;
    }

    auto host_local_coord_range = control_plane.get_coord_range(MeshId{0}, MeshScope::LOCAL);
    if (host_local_coord_range.contains(src_mesh_coord)) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_EQ(direction, expected_direction);

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, expected_direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }

    if (host_local_coord_range.contains(dst_mesh_coord)) {
        const auto& reverse_direction = control_plane.get_forwarding_direction(dst_node_id, src_node_id);
        EXPECT_EQ(reverse_direction, expected_reverse_direction);

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id);
        EXPECT_TRUE(!eth_chans.empty());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id, expected_reverse_direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
    }
}

TEST(MultiHost, TestQuadGalaxyFabric1DSanity) {
    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Validate control plane apis
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    FabricNodeId src_node_id(MeshId{0}, 3);  // On host rank 0
    MeshCoordinate src_mesh_coord(0, 3);
    FabricNodeId dst_node_id(MeshId{0}, 12);  // On host rank 3
    MeshCoordinate dst_mesh_coord(0, 12);

    auto host_local_coord_range = control_plane.get_coord_range(MeshId{0}, MeshScope::LOCAL);
    if (host_local_coord_range.contains(src_mesh_coord)) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_EQ(direction, RoutingDirection::W);

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, RoutingDirection::W);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }

    if (host_local_coord_range.contains(dst_mesh_coord)) {
        const auto& reverse_direction = control_plane.get_forwarding_direction(dst_node_id, src_node_id);
        EXPECT_EQ(reverse_direction, RoutingDirection::E);

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id);
        EXPECT_TRUE(!eth_chans.empty());
        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(dst_node_id, src_node_id, RoutingDirection::E);
        EXPECT_TRUE(!eth_chans_by_direction.empty());
    }
}

TEST(MultiHost, TestBHQB4x4ControlPlaneInit) {
    // This test is intended for Blackhole 4x4 mesh spanning 2x2 hosts (BHQB)
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::P150_X4) {
        log_info(tt::LogTest, "This test is only for Blackhole QuietBox (BHQB)");
        GTEST_SKIP();
    }

    const std::filesystem::path bhqb_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/bh_qb_4x4_mesh_graph_descriptor.textproto";
    auto control_plane = std::make_unique<ControlPlane>(bhqb_mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestBHQB4x4Fabric2DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::P150_X4) {
        log_info(tt::LogTest, "This test is only for Blackhole QuietBox (BHQB)");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // 4x4 torus has 32 unique undirected adjacencies: (horizontal 16 + vertical 16)
    // With bidirectional and 2 ethernet channels per direction -> 32 * 2 * 2 = 128
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_EQ(intramesh_connections.size(), 128);

    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestBHQB4x4Fabric1DSanity) {
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::P150_X4) {
        log_info(tt::LogTest, "This test is only for Blackhole QuietBox (BHQB)");
        GTEST_SKIP();
    }

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Intra-mesh adjacency count is determined by the MGD, independent of fabric config
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_EQ(intramesh_connections.size(), 128);

    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestClosetBox3PodTTSwitchControlPlaneInit) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto";
    auto control_plane = std::make_unique<ControlPlane>(mesh_graph_desc_path.string());

    control_plane->configure_routing_tables_for_fabric_ethernet_channels(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestClosetBox3PodTTSwitchFabric2DSanity) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    const auto& switch_ids = mesh_graph.get_switch_ids();
    EXPECT_EQ(switch_ids.size(), 1) << "Should have exactly 1 switch";
    SwitchId switch_id = switch_ids[0];
    MeshId switch_mesh_id = mesh_graph.get_mesh_id_for_switch(switch_id);

    // Test intermesh connections - switch should have connections to all 3 meshes
    const auto& intermesh_connections = get_all_intermesh_connections(control_plane);
    EXPECT_GT(intermesh_connections.size(), 0) << "Should have intermesh connections";

    // Count connections involving the switch
    size_t switch_connections = 0;
    for (const auto& [src_node_id, dst_node_id] : intermesh_connections) {
        if (src_node_id.mesh_id == switch_mesh_id || dst_node_id.mesh_id == switch_mesh_id) {
            switch_connections++;
            // Verify connectivity through control plane APIs
            const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
            EXPECT_TRUE(direction.has_value()) << "Should have forwarding direction for switch connections";

            const auto& eth_chans_by_direction =
                control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
            EXPECT_TRUE(!eth_chans_by_direction.empty()) << "Should have ethernet channels for switch connections";

            const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
            EXPECT_TRUE(!eth_chans.empty()) << "Should have ethernet channels for switch connections";
        }
    }
    EXPECT_GT(switch_connections, 0) << "Switch should have intermesh connections";

    // Test intramesh connections for switch (switch internal connectivity)
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    size_t switch_intramesh_connections = 0;
    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        if (src_node_id.mesh_id == switch_mesh_id) {
            switch_intramesh_connections++;
            const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
            EXPECT_TRUE(direction.has_value());
            const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
            EXPECT_TRUE(!eth_chans.empty());
        }
    }
    EXPECT_GT(switch_intramesh_connections, 0) << "Switch should have intramesh connections";
}

TEST(MultiHost, TestClosetBox3PodTTSwitchFabric1DSanity) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Test switch APIs are still accessible with 1D fabric config
    const auto& switch_ids = mesh_graph.get_switch_ids();
    EXPECT_EQ(switch_ids.size(), 1) << "Should have exactly 1 switch";

    SwitchId switch_id = switch_ids[0];
    MeshId switch_mesh_id = mesh_graph.get_mesh_id_for_switch(switch_id);

    // Test intermesh connections with 1D fabric config
    const auto& intermesh_connections = get_all_intermesh_connections(control_plane);
    EXPECT_GT(intermesh_connections.size(), 0) << "Should have intermesh connections";

    for (const auto& [src_node_id, dst_node_id] : intermesh_connections) {
        if (src_node_id.mesh_id == switch_mesh_id || dst_node_id.mesh_id == switch_mesh_id) {
            const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
            EXPECT_TRUE(direction.has_value());

            const auto& eth_chans_by_direction =
                control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
            EXPECT_TRUE(!eth_chans_by_direction.empty());

            const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
            EXPECT_TRUE(!eth_chans.empty());
        }
    }

    // Intra-mesh adjacency count is determined by the MGD, independent of fabric config
    const auto& intramesh_connections = get_all_intramesh_connections(control_plane);
    EXPECT_GT(intramesh_connections.size(), 0);

    for (const auto& [src_node_id, dst_node_id] : intramesh_connections) {
        const auto& direction = control_plane.get_forwarding_direction(src_node_id, dst_node_id);
        EXPECT_TRUE(direction.has_value());

        const auto& eth_chans_by_direction =
            control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id, *direction);
        EXPECT_TRUE(!eth_chans_by_direction.empty());

        const auto& eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_node_id, dst_node_id);
        EXPECT_TRUE(!eth_chans.empty());
    }
}

TEST(MultiHost, TestClosetBox3PodTTSwitchAPIs) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_3pod_ttswitch_mgd.textproto";

    auto control_plane = std::make_unique<ControlPlane>(mesh_graph_desc_path.string());
    const auto& mesh_graph = control_plane->get_mesh_graph();

    // ========== MeshGraph API Tests ==========
    // Test get_switch_ids()
    const auto& switch_ids = mesh_graph.get_switch_ids();
    ASSERT_EQ(switch_ids.size(), 1) << "Should have exactly 1 switch";
    EXPECT_EQ(*switch_ids[0], 3) << "Switch ID should be 0";

    SwitchId switch_id = switch_ids[0];

    // Test get_mesh_id_for_switch()
    MeshId switch_mesh_id = mesh_graph.get_mesh_id_for_switch(switch_id);
    EXPECT_EQ(*switch_mesh_id, 3) << "Switch should have a mapped mesh_id";

    // Verify switch mesh_id is unique (not used by regular meshes)
    const auto& all_mesh_ids = mesh_graph.get_mesh_ids();
    size_t switch_mesh_id_count = 0;
    for (const auto& mesh_id : all_mesh_ids) {
        if (mesh_id == switch_mesh_id) {
            switch_mesh_id_count++;
        }
    }
    EXPECT_EQ(switch_mesh_id_count, 1) << "Switch mesh_id should be unique";

    // Test get_meshes_connected_to_switch()
    const auto& connected_meshes = mesh_graph.get_meshes_connected_to_switch(switch_id);
    EXPECT_EQ(connected_meshes.size(), 3) << "Switch should be connected to 3 meshes";

    // Verify connected meshes are the expected ones (mesh_ids 0, 1, 2)
    std::set<uint32_t> connected_mesh_id_values;
    for (const auto& mesh_id : connected_meshes) {
        connected_mesh_id_values.insert(*mesh_id);
    }
    EXPECT_EQ(connected_mesh_id_values.size(), 3) << "Should have 3 unique connected meshes";
    EXPECT_TRUE(connected_mesh_id_values.find(0) != connected_mesh_id_values.end());
    EXPECT_TRUE(connected_mesh_id_values.find(1) != connected_mesh_id_values.end());
    EXPECT_TRUE(connected_mesh_id_values.find(2) != connected_mesh_id_values.end());

    // Test is_mesh_connected_to_switch() for each connected mesh
    for (const auto& mesh_id : connected_meshes) {
        EXPECT_TRUE(mesh_graph.is_mesh_connected_to_switch(mesh_id, switch_id))
            << "Mesh " << *mesh_id << " should be connected to switch";
    }

    // Test is_mesh_connected_to_switch() for non-connected mesh (if any exists)
    // In this topology, all meshes are connected to the switch, so we test with a non-existent mesh_id
    MeshId non_existent_mesh_id(999);
    EXPECT_FALSE(mesh_graph.is_mesh_connected_to_switch(non_existent_mesh_id, switch_id))
        << "Non-existent mesh should not be connected to switch";

    // Test get_switch_for_mesh() for each connected mesh
    for (const auto& mesh_id : connected_meshes) {
        auto switch_for_mesh = mesh_graph.get_switch_for_mesh(mesh_id);
        ASSERT_TRUE(switch_for_mesh.has_value()) << "Mesh " << *mesh_id << " should have a connected switch";
        EXPECT_EQ(*switch_for_mesh.value(), *switch_id) << "Mesh " << *mesh_id << " should be connected to switch 0";
    }

    // Test get_switch_for_mesh() for non-connected mesh
    auto switch_for_non_existent = mesh_graph.get_switch_for_mesh(non_existent_mesh_id);
    EXPECT_FALSE(switch_for_non_existent.has_value()) << "Non-existent mesh should not have a switch";

    // Test that switch mesh_id works with other MeshGraph APIs
    const auto& host_ranks = mesh_graph.get_host_ranks(switch_mesh_id);
    EXPECT_EQ(host_ranks.size(), 1) << "Switch should have exactly 1 host rank (single host constraint)";

    const auto& chip_ids = mesh_graph.get_chip_ids(switch_mesh_id);
    EXPECT_EQ(chip_ids.size(), 8) << "Switch should have 2*4=8 chips";

    const auto& mesh_shape = mesh_graph.get_mesh_shape(switch_mesh_id);
    EXPECT_EQ(mesh_shape, MeshShape(2, 4)) << "Switch should have 2x4 shape";

    // Test coord range for switch
    const auto& coord_range = mesh_graph.get_coord_range(switch_mesh_id);
    EXPECT_EQ(coord_range, MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)))
        << "Switch should have coord range (0,0) to (1,3)";

    // Test host rank for chips in switch
    for (uint32_t chip_id = 0; chip_id < 8; ++chip_id) {
        auto host_rank = mesh_graph.get_host_rank_for_chip(switch_mesh_id, chip_id);
        EXPECT_EQ(*host_rank, MeshHostRankId{0}) << "All switch chips should be on host rank 0";
    }
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
