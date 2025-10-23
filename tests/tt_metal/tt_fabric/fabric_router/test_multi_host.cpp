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

#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed_context.hpp>

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

TEST(MultiHost, TestClosetBoxTTSwitchControlPlaneInit) {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    auto host_name = std::string(hostname);

    log_info(tt::LogTest, "Host name: {}", host_name);

    auto& instance = tt::tt_metal::MetalContext::instance();

    // Save the cluster descriptor in the WH closet box directory
    const std::filesystem::path output_dir =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_cluster_desc";
    std::filesystem::create_directories(output_dir);
    const auto filename = "closet_box_cluster_desc_" + host_name + ".yaml";
    const auto output_path = (output_dir / filename).string();

    auto cluster_descriptor = instance.get_cluster().get_cluster_desc()->serialize_to_file(output_path);

    log_info(tt::LogTest, "Cluster descriptor saved to: {}", cluster_descriptor);
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
