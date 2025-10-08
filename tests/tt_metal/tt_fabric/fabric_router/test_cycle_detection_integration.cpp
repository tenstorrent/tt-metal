// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <vector>

#include <tt-metalium/fabric_types.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_fabric {
namespace multi_host_tests {

namespace {
constexpr auto kFabricConfig = FabricConfig::FABRIC_2D_DYNAMIC;
constexpr auto kReliabilityMode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;
}  // namespace

// Helper function to get all intermesh traffic pairs from connectivity
std::vector<std::pair<FabricNodeId, FabricNodeId>> get_all_intermesh_traffic_for_cycle_detection(
    const ControlPlane& control_plane) {
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs;
    const auto& inter_conn = control_plane.get_mesh_graph().get_inter_mesh_connectivity();

    for (size_t mesh_id_val = 0; mesh_id_val < inter_conn.size(); ++mesh_id_val) {
        const auto& mesh = inter_conn[mesh_id_val];
        for (size_t chip_id = 0; chip_id < mesh.size(); ++chip_id) {
            const auto& connections = mesh[chip_id];
            for (const auto& [dst_mesh_id, edge] : connections) {
                for (auto dst_chip_id : edge.connected_chip_ids) {
                    traffic_pairs.push_back(
                        {FabricNodeId(
                             MeshId(static_cast<unsigned int>(mesh_id_val)), static_cast<std::uint32_t>(chip_id)),
                         FabricNodeId(dst_mesh_id, dst_chip_id)});
                }
            }
        }
    }
    return traffic_pairs;
}

TEST(MultiHost, TestDualGalaxyCycleDetectionNoCycles) {
    // This test verifies that bidirectional intermesh connections in dual Galaxy setup
    // are correctly identified as safe traffic (not cycles)
    // Run with: tt-run --mock-cluster-rank-binding
    // tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml
    //           --rank-binding tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml

    if (!tt::tt_metal::MetalContext::instance().get_cluster().is_ubb_galaxy()) {
        log_info(tt::LogTest, "This test is only for GALAXY");
        GTEST_SKIP();
    }

    // Configure ethernet cores for fabric routing
    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        kFabricConfig, num_routing_planes);

    const std::filesystem::path dual_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<ControlPlane>(dual_galaxy_mesh_graph_desc_path.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    // Get all intermesh connections - these include bidirectional links between the two Galaxy meshes
    auto intermesh_pairs = get_all_intermesh_traffic_for_cycle_detection(*control_plane);

    log_info(tt::LogTest, "Testing {} intermesh traffic pairs for cycles in dual Galaxy", intermesh_pairs.size());

    // Test cycle detection - bidirectional intermesh traffic should NOT be detected as cycles
    bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "DualGalaxyCycleDetectionTest");

    EXPECT_FALSE(has_cycles)
        << "Bidirectional intermesh traffic between dual Galaxy meshes should NOT be detected as cycles";
}

TEST(MultiHost, TestDual2x4CycleDetectionNoCycles) {
    // This test verifies that bidirectional intermesh connections in dual 2x4 T3K setup
    // are correctly identified as safe traffic (not cycles)
    // Run with: tt-run --mock-cluster-rank-binding
    // tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml
    //           --rank-binding tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml

    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        log_info(tt::LogTest, "This test is only for T3K");
        GTEST_SKIP();
    }

    // Configure ethernet cores for fabric routing
    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        kFabricConfig, num_routing_planes);

    const std::filesystem::path dual_t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<ControlPlane>(dual_t3k_mesh_graph_desc_path.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    auto intermesh_pairs = get_all_intermesh_traffic_for_cycle_detection(*control_plane);

    log_info(tt::LogTest, "Testing {} intermesh traffic pairs for cycles in dual T3K 2x4", intermesh_pairs.size());

    // Test cycle detection
    bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "DualT3K2x4CycleDetectionTest");

    EXPECT_FALSE(has_cycles)
        << "Bidirectional intermesh traffic between dual T3K 2x4 meshes should NOT be detected as cycles";
}

TEST(MultiHost, TestSplit2x2CycleDetectionNoCycles) {
    // This test verifies bidirectional intermesh connections in split 2x2 T3K setup
    // Run with: tt-run --mock-cluster-rank-binding
    // tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml
    //           --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml

    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        log_info(tt::LogTest, "This test is only for T3K");
        GTEST_SKIP();
    }

    // Configure ethernet cores for fabric routing
    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        kFabricConfig, num_routing_planes);

    const std::filesystem::path split_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<ControlPlane>(split_2x2_mesh_graph_desc_path.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    auto intermesh_pairs = get_all_intermesh_traffic_for_cycle_detection(*control_plane);

    log_info(tt::LogTest, "Testing {} intermesh traffic pairs for cycles in split 2x2", intermesh_pairs.size());

    bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "Split2x2CycleDetectionTest");

    EXPECT_FALSE(has_cycles) << "Bidirectional intermesh traffic in split 2x2 setup should NOT be detected as cycles";
}

TEST(MultiHost, TestBigMesh2x4CycleDetectionNoCycles) {
    // This test verifies bidirectional intermesh connections in big mesh 2x4 T3K setup
    // Run with: tt-run --mock-cluster-rank-binding
    // tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml
    //           --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml

    if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::tt_metal::ClusterType::T3K) {
        log_info(tt::LogTest, "This test is only for T3K");
        GTEST_SKIP();
    }

    // Configure ethernet cores for fabric routing
    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        kFabricConfig, num_routing_planes);

    const std::filesystem::path big_mesh_2x4_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";

    auto control_plane = std::make_unique<ControlPlane>(big_mesh_2x4_mesh_graph_desc_path.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    auto intermesh_pairs = get_all_intermesh_traffic_for_cycle_detection(*control_plane);

    log_info(tt::LogTest, "Testing {} intermesh traffic pairs for cycles in big mesh 2x4", intermesh_pairs.size());

    bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "BigMesh2x4CycleDetectionTest");

    EXPECT_FALSE(has_cycles)
        << "Bidirectional intermesh traffic in big mesh 2x4 setup should NOT be detected as cycles";
}

TEST(MultiHost, TestGalaxy4x4DualMeshUnicastNoCycle) {
    // This test verifies cycle-free unicast traffic pattern across two 4x4 meshes
    // Pattern: [0,15] -> [0,12] -> [1,0] -> [1,3] -> [0,15] (should NOT form a cycle)
    // Run with: tt-run --mock-cluster-rank-binding
    // tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml
    //           --rank-binding tests/tt_metal/distributed/config/galaxy_4x4_strict_connection_rank_bindings.yaml

    // Configure ethernet cores for fabric routing
    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        kFabricConfig, num_routing_planes);

    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_4x4_dual_mesh_graph_strict.yaml";

    auto control_plane = std::make_unique<ControlPlane>(mesh_graph_desc_path.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    // Define traffic pattern from test_4x4_cycles.yaml - UnicastNoCycle
    // Device topology: Each mesh is 4x4 in row-major order
    // [0,15] -> [0,12] -> [1,0] -> [1,3] -> [0,15]
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs = {
        {FabricNodeId(MeshId{0}, 15), FabricNodeId(MeshId{0}, 12)},  // [0,15] -> [0,12] (intramesh)
        {FabricNodeId(MeshId{0}, 12), FabricNodeId(MeshId{1}, 0)},   // [0,12] -> [1,0]  (intermesh)
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 3)},    // [1,0]  -> [1,3]  (intramesh)
        {FabricNodeId(MeshId{1}, 3), FabricNodeId(MeshId{0}, 15)},   // [1,3]  -> [0,15] (intermesh)
    };

    log_info(
        tt::LogTest,
        "Testing {} unicast traffic pairs for cycles in 4x4 dual mesh (UnicastNoCycle pattern)",
        traffic_pairs.size());
    log_info(tt::LogTest, "Traffic pattern: [0,15] -> [0,12] -> [1,0] -> [1,3] -> [0,15]");

    // Test cycle detection - this pattern should NOT create a cycle
    bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastNoCycle");

    // Debug: Print result before asserting
    log_info(tt::LogTest, "=== Cycle Detection Result ===");
    log_info(tt::LogTest, "Expected: NO cycles");
    log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
    log_info(tt::LogTest, "==============================");

    EXPECT_FALSE(has_cycles) << "Unicast traffic pattern [0,15] -> [0,12] -> [1,0] -> [1,3] -> [0,15] should NOT "
                                "form a cycle";
}

TEST(MultiHost, TestGalaxy4x4DualMeshUnicastHasCycle) {
    // This test verifies cycle detection with a traffic pattern that creates cycles
    // Pattern involves: [0,15] -> [0,12], [0,13] -> [1,1], [1,0] -> [1,3], [1,2] -> [0,14]
    // Expected cycles from devices: [0,15] [0,14] [0,13] [0,12] [1,0] [1,1] [1,2] [1,3]
    // Run with: tt-run --mock-cluster-rank-binding
    // tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml
    //           --rank-binding tests/tt_metal/distributed/config/galaxy_4x4_strict_connection_rank_bindings.yaml

    // Configure ethernet cores for fabric routing
    uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
    tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
        kFabricConfig, num_routing_planes);

    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_4x4_dual_mesh_graph_strict.yaml";

    auto control_plane = std::make_unique<ControlPlane>(mesh_graph_desc_path.string());
    control_plane->initialize_fabric_context(kFabricConfig);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    // Define traffic pattern from test_4x4_cycles.yaml - UnicastHasCycle
    // This pattern creates cycles involving 8 devices across two meshes
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs = {
        {FabricNodeId(MeshId{0}, 15), FabricNodeId(MeshId{0}, 12)},  // [0,15] -> [0,12] (intramesh)
        {FabricNodeId(MeshId{0}, 13), FabricNodeId(MeshId{1}, 1)},   // [0,13] -> [1,1]  (intermesh)
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 3)},    // [1,0]  -> [1,3]  (intramesh)
        {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{0}, 14)},   // [1,2]  -> [0,14] (intermesh)
    };

    log_info(
        tt::LogTest,
        "Testing {} unicast traffic pairs for cycles in 4x4 dual mesh (UnicastHasCycle pattern)",
        traffic_pairs.size());
    log_info(
        tt::LogTest, "Traffic pattern creates cycles involving: [0,15] [0,14] [0,13] [0,12] [1,0] [1,1] [1,2] [1,3]");

    // Test cycle detection - this pattern SHOULD create cycles
    bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastHasCycle");

    // Debug: Print result before asserting
    log_info(tt::LogTest, "=== Cycle Detection Result ===");
    log_info(tt::LogTest, "Expected: HAS cycles");
    log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
    log_info(tt::LogTest, "==============================");

    EXPECT_TRUE(has_cycles) << "Unicast traffic pattern with cross-mesh dependencies should form cycles and be "
                               "detected";
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
