// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/distributed_context.hpp>
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

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "DualGalaxyCycleDetectionTest");
        EXPECT_FALSE(has_cycles)
            << "Bidirectional intermesh traffic between dual Galaxy meshes should NOT be detected as cycles";
    }
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

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "DualT3K2x4CycleDetectionTest");
        EXPECT_FALSE(has_cycles)
            << "Bidirectional intermesh traffic between dual T3K 2x4 meshes should NOT be detected as cycles";
    }
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

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "Split2x2CycleDetectionTest");
        EXPECT_FALSE(has_cycles)
            << "Bidirectional intermesh traffic in split 2x2 setup should NOT be detected as cycles";
    }
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

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(intermesh_pairs, "BigMesh2x4CycleDetectionTest");
        EXPECT_FALSE(has_cycles)
            << "Bidirectional intermesh traffic in big mesh 2x4 setup should NOT be detected as cycles";
    }
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

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastNoCycle");

        // Debug: Print result
        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Expected: NO cycles");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        EXPECT_FALSE(has_cycles) << "Unicast traffic pattern [0,15] -> [0,12] -> [1,0] -> [1,3] -> [0,15] should NOT "
                                    "form a cycle";
    }
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

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastHasCycle");

        // Debug: Print result
        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Expected: HAS cycles");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        EXPECT_TRUE(has_cycles) << "Unicast traffic pattern with cross-mesh dependencies should form cycles and be "
                                   "detected";
    }
}

TEST(MultiHost, TestGalaxy4x4DualMeshUnicastWithChipId) {
    // This test verifies basic communication patterns using chip IDs
    // Adapted from test_galaxy_4x4.yaml UnicastWithChipId
    // Pattern includes:
    // - Intra-mesh corner-to-corner communication in both meshes
    // - Cross-mesh communication in both directions
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

    // Define traffic pattern from test_galaxy_4x4.yaml - UnicastWithChipId
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs;

    // Within Mesh 0 - test corner to corner
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 15)});  // [0,0] -> [0,15]

    // Within Mesh 1 - test corner to corner
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 15)});  // [1,0] -> [1,15]

    // Cross-mesh communication - Mesh 0 to Mesh 1
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{1}, 0)});  // [0,3] -> [1,0]
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{1}, 5)});  // [0,3] -> [1,5]

    // Cross-mesh communication - Mesh 1 to Mesh 0
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 3), FabricNodeId(MeshId{0}, 0)});  // [1,3] -> [0,0]
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 3), FabricNodeId(MeshId{0}, 5)});  // [1,3] -> [0,5]

    log_info(tt::LogTest, "Testing {} traffic pairs for basic unicast patterns in 4x4 dual mesh", traffic_pairs.size());
    log_info(tt::LogTest, "Pattern: intra-mesh corner-to-corner + cross-mesh bidirectional");

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastWithChipId");

        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        // This pattern should not create cycles - it's straightforward unicast communication
        EXPECT_FALSE(has_cycles) << "Basic unicast patterns should not create resource dependency cycles";
    }
}

TEST(MultiHost, TestGalaxy4x4DualMeshUnicastWithMeshCoordinates) {
    // This test verifies communication patterns using mesh coordinates
    // Adapted from test_galaxy_4x4.yaml UnicastWithMeshCoordinates
    // For a 4x4 mesh in row-major order:
    //   [0,0] = chip 0,  [3,3] = chip 15
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

    // Define traffic pattern from test_galaxy_4x4.yaml - UnicastWithMeshCoordinates
    // Using mesh coordinates: [0, [row, col]]
    // For 4x4 mesh: chip_id = row * 4 + col
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs;

    // [0, [0,0]] -> [1, [0,0]]: chip 0 of mesh 0 to chip 0 of mesh 1
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 0)});

    // [0, [3,3]] -> [1, [3,3]]: chip 15 of mesh 0 to chip 15 of mesh 1
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 15), FabricNodeId(MeshId{1}, 15)});

    log_info(tt::LogTest, "Testing {} traffic pairs using mesh coordinates in 4x4 dual mesh", traffic_pairs.size());
    log_info(tt::LogTest, "Pattern: corresponding corners across meshes");

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastWithMeshCoordinates");

        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        // This pattern should not create cycles - it's simple cross-mesh unicast
        EXPECT_FALSE(has_cycles)
            << "Cross-mesh unicast using mesh coordinates should not create resource dependency cycles";
    }
}

TEST(MultiHost, TestGalaxy4x4DualMeshUnicastAllToAllSubset) {
    // This test verifies a TRUE all-to-all communication pattern across all devices
    // Adapted from test_galaxy_4x4.yaml UnicastAlltoAll
    // Uses ALL 32 devices (16 from each mesh) where every device sends to every other device,
    // creating 992 traffic pairs. This includes both intra-mesh and inter-mesh traffic with
    // maximum path overlap, which should create circular resource dependencies.
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

    // Create a TRUE all-to-all pattern with ALL 16 devices per mesh (32 total)
    // Each device sends to every other device across both meshes
    // This creates maximum overlapping intra-mesh and inter-mesh patterns
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs;

    // All devices across both meshes (16 devices × 2 meshes = 32 total)
    std::vector<FabricNodeId> all_devices;
    for (uint32_t mesh_id = 0; mesh_id < 2; ++mesh_id) {
        for (uint32_t chip_id = 0; chip_id < 16; ++chip_id) {
            all_devices.push_back(FabricNodeId(MeshId{mesh_id}, chip_id));
        }
    }

    // True all-to-all: every device sends to every other device (excluding self)
    // 32 devices × 31 destinations = 992 traffic pairs
    for (const auto& src : all_devices) {
        for (const auto& dst : all_devices) {
            if (src != dst) {  // Don't send to self
                traffic_pairs.push_back({src, dst});
            }
        }
    }

    log_info(
        tt::LogTest, "Testing {} traffic pairs for TRUE all-to-all pattern in 4x4 dual mesh", traffic_pairs.size());
    log_info(tt::LogTest, "Pattern: 32 devices (16 per mesh), every device sends to every other device");

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastAllToAllSubset");

        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Expected: HAS cycles");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        // True all-to-all with 992 traffic pairs creates maximum path overlap where flows
        // traverse each other's destinations, forming circular resource dependencies
        EXPECT_TRUE(has_cycles) << "TRUE all-to-all pattern (all 32 devices) should create resource dependency cycles";
    }
}

TEST(MultiHost, TestGalaxy4x4DualMeshUnicastRandomPairing) {
    // This test verifies a random pairing pattern (deterministic for testing)
    // Adapted from test_galaxy_4x4.yaml UnicastRandomPairing
    // Uses ALL 32 chips where each chip appears in exactly one sender/receiver pair
    // Creates 32 pairs with a mix of intra-mesh and inter-mesh communication
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

    // Create deterministic "random" pairing where each chip appears exactly once
    // 32 chips total (16 per mesh) → 16 pairs
    // Mix of intra-mesh and inter-mesh communication
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs = {
        // Inter-mesh pairs (mesh 0 to mesh 1)
        {FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 15)},
        {FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{1}, 14)},
        {FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{1}, 13)},
        {FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{1}, 12)},
        {FabricNodeId(MeshId{0}, 4), FabricNodeId(MeshId{1}, 11)},
        {FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{1}, 10)},
        // Inter-mesh pairs (mesh 1 to mesh 0)
        {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{0}, 15)},
        {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{0}, 14)},
        {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{0}, 13)},
        {FabricNodeId(MeshId{1}, 3), FabricNodeId(MeshId{0}, 12)},
        {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{0}, 11)},
        {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{0}, 10)},
        // Intra-mesh pairs (within mesh 0)
        {FabricNodeId(MeshId{0}, 6), FabricNodeId(MeshId{0}, 9)},
        {FabricNodeId(MeshId{0}, 7), FabricNodeId(MeshId{0}, 8)},
        // Intra-mesh pairs (within mesh 1)
        {FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{1}, 9)},
        {FabricNodeId(MeshId{1}, 7), FabricNodeId(MeshId{1}, 8)},
    };

    log_info(tt::LogTest, "Testing {} traffic pairs for random pairing pattern in 4x4 dual mesh", traffic_pairs.size());
    log_info(tt::LogTest, "Pattern: 32 chips, each appears in exactly one sender/receiver pair");

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4UnicastRandomPairing");

        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        // This particular arrangement should not create cycles
        EXPECT_FALSE(has_cycles) << "Random unidirectional pairing should not create resource dependency cycles";
    }
}

TEST(MultiHost, TestGalaxy4x4DualMeshCycleDetectionStressTest) {
    // This test stresses cycle detection with complex bidirectional, fan-out, and fan-in patterns
    // Adapted from test_galaxy_4x4.yaml CycleDetectionStressTest
    // Pattern includes:
    // - 3 bidirectional traffic pairs
    // - 1 fan-out pattern (one source to multiple destinations)
    // - 1 fan-in pattern (multiple sources to one destination)
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

    // Define complex traffic pattern from test_galaxy_4x4.yaml - CycleDetectionStressTest
    std::vector<std::pair<FabricNodeId, FabricNodeId>> traffic_pairs;

    // Bidirectional traffic pattern 1: Corner devices
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 15)});  // [0,0] -> [1,15]
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 15), FabricNodeId(MeshId{0}, 0)});  // [1,15] -> [0,0]

    // Bidirectional traffic pattern 2: Middle boundary devices
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 12), FabricNodeId(MeshId{1}, 3)});  // [0,12] -> [1,3]
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 3), FabricNodeId(MeshId{0}, 12)});  // [1,3] -> [0,12]

    // Bidirectional traffic pattern 3: Another pair
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 7), FabricNodeId(MeshId{1}, 8)});  // [0,7] -> [1,8]
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 8), FabricNodeId(MeshId{0}, 7)});  // [1,8] -> [0,7]

    // Fan-out pattern: one source to multiple destinations
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{1}, 2)});   // [0,5] -> [1,2]
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{1}, 7)});   // [0,5] -> [1,7]
    traffic_pairs.push_back({FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{1}, 11)});  // [0,5] -> [1,11]

    // Fan-in pattern: multiple sources to one destination
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{0}, 10)});   // [1,6] -> [0,10]
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 9), FabricNodeId(MeshId{0}, 10)});   // [1,9] -> [0,10]
    traffic_pairs.push_back({FabricNodeId(MeshId{1}, 14), FabricNodeId(MeshId{0}, 10)});  // [1,14] -> [0,10]

    log_info(
        tt::LogTest, "Testing {} traffic pairs for cycle detection stress test in 4x4 dual mesh", traffic_pairs.size());
    log_info(tt::LogTest, "Pattern: 3 bidirectional pairs + fan-out + fan-in");

    // Only run cycle detection on rank 0 to avoid redundant computation
    auto distributed_ctx = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*distributed_ctx->rank() == 0) {
        log_info(tt::LogTest, "Running cycle detection on rank 0");
        bool has_cycles = control_plane->detect_inter_mesh_cycles(traffic_pairs, "Galaxy4x4CycleStressTest");

        log_info(tt::LogTest, "=== Cycle Detection Result ===");
        log_info(tt::LogTest, "Expected: NO cycles");
        log_info(tt::LogTest, "Actual: {} cycles detected", has_cycles ? "HAS" : "NO");
        log_info(tt::LogTest, "==============================");

        // While this pattern has bidirectional traffic, the routing paths do not create
        // circular resource dependencies. Bidirectional traffic alone doesn't guarantee cycles -
        // cycles only occur when routing paths share resources in a circular manner (like in
        // the HasCycle and AllToAll patterns). This stress test validates that cycle detection
        // correctly identifies when complex traffic patterns do NOT create cycles.
        EXPECT_FALSE(has_cycles) << "Stress test pattern should not create resource dependency cycles";
    }
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
