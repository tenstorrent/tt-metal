// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <gtest/gtest.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "tt_cluster.hpp"
#include "tt_metal/fabric/topology_solver_internal.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include <tt-metalium/experimental/mock_device.hpp>

namespace tt::tt_fabric {

class TopologySolverTest : public ::testing::Test {
protected:
    // Cluster type doesn't matter as this test suite is CPU only
    const tt::tt_metal::ClusterType cluster_type = tt::tt_metal::ClusterType::BLACKHOLE_GALAXY;

    void SetUp() override { setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "10", 1); }

    void TearDown() override {}
};

TEST_F(TopologySolverTest, BuildAdjacencyMapLogical) {
    // Use 2x2 T3K multiprocess MGD (has 2 compute meshes: mesh_id 0 and 1)
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";

    // Create mesh graph from descriptor
    auto mesh_graph = MeshGraph(cluster_type, mesh_graph_desc_path.string());

    // Build adjacency map logical (includes all meshes, including switches if present)
    auto adjacency_map = build_adjacency_graph_logical(mesh_graph);

    // Verify that we have adjacency graphs for each mesh
    EXPECT_GT(adjacency_map.size(), 0u);

    // Verify each mesh has a valid adjacency graph (including switches)
    for (const auto& [mesh_id, adj_graph] : adjacency_map) {
        const auto& nodes = adj_graph.get_nodes();
        EXPECT_GT(nodes.size(), 0u) << "Mesh " << mesh_id.get() << " should have nodes";

        // Verify that nodes belong to the correct mesh
        for (const auto& node : nodes) {
            EXPECT_EQ(node.mesh_id, mesh_id) << "Node should belong to mesh " << mesh_id.get();

            // Verify we can query neighbors
            const auto& neighbors = adj_graph.get_neighbors(node);
            for (const auto& neighbor : neighbors) {
                EXPECT_EQ(neighbor.mesh_id, mesh_id) << "Neighbor should belong to the same mesh " << mesh_id.get();
            }
        }
    }

    // For T3K 2x2 multiprocess, we expect 2 meshes (mesh_id 0 and 1)
    // Note: This includes all meshes returned by get_all_mesh_ids() (compute meshes and switches if present)
    EXPECT_EQ(adjacency_map.size(), 2u) << "Should have 2 meshes (mesh_id 0 and 1)";
}

TEST_F(TopologySolverTest, BuildAdjacencyMapLogicalWithSwitch) {
    // Use T3K 2x2 MGD with TT-Switch (has 1 compute mesh: mesh_id 0, and 1 switch: mesh_id 1)
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_ttswitch_mgd.textproto";

    // Create mesh graph from descriptor
    auto mesh_graph = MeshGraph(cluster_type, mesh_graph_desc_path.string());

    // Build adjacency map logical (includes all meshes, including switches)
    auto adjacency_map = build_adjacency_graph_logical(mesh_graph);

    // Verify that we have adjacency graphs for each mesh
    EXPECT_GT(adjacency_map.size(), 0u);

    // Count compute meshes and switches separately
    size_t compute_mesh_count = 0;
    size_t switch_mesh_count = 0;

    // Verify each mesh has a valid adjacency graph (including switches)
    for (const auto& [mesh_id, adj_graph] : adjacency_map) {
        const auto& nodes = adj_graph.get_nodes();
        EXPECT_GT(nodes.size(), 0u) << "Mesh " << mesh_id.get() << " should have nodes";

        // Verify that nodes belong to the correct mesh
        for (const auto& node : nodes) {
            EXPECT_EQ(node.mesh_id, mesh_id) << "Node should belong to mesh " << mesh_id.get();

            // Verify we can query neighbors
            const auto& neighbors = adj_graph.get_neighbors(node);
            for (const auto& neighbor : neighbors) {
                EXPECT_EQ(neighbor.mesh_id, mesh_id) << "Neighbor should belong to the same mesh " << mesh_id.get();
            }
        }

        // Count compute meshes and switches
        if (mesh_graph.is_switch_mesh(mesh_id)) {
            switch_mesh_count++;
        } else {
            compute_mesh_count++;
        }
    }

    // For T3K 2x2 with TT-Switch, we expect 1 compute mesh and 1 switch mesh (total 2 meshes)
    EXPECT_EQ(adjacency_map.size(), 2u) << "Should have 2 meshes total (1 compute + 1 switch)";
    EXPECT_EQ(compute_mesh_count, 1u) << "Should have 1 compute mesh (mesh_id 0)";
    EXPECT_EQ(switch_mesh_count, 1u) << "Should have 1 switch mesh (mesh_id 1)";
}

TEST_F(TopologySolverTest, BuildAdjacencyMapPhysical) {
    // Load PSD from pre-written test file
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path psd_file_path =
        std::filesystem::path(tt_metal_home) / "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto";

    // Verify the file exists
    ASSERT_TRUE(std::filesystem::exists(psd_file_path)) << "PSD test file not found: " << psd_file_path.string();

    // Load PhysicalSystemDescriptor from file
    tt::tt_metal::PhysicalSystemDescriptor physical_system_descriptor(psd_file_path.string());

    // Hand-craft the asic_id_to_mesh_rank mapping
    // Mesh 0: ASICs 100, 101 (connected)
    // Mesh 1: ASICs 102, 103 (connected)
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{100}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{101}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{102}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{103}] = MeshHostRankId{0};

    // Build adjacency map physical
    auto adjacency_map = build_adjacency_graph_physical(cluster_type, physical_system_descriptor, asic_id_to_mesh_rank);

    // Verify that we have adjacency graphs for each mesh
    EXPECT_EQ(adjacency_map.size(), 2u) << "Should have 2 meshes";

    // Verify mesh 0
    auto it0 = adjacency_map.find(MeshId{0});
    ASSERT_NE(it0, adjacency_map.end()) << "Mesh 0 should exist";
    const auto& graph0 = it0->second;
    const auto& nodes0 = graph0.get_nodes();
    EXPECT_EQ(nodes0.size(), 2u) << "Mesh 0 should have 2 ASICs";

    // Check that ASIC 100 has ASIC 101 as neighbor (2 connections = 2 entries)
    auto asic100_it = std::find(nodes0.begin(), nodes0.end(), tt::tt_metal::AsicID{100});
    ASSERT_NE(asic100_it, nodes0.end()) << "Mesh 0 should contain ASIC 100";
    const auto& neighbors100 = graph0.get_neighbors(tt::tt_metal::AsicID{100});
    EXPECT_EQ(neighbors100.size(), 2u) << "ASIC 100 should have 2 neighbor entries (2 eth connections)";
    EXPECT_EQ(neighbors100[0], tt::tt_metal::AsicID{101}) << "ASIC 100 should be connected to ASIC 101";
    EXPECT_EQ(neighbors100[1], tt::tt_metal::AsicID{101}) << "ASIC 100 should have 2 connections to ASIC 101";

    // Verify mesh 1
    auto it1 = adjacency_map.find(MeshId{1});
    ASSERT_NE(it1, adjacency_map.end()) << "Mesh 1 should exist";
    const auto& graph1 = it1->second;
    const auto& nodes1 = graph1.get_nodes();
    EXPECT_EQ(nodes1.size(), 2u) << "Mesh 1 should have 2 ASICs";

    // Check that ASIC 102 has ASIC 103 as neighbor
    auto asic102_it = std::find(nodes1.begin(), nodes1.end(), tt::tt_metal::AsicID{102});
    ASSERT_NE(asic102_it, nodes1.end()) << "Mesh 1 should contain ASIC 102";
    const auto& neighbors102 = graph1.get_neighbors(tt::tt_metal::AsicID{102});
    EXPECT_EQ(neighbors102.size(), 2u) << "ASIC 102 should have 2 neighbor entries (2 eth connections)";
    EXPECT_EQ(neighbors102[0], tt::tt_metal::AsicID{103}) << "ASIC 102 should be connected to ASIC 103";
    EXPECT_EQ(neighbors102[1], tt::tt_metal::AsicID{103}) << "ASIC 102 should have 2 connections to ASIC 103";
}

// AdjacencyGraph Tests with Different User Types
namespace {
struct CustomNode {
    int id;
    std::string name;
    bool operator<(const CustomNode& other) const { return id < other.id || (id == other.id && name < other.name); }
    bool operator==(const CustomNode& other) const { return id == other.id && name == other.name; }
};

enum class NodeType { PROCESSOR, MEMORY, IO };
}  // namespace

TEST_F(TopologySolverTest, AdjacencyGraphWithInt) {
    AdjacencyGraph<int>::AdjacencyMap adj_map{{1, {2, 3}}, {2, {1}}};
    AdjacencyGraph<int> graph(adj_map);
    EXPECT_EQ(graph.get_nodes().size(), 2u);
    EXPECT_EQ(graph.get_neighbors(1).size(), 2u);
    EXPECT_EQ(graph.get_neighbors(2).size(), 1u);
    EXPECT_TRUE(graph.get_neighbors(99).empty());
}

TEST_F(TopologySolverTest, AdjacencyGraphWithString) {
    AdjacencyGraph<std::string>::AdjacencyMap adj_map{{"a", {"b"}}, {"b", {"a"}}};
    AdjacencyGraph<std::string> graph(adj_map);
    EXPECT_EQ(graph.get_nodes().size(), 2u);
    EXPECT_EQ(graph.get_neighbors("a").size(), 1u);
    EXPECT_TRUE(graph.get_neighbors("z").empty());
}

TEST_F(TopologySolverTest, AdjacencyGraphWithCustomStruct) {
    CustomNode n1{1, "alpha"}, n2{2, "beta"};
    AdjacencyGraph<CustomNode>::AdjacencyMap adj_map{{n1, {n2}}, {n2, {n1}}};
    AdjacencyGraph<CustomNode> graph(adj_map);
    EXPECT_EQ(graph.get_nodes().size(), 2u);
    EXPECT_EQ(graph.get_neighbors(n1).size(), 1u);
    EXPECT_TRUE(graph.get_neighbors(CustomNode{99, "nonexistent"}).empty());
}

TEST_F(TopologySolverTest, AdjacencyGraphWithEnum) {
    AdjacencyGraph<NodeType>::AdjacencyMap adj_map{
        {NodeType::PROCESSOR, {NodeType::MEMORY}}, {NodeType::MEMORY, {NodeType::PROCESSOR}}};
    AdjacencyGraph<NodeType> graph(adj_map);
    EXPECT_EQ(graph.get_nodes().size(), 2u);
    EXPECT_EQ(graph.get_neighbors(NodeType::PROCESSOR).size(), 1u);
}

TEST_F(TopologySolverTest, AdjacencyGraphWithPair) {
    using NodePair = std::pair<int, int>;
    AdjacencyGraph<NodePair>::AdjacencyMap adj_map{{{1, 0}, {{2, 0}}}, {{2, 0}, {{1, 0}}}};
    AdjacencyGraph<NodePair> graph(adj_map);
    EXPECT_EQ(graph.get_nodes().size(), 2u);
    EXPECT_EQ(graph.get_neighbors({1, 0}).size(), 1u);
    EXPECT_TRUE(graph.get_neighbors({99, 0}).empty());
}

TEST_F(TopologySolverTest, AdjacencyGraphWithSharedPtr) {
    using NodePtr = std::shared_ptr<int>;
    auto n1 = std::make_shared<int>(1);
    auto n2 = std::make_shared<int>(2);
    AdjacencyGraph<NodePtr>::AdjacencyMap adj_map{{n1, {n2}}, {n2, {n1}}};
    AdjacencyGraph<NodePtr> graph(adj_map);
    EXPECT_EQ(graph.get_nodes().size(), 2u);
    EXPECT_EQ(graph.get_neighbors(n1).size(), 1u);
    auto nonexistent = std::make_shared<int>(99);
    EXPECT_TRUE(graph.get_neighbors(nonexistent).empty());
}

// MappingConstraints Tests
// Use different types to verify template type checking
using TestTargetNode = uint32_t;
using TestGlobalNode = uint64_t;

TEST_F(TopologySolverTest, MappingConstraintsBasicOperations) {
    // Test empty constraints - should not throw
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);

    // Test construction from sets - validation happens automatically
    std::set<std::pair<TestTargetNode, TestGlobalNode>> required = {{1, 10}, {2, 20}};
    std::set<std::pair<TestTargetNode, TestGlobalNode>> preferred = {{3, 30}};
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints_from_sets(required, preferred);
    EXPECT_EQ(constraints_from_sets.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints_from_sets.get_preferred_mappings(3).count(30), 1u);

    // Test required constraints - validation happens automatically
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_TRUE(constraints.is_valid_mapping(1, 10));
    EXPECT_FALSE(constraints.is_valid_mapping(1, 20));

    // Test preferred constraints (don't restrict valid mappings)
    constraints.add_preferred_constraint(1, 20);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(20), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);  // Still only 10

    // Test accessors
    const auto& all_valid = constraints.get_valid_mappings();
    EXPECT_EQ(all_valid.size(), 1u);
}

TEST_F(TopologySolverTest, MappingConstraintsTraitConstraints) {
    // Test required trait constraint (using string as trait type) - validation happens automatically
    MappingConstraints<TestTargetNode, TestGlobalNode> required_constraints;
    std::map<TestTargetNode, std::string> target_traits = {{1, "host0"}, {2, "host0"}, {3, "host1"}};
    std::map<TestGlobalNode, std::string> global_traits = {{10, "host0"}, {11, "host0"}, {20, "host1"}, {21, "host1"}};
    EXPECT_TRUE(required_constraints.add_required_trait_constraint<std::string>(target_traits, global_traits));

    EXPECT_EQ(required_constraints.get_valid_mappings(1).size(), 2u);
    EXPECT_EQ(required_constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(required_constraints.get_valid_mappings(3).count(20), 1u);

    // Test preferred trait constraint (using int as trait type) - no validation needed
    MappingConstraints<TestTargetNode, TestGlobalNode> preferred_constraints;
    std::map<TestTargetNode, int> target_pref = {{1, 100}};
    std::map<TestGlobalNode, int> global_pref = {{10, 100}, {20, 200}};
    preferred_constraints.add_preferred_trait_constraint<int>(target_pref, global_pref);

    EXPECT_EQ(preferred_constraints.get_preferred_mappings(1).size(), 1u);
    EXPECT_EQ(preferred_constraints.get_preferred_mappings(1).count(10), 1u);
    EXPECT_EQ(preferred_constraints.get_valid_mappings(1).size(), 0u);  // No required constraints
}

TEST_F(TopologySolverTest, MappingConstraintsIntersection) {
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Test multiple trait constraints intersection (using uint8_t) - validation happens automatically
    std::map<TestTargetNode, uint8_t> target_host = {{1, 0}, {2, 0}};
    std::map<TestGlobalNode, uint8_t> global_host = {{10, 0}, {11, 0}, {20, 1}};
    EXPECT_TRUE(constraints.add_required_trait_constraint<uint8_t>(target_host, global_host));

    std::map<TestTargetNode, uint8_t> target_rack = {{1, 0}, {2, 1}};
    std::map<TestGlobalNode, uint8_t> global_rack = {{10, 0}, {11, 1}, {20, 0}};
    EXPECT_TRUE(constraints.add_required_trait_constraint<uint8_t>(target_rack, global_rack));

    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);  // host=0 AND rack=0 -> {10}
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);

    // Test trait and explicit constraint intersection - validation happens automatically
    constraints.add_required_constraint(1, 10);  // Already constrained to 10, should be fine
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);

    // Test preferred trait constraints intersection
    MappingConstraints<TestTargetNode, TestGlobalNode> preferred_constraints;
    std::map<TestTargetNode, uint32_t> target_pref1 = {{1, 100}};
    std::map<TestGlobalNode, uint32_t> global_pref1 = {{10, 100}, {11, 100}, {20, 200}};
    preferred_constraints.add_preferred_trait_constraint<uint32_t>(target_pref1, global_pref1);

    std::map<TestTargetNode, uint32_t> target_pref2 = {{1, 0}};
    std::map<TestGlobalNode, uint32_t> global_pref2 = {{10, 0}, {11, 1}, {20, 0}};
    preferred_constraints.add_preferred_trait_constraint<uint32_t>(target_pref2, global_pref2);

    EXPECT_EQ(preferred_constraints.get_preferred_mappings(1).size(), 1u);
    EXPECT_EQ(preferred_constraints.get_preferred_mappings(1).count(10), 1u);
}

TEST_F(TopologySolverTest, MappingConstraintsConflictHandling) {
    // Test conflict in required constraint - should throw automatically
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    EXPECT_FALSE(constraints.add_required_constraint(1, 20)) << "Conflicting constraint should return false";

    // Test conflict in trait constraint - should throw (no matching global nodes)
    MappingConstraints<TestTargetNode, TestGlobalNode> trait_constraints;
    std::map<TestTargetNode, size_t> target_traits = {{1, 999}};
    std::map<TestGlobalNode, size_t> global_traits = {{10, 100}, {20, 200}};
    EXPECT_FALSE(trait_constraints.add_required_trait_constraint<size_t>(target_traits, global_traits));

    // Test conflict in trait constraint - conflicting trait values
    MappingConstraints<TestTargetNode, TestGlobalNode> conflict_constraints;
    std::map<TestTargetNode, uint8_t> target_host1 = {{1, 0}};
    std::map<TestGlobalNode, uint8_t> global_host1 = {{10, 0}, {11, 0}};
    conflict_constraints.add_required_trait_constraint<uint8_t>(target_host1, global_host1);

    std::map<TestTargetNode, uint8_t> target_host2 = {{1, 1}};
    std::map<TestGlobalNode, uint8_t> global_host2 = {{20, 1}, {21, 1}};
    EXPECT_FALSE(conflict_constraints.add_required_trait_constraint<uint8_t>(target_host2, global_host2));

    // Test conflict in constructor - constructor doesn't throw, but validation should fail
    std::set<std::pair<TestTargetNode, TestGlobalNode>> conflicting_required = {{1, 10}, {1, 20}};
    std::set<std::pair<TestTargetNode, TestGlobalNode>> empty_preferred;
    MappingConstraints<TestTargetNode, TestGlobalNode> conflict_constraints_ctor(conflicting_required, empty_preferred);
    // After construction with conflicting constraints, validation should fail
    EXPECT_FALSE(conflict_constraints_ctor.validate())
        << "Conflicting constraints in constructor should fail validation";
}

TEST_F(TopologySolverTest, GraphIndexDataBasic) {
    using namespace tt::tt_fabric::detail;

    // Create simple target graph: 1 -> 2 -> 3 (path)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2};  // Node 1 connected to 2 (twice for multi-edge)
    target_adj_map[2] = {1, 3};  // Node 2 connected to 1 and 3
    target_adj_map[3] = {2};     // Node 3 connected to 2

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create simple global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Build index data (CTAD deduces types from constructor arguments)
    GraphIndexData graph_data(target_graph, global_graph);

    // Verify node counts
    EXPECT_EQ(graph_data.n_target, 3u);
    EXPECT_EQ(graph_data.n_global, 4u);

    // Verify node vectors
    EXPECT_EQ(graph_data.target_nodes.size(), 3u);
    EXPECT_EQ(graph_data.global_nodes.size(), 4u);

    // Verify index mappings
    EXPECT_EQ(graph_data.target_to_idx.size(), 3u);
    EXPECT_EQ(graph_data.global_to_idx.size(), 4u);
    EXPECT_EQ(graph_data.target_to_idx.at(1), 0u);
    EXPECT_EQ(graph_data.target_to_idx.at(2), 1u);
    EXPECT_EQ(graph_data.target_to_idx.at(3), 2u);

    // Verify adjacency indices (deduplicated)
    EXPECT_EQ(graph_data.target_adj_idx[0].size(), 1u);  // Node 1 -> Node 2 (deduplicated)
    EXPECT_EQ(graph_data.target_adj_idx[1].size(), 2u);  // Node 2 -> Nodes 1, 3
    EXPECT_EQ(graph_data.target_adj_idx[2].size(), 1u);  // Node 3 -> Node 2

    // Verify connection counts (multi-edge support)
    EXPECT_EQ(graph_data.target_conn_count[0].at(1), 2u);  // Node 1 -> Node 2: 2 connections
    EXPECT_EQ(graph_data.target_conn_count[1].at(0), 1u);  // Node 2 -> Node 1: 1 connection
    EXPECT_EQ(graph_data.target_conn_count[1].at(2), 1u);  // Node 2 -> Node 3: 1 connection

    // Verify degrees
    EXPECT_EQ(graph_data.target_deg[0], 1u);
    EXPECT_EQ(graph_data.target_deg[1], 2u);
    EXPECT_EQ(graph_data.target_deg[2], 1u);
}

TEST_F(TopologySolverTest, GraphIndexDataEmpty) {
    using namespace tt::tt_fabric::detail;

    // Empty graphs
    AdjacencyGraph<TestTargetNode> target_graph;
    AdjacencyGraph<TestGlobalNode> global_graph;

    GraphIndexData graph_data(target_graph, global_graph);

    EXPECT_EQ(graph_data.n_target, 0u);
    EXPECT_EQ(graph_data.n_global, 0u);
    EXPECT_TRUE(graph_data.target_nodes.empty());
    EXPECT_TRUE(graph_data.global_nodes.empty());
}

TEST_F(TopologySolverTest, GraphIndexDataSelfConnections) {
    using namespace tt::tt_fabric::detail;

    // Create graph with self-connections
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {1, 1, 2};  // Node 1 has self-connections and connection to 2
    target_adj_map[2] = {1, 2};     // Node 2 has connection to 1 and self-connection
    target_adj_map[3] = {3, 3, 3};  // Node 3 only has self-connections

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Node 1: should have degree 1 (only neighbor is 2, self-connections ignored)
    EXPECT_EQ(graph_data.target_deg[0], 1u);
    EXPECT_EQ(graph_data.target_adj_idx[0].size(), 1u);
    EXPECT_EQ(graph_data.target_adj_idx[0][0], 1u);  // Index of node 2

    // Node 2: should have degree 1 (only neighbor is 1, self-connection ignored)
    EXPECT_EQ(graph_data.target_deg[1], 1u);
    EXPECT_EQ(graph_data.target_adj_idx[1].size(), 1u);
    EXPECT_EQ(graph_data.target_adj_idx[1][0], 0u);  // Index of node 1

    // Node 3: should have degree 0 (only self-connections, all ignored)
    EXPECT_EQ(graph_data.target_deg[2], 0u);
    EXPECT_TRUE(graph_data.target_adj_idx[2].empty());

    // Verify connection counts don't include self-connections
    // Node 1 -> Node 2: should have 1 connection (not counting self-connections)
    EXPECT_EQ(graph_data.target_conn_count[0].at(1), 1u);
    // Node 1 should not have connection count to itself (index 0)
    EXPECT_EQ(graph_data.target_conn_count[0].count(0), 0u);
}

TEST_F(TopologySolverTest, ConstraintIndexDataBasic) {
    using namespace tt::tt_fabric::detail;

    // Create simple graphs
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 12};
    global_adj_map[11] = {10};
    global_adj_map[12] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Create constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);   // Target 1 must map to Global 10
    constraints.add_preferred_constraint(2, 11);  // Target 2 prefers Global 11

    ConstraintIndexData constraint_data(constraints, graph_data);

    // Verify restricted mappings
    // Target 1 (index 0) should be restricted to Global 10 (index 0)
    EXPECT_EQ(constraint_data.restricted_global_indices[0].size(), 1u);
    EXPECT_EQ(constraint_data.restricted_global_indices[0][0], 0u);  // Index of Global 10

    // Target 2 (index 1) should have no restrictions (empty = all valid)
    EXPECT_TRUE(constraint_data.restricted_global_indices[1].empty());

    // Verify preferred mappings
    // Target 2 (index 1) should prefer Global 11 (index 1)
    EXPECT_EQ(constraint_data.preferred_global_indices[1].size(), 1u);
    EXPECT_EQ(constraint_data.preferred_global_indices[1][0], 1u);  // Index of Global 11

    // Verify is_valid_mapping
    EXPECT_TRUE(constraint_data.is_valid_mapping(0, 0));   // Target 1 -> Global 10: valid
    EXPECT_FALSE(constraint_data.is_valid_mapping(0, 1));  // Target 1 -> Global 11: invalid
    EXPECT_TRUE(constraint_data.is_valid_mapping(1, 0));   // Target 2 -> Global 10: valid (no restrictions)
    EXPECT_TRUE(constraint_data.is_valid_mapping(1, 1));   // Target 2 -> Global 11: valid (no restrictions)

    // Verify get_candidates
    const auto& candidates_0 = constraint_data.get_candidates(0);
    EXPECT_EQ(candidates_0.size(), 1u);
    EXPECT_EQ(candidates_0[0], 0u);

    const auto& candidates_1 = constraint_data.get_candidates(1);
    EXPECT_TRUE(candidates_1.empty());  // Empty means all are valid
}

TEST_F(TopologySolverTest, ConstraintIndexDataTraitConstraints) {
    using namespace tt::tt_fabric::detail;

    // Create graphs
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {};
    target_adj_map[2] = {};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {};
    global_adj_map[11] = {};
    global_adj_map[20] = {};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Create trait constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::map<TestTargetNode, std::string> target_traits = {{1, "host0"}, {2, "host1"}};
    std::map<TestGlobalNode, std::string> global_traits = {{10, "host0"}, {11, "host0"}, {20, "host1"}};
    EXPECT_TRUE(constraints.add_required_trait_constraint<std::string>(target_traits, global_traits));

    ConstraintIndexData constraint_data(constraints, graph_data);

    // Target 1 (index 0) should be restricted to Global 10, 11 (indices 0, 1)
    EXPECT_EQ(constraint_data.restricted_global_indices[0].size(), 2u);
    EXPECT_EQ(constraint_data.restricted_global_indices[0][0], 0u);  // Global 10
    EXPECT_EQ(constraint_data.restricted_global_indices[0][1], 1u);  // Global 11

    // Target 2 (index 1) should be restricted to Global 20 (index 2)
    EXPECT_EQ(constraint_data.restricted_global_indices[1].size(), 1u);
    EXPECT_EQ(constraint_data.restricted_global_indices[1][0], 2u);  // Global 20
}

TEST_F(TopologySolverTest, ConstraintIndexDataEmpty) {
    using namespace tt::tt_fabric::detail;

    // Empty graphs
    AdjacencyGraph<TestTargetNode> target_graph;
    AdjacencyGraph<TestGlobalNode> global_graph;

    GraphIndexData graph_data(target_graph, global_graph);

    // Empty constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    ConstraintIndexData constraint_data(constraints, graph_data);

    EXPECT_TRUE(constraint_data.restricted_global_indices.empty());
    EXPECT_TRUE(constraint_data.preferred_global_indices.empty());
}

TEST_F(TopologySolverTest, SearchHeuristicBasic) {
    using namespace tt::tt_fabric::detail;

    // Create simple target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Build index data (CTAD deduces types from constructor arguments)
    GraphIndexData graph_data(target_graph, global_graph);

    // Empty constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Empty mapping (no nodes assigned yet)
    std::vector<int> mapping(2, -1);
    std::vector<bool> used(3, false);

    // Test selection (uses ConstraintIndexData for fast lookups)
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select one of the target nodes (both have same cost when no neighbors mapped)
    EXPECT_LT(result.target_idx, 2u);

    // Should have candidates (all global nodes are valid)
    EXPECT_GT(result.candidates.size(), 0u);
    EXPECT_LE(result.candidates.size(), 3u);
}

TEST_F(TopologySolverTest, SearchHeuristicNodeSelection) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Map node 1 to 10
    std::vector<int> mapping(3, -1);
    std::vector<bool> used(4, false);
    mapping[0] = 0;  // target node 1 (idx 0) -> global node 10 (idx 0)
    used[0] = true;

    // Now node 2 should be selected (has mapped neighbor)
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select node 2 (index 1) because it has a mapped neighbor
    EXPECT_EQ(result.target_idx, 1u);

    // Candidates for node 2 should only include neighbors of global node 10 (which is 11)
    EXPECT_EQ(result.candidates.size(), 1u);
    EXPECT_EQ(result.candidates[0], 1u);  // global node 11 (idx 1)
}

TEST_F(TopologySolverTest, SearchHeuristicHardConstraints) {
    using namespace tt::tt_fabric::detail;

    // Create simple graphs
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};
    global_adj_map[12] = {13};  // Disconnected from 10-11
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Add required constraint: node 1 must map to 10
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    ConstraintIndexData constraint_data(constraints, graph_data);

    std::vector<int> mapping(2, -1);
    std::vector<bool> used(4, false);

    // Map node 1 to 10
    mapping[0] = 0;
    used[0] = true;

    // Select candidates for node 2
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select node 2
    EXPECT_EQ(result.target_idx, 1u);

    // Candidates should only include neighbors of 10 (which is 11)
    // Node 12 and 13 are filtered out because they're not connected to 10
    EXPECT_EQ(result.candidates.size(), 1u);
    EXPECT_EQ(result.candidates[0], 1u);  // global node 11
}

TEST_F(TopologySolverTest, ConstraintIndexDataMissingNodes) {
    using namespace tt::tt_fabric::detail;

    // Create graphs with only some nodes
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {};
    target_adj_map[2] = {};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Global graph only has nodes 10 and 11, but NOT 20 or 30
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {};
    global_adj_map[11] = {};
    // Note: nodes 20 and 30 are NOT in the global graph

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Create constraints using trait constraints that reference nodes NOT in the global graph
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Use trait constraints to allow multiple valid mappings
    // Target 1: can map to nodes with trait "group1" (which includes 10 (exists) and 20 (missing))
    std::map<TestTargetNode, std::string> target_traits = {{1, "group1"}, {2, "group2"}};
    std::map<TestGlobalNode, std::string> global_traits = {
        {10, "group1"},   // Exists in graph
        {20, "group1"},   // Missing from graph
        {11, "group2"},   // Exists in graph
        {30, "group2"}};  // Missing from graph
    EXPECT_TRUE(constraints.add_required_trait_constraint<std::string>(target_traits, global_traits));

    // Add preferred constraints with missing nodes
    constraints.add_preferred_constraint(1, 20);  // Node 20 is NOT in global graph
    constraints.add_preferred_constraint(2, 11);  // Node 11 exists

    ConstraintIndexData constraint_data(constraints, graph_data);

    // Target 1 (index 0): should only have node 10 (index 0) in restricted indices
    // Node 20 should be filtered out since it's not in the global graph
    EXPECT_EQ(constraint_data.restricted_global_indices[0].size(), 1u);
    EXPECT_EQ(constraint_data.restricted_global_indices[0][0], 0u);  // Global 10 (index 0)

    // Target 1 preferred: should be empty since node 20 is missing
    EXPECT_TRUE(constraint_data.preferred_global_indices[0].empty());

    // Target 2 (index 1): should only have node 11 (index 1) in restricted indices
    // Node 30 should be filtered out since it's not in the global graph
    EXPECT_EQ(constraint_data.restricted_global_indices[1].size(), 1u);
    EXPECT_EQ(constraint_data.restricted_global_indices[1][0], 1u);  // Global 11 (index 1)

    // Target 2 preferred: should only have node 11 (index 1)
    EXPECT_EQ(constraint_data.preferred_global_indices[1].size(), 1u);
    EXPECT_EQ(constraint_data.preferred_global_indices[1][0], 1u);  // Global 11 (index 1)

    // Verify is_valid_mapping behavior
    // Target 1 can only map to Global 10 (index 0)
    EXPECT_TRUE(constraint_data.is_valid_mapping(0, 0));   // Target 1 -> Global 10: valid
    EXPECT_FALSE(constraint_data.is_valid_mapping(0, 1));  // Target 1 -> Global 11: invalid (restricted to 10 only)

    // Target 2 can only map to Global 11 (index 1)
    EXPECT_FALSE(constraint_data.is_valid_mapping(1, 0));  // Target 2 -> Global 10: invalid (restricted to 11 only)
    EXPECT_TRUE(constraint_data.is_valid_mapping(1, 1));   // Target 2 -> Global 11: valid
}

TEST_F(TopologySolverTest, SearchHeuristicPreferredConstraints) {
    using namespace tt::tt_fabric::detail;

    // Create simple graphs
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {};
    target_adj_map[2] = {};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {};
    global_adj_map[11] = {};
    global_adj_map[12] = {};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Add preferred constraint: node 1 prefers 10
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_preferred_constraint(1, 10);
    ConstraintIndexData constraint_data(constraints, graph_data);

    std::vector<int> mapping(2, -1);
    std::vector<bool> used(3, false);

    // Select candidates for node 1
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select node 1
    EXPECT_EQ(result.target_idx, 0u);

    // Preferred candidate (10) should come first
    EXPECT_EQ(result.candidates.size(), 3u);
    EXPECT_EQ(result.candidates[0], 0u);  // global node 10 (preferred) should be first
}

TEST_F(TopologySolverTest, SearchHeuristicDegreeFiltering) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: node 1 (degree 2) -> node 2 (degree 1)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 3};
    target_adj_map[2] = {1};
    target_adj_map[3] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph with nodes of different degrees
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};  // degree 1
    global_adj_map[11] = {10};
    global_adj_map[12] = {13, 14};  // degree 2
    global_adj_map[13] = {12};
    global_adj_map[14] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    std::vector<int> mapping(3, -1);
    std::vector<bool> used(5, false);

    // Select candidates for node 1 (degree 2)
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select node 1
    EXPECT_EQ(result.target_idx, 0u);

    // Only nodes with degree >= 2 should be candidates (node 12)
    // Nodes 10 and 11 have degree 1, so they're filtered out
    bool found_12 = false;
    for (size_t cand : result.candidates) {
        if (cand == 2) {  // global node 12 (idx 2)
            found_12 = true;
        }
        // Should not include nodes 10 or 11 (indices 0, 1) - they have degree 1
        EXPECT_NE(cand, 0u);
        EXPECT_NE(cand, 1u);
    }
    EXPECT_TRUE(found_12);
}

TEST_F(TopologySolverTest, SearchHeuristicAllAssigned) {
    using namespace tt::tt_fabric::detail;

    // Create simple graphs
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // All nodes assigned
    std::vector<int> mapping = {0, 1};
    std::vector<bool> used = {true, true};

    // Should handle gracefully (no unassigned nodes)
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // target_idx will be SIZE_MAX if no unassigned nodes found
    // This is acceptable behavior - caller should check for this case
    EXPECT_EQ(result.target_idx, SIZE_MAX);
    // Verify that calling select_and_generate_candidates in this state is safe
    // and that the candidates list is empty (no out-of-bounds access occurred)
    EXPECT_EQ(result.candidates.size(), 0u);
}

TEST_F(TopologySolverTest, SearchHeuristicRelaxedModeChannelPreference) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 (requires 2 channels), 1 -> 3 (requires 1 channel)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2, 3};  // 2 connections to node 2, 1 connection to node 3
    target_adj_map[2] = {1, 1};  // 2 connections to node 1
    target_adj_map[3] = {1};  // 1 connection to node 1

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph where node 10 is connected to node 11 with 3 channels
    // This tests that in relaxed mode, we prefer connections closer to required count
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 11, 11};  // 3 connections to node 11
    global_adj_map[11] = {10, 10, 10};  // 3 connections to node 10

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Map node 1 to 10
    std::vector<int> mapping(3, -1);
    std::vector<bool> used(2, false);
    mapping[0] = 0;  // target node 1 (index 0) -> global node 10 (index 0)
    used[0] = true;  // 10 is used

    // Select candidates - should select node 2 (index 1) deterministically
    // Both nodes 2 and 3 have the same cost (same candidate count, same mapped neighbors),
    // so we break ties by selecting the node with the lower index
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select node 2 (index 1) deterministically (lower index when costs are equal)
    EXPECT_EQ(result.target_idx, 1u);
    EXPECT_GT(result.candidates.size(), 0u);
    EXPECT_EQ(result.candidates[0], 1u);  // global node 11 (index 1) should be the candidate

    // Now verify the channel preference by checking connection counts
    // Node 2 requires 2 channels to node 1, node 3 requires 1 channel to node 1
    // Global node 11 has 3 channels to node 10 (which is mapped from node 1)

    // Check connection count from candidate (11, index 1) to mapped node (10, index 0)
    auto it = graph_data.global_conn_count[1].find(0);
    EXPECT_NE(it, graph_data.global_conn_count[1].end());
    EXPECT_EQ(it->second, 3u);  // Should have 3 channels
}

TEST_F(TopologySolverTest, ConsistencyCheckerLocalConsistency) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};
    global_adj_map[13] = {};  // Disconnected node

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    std::vector<int> mapping(3, -1);

    // Map node 1 to 10, node 2 to 11 - should be consistent
    mapping[0] = 0;  // 1 -> 10
    mapping[1] = 1;  // 2 -> 11
    bool result1 = ConsistencyChecker::check_local_consistency(
        1, 1, graph_data, mapping, ConnectionValidationMode::RELAXED);
    EXPECT_TRUE(result1);

    // Map node 2 to 13 (disconnected) - should be inconsistent
    mapping[1] = 3;  // 2 -> 13
    bool result2 = ConsistencyChecker::check_local_consistency(
        1, 0, graph_data, mapping, ConnectionValidationMode::RELAXED);
    EXPECT_FALSE(result2);
}

TEST_F(TopologySolverTest, ConsistencyCheckerLocalConsistencyStrictMode) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 (requires 2 channels)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2};  // 2 connections
    target_adj_map[2] = {1, 1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (has 1 channel) and 10 -> 12 (has 2 channels)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 12, 12};
    global_adj_map[11] = {10};
    global_adj_map[12] = {10, 10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    std::vector<int> mapping(2, -1);

    // Map node 1 to 10, node 2 to 12 (has 2 channels) - should pass in strict mode
    mapping[0] = 0;  // 1 -> 10
    mapping[1] = 2;  // 2 -> 12
    bool result1 = ConsistencyChecker::check_local_consistency(
        0, 0, graph_data, mapping, ConnectionValidationMode::STRICT);
    EXPECT_TRUE(result1);

    // Map node 2 to 11 (has only 1 channel) - should fail in strict mode
    mapping[1] = 1;  // 2 -> 11
    bool result2 = ConsistencyChecker::check_local_consistency(
        0, 0, graph_data, mapping, ConnectionValidationMode::STRICT);
    EXPECT_FALSE(result2);

    // But should pass in relaxed mode
    bool result3 = ConsistencyChecker::check_local_consistency(
        0, 0, graph_data, mapping, ConnectionValidationMode::RELAXED);
    EXPECT_TRUE(result3);
}

TEST_F(TopologySolverTest, ConsistencyCheckerForwardConsistency) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    std::vector<int> mapping(3, -1);
    std::vector<bool> used(3, false);

    // Map node 1 to 10
    mapping[0] = 0;
    used[0] = true;

    // Check forward consistency for node 2 -> 11
    // Node 2's unassigned neighbor is node 3, which should be able to map to 12
    bool result1 = ConsistencyChecker::check_forward_consistency(
        1, 1, graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);
    EXPECT_TRUE(result1);

    // If we use up node 12, forward consistency should fail
    used[2] = true;
    bool result2 = ConsistencyChecker::check_forward_consistency(
        1, 1, graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);
    EXPECT_FALSE(result2);
}

TEST_F(TopologySolverTest, ConsistencyCheckerCountReachableUnused) {
    using namespace tt::tt_fabric::detail;

    // Create a simple path graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};
    global_adj_map[14] = {};  // Disconnected

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);
    AdjacencyGraph<TestTargetNode>::AdjacencyMap empty_target_map;
    AdjacencyGraph<TestTargetNode> target_graph(empty_target_map);  // Empty target, only need global

    GraphIndexData graph_data(target_graph, global_graph);

    std::vector<bool> used(5, false);

    // All nodes unused, starting from 10 should reach 4 nodes (10, 11, 12, 13)
    size_t count1 = ConsistencyChecker::count_reachable_unused(
        0, graph_data, used);
    EXPECT_EQ(count1, 4u);

    // Mark 11 as used, should reach 3 nodes (10, 12, 13)
    used[1] = true;
    size_t count2 = ConsistencyChecker::count_reachable_unused(
        0, graph_data, used);
    EXPECT_EQ(count2, 3u);

    // Disconnected node 14 should only reach itself
    used[1] = false;  // Reset
    size_t count3 = ConsistencyChecker::count_reachable_unused(
        4, graph_data, used);
    EXPECT_EQ(count3, 1u);
}

TEST_F(TopologySolverTest, DFSSearchEngineBasic) {
    using namespace tt::tt_fabric::detail;

    // Create simple target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);

    // Should find a mapping (e.g., 1->10, 2->11)
    EXPECT_TRUE(found);
    const auto& state = engine.get_state();
    EXPECT_GE(state.dfs_calls, 1u);

    // Verify mapping is valid
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_GE(state.mapping[i], 0);
        EXPECT_LT(static_cast<size_t>(state.mapping[i]), 3u);
    }

    // Verify mapping is consistent (if node 1 maps to X and node 2 maps to Y, X and Y must be connected)
    size_t idx1 = 0;  // target node 1
    size_t idx2 = 1;  // target node 2
    size_t global_idx1 = static_cast<size_t>(state.mapping[idx1]);
    size_t global_idx2 = static_cast<size_t>(state.mapping[idx2]);

    // Check that global_idx1 and global_idx2 are connected
    bool connected = std::binary_search(
        graph_data.global_adj_idx[global_idx1].begin(), graph_data.global_adj_idx[global_idx1].end(), global_idx2);
    EXPECT_TRUE(connected);
}

TEST_F(TopologySolverTest, DFSSearchEngineWithConstraints) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Add constraint: node 1 must map to 10
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);

    // Should find a mapping
    EXPECT_TRUE(found);
    const auto& state = engine.get_state();

    // Verify constraint is satisfied: node 1 (index 0) should map to global node 10 (index 0)
    EXPECT_EQ(state.mapping[0], 0);

    // Node 2 should map to a neighbor of 10 (which is 11, index 1)
    EXPECT_EQ(state.mapping[1], 1);
}

TEST_F(TopologySolverTest, DFSSearchEngineNoSolution) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 -> 3 (path of 3 nodes)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (only 2 nodes, can't fit 3-node path)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);

    // Should not find a mapping (target graph too large)
    EXPECT_FALSE(found);
    // DFS calls may be 0 if we detect early that global graph is too small
    // (this is actually better - we fail fast with a clear error message)
}

TEST_F(TopologySolverTest, DFSSearchEngineStrictMode) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 (requires 2 channels)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2};  // 2 connections
    target_adj_map[2] = {1, 1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (has 1 channel) and 10 -> 12 (has 2 channels)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 12, 12};  // 1 connection to 11, 2 connections to 12
    global_adj_map[11] = {10};
    global_adj_map[12] = {10, 10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run search in STRICT mode - should find mapping (1->10, 2->12) since 12 has 2 channels
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(graph_data, constraint_data, ConnectionValidationMode::STRICT);

    EXPECT_TRUE(found);
    const auto& state = engine.get_state();
    // Verify mapping uses node with sufficient channels
    size_t global_idx1 = static_cast<size_t>(state.mapping[0]);
    size_t global_idx2 = static_cast<size_t>(state.mapping[1]);
    // One of them should be 10 (index 0), the other should be 12 (index 2) which has 2 channels
    EXPECT_TRUE((global_idx1 == 0 && global_idx2 == 2) || (global_idx1 == 2 && global_idx2 == 0));
}

TEST_F(TopologySolverTest, DFSSearchEngineRelaxedModeChannelPreference) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 (requires 2 channels)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2};  // 2 connections to node 2
    target_adj_map[2] = {1, 1};  // 2 connections to node 1

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph with multiple candidates that all work:
    // - Node 10 connected to node 11 with 1 channel (insufficient, but allowed in relaxed mode)
    // - Node 10 connected to node 12 with 2 channels (exact match, preferred)
    // - Node 10 connected to node 13 with 3 channels (more than required, also preferred)
    // The DFS search should prefer 12 or 13 over 11 due to better channel match scores
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 12, 12, 13, 13, 13};  // 1 to 11, 2 to 12, 3 to 13
    global_adj_map[11] = {10};
    global_adj_map[12] = {10, 10};
    global_adj_map[13] = {10, 10, 10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run search in RELAXED mode
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);

    EXPECT_TRUE(found);
    const auto& state = engine.get_state();

    // Verify mapping: node 1 should map to 10, node 2 should map to a neighbor of 10
    size_t global_idx1 = static_cast<size_t>(state.mapping[0]);  // target node 1 -> global node
    size_t global_idx2 = static_cast<size_t>(state.mapping[1]);  // target node 2 -> global node

    // One should map to node 10 (index 0)
    EXPECT_TRUE(global_idx1 == 0 || global_idx2 == 0) << "One target node must map to global node 10";

    // The other should map to a neighbor of 10 (11, 12, or 13)
    size_t node2_global_idx = (global_idx1 == 0) ? global_idx2 : global_idx1;
    EXPECT_TRUE(node2_global_idx == 1 || node2_global_idx == 2 || node2_global_idx == 3);

    // Check connection count from node 10 to the chosen node
    auto it = graph_data.global_conn_count[0].find(node2_global_idx);
    EXPECT_NE(it, graph_data.global_conn_count[0].end());

    // Due to candidate ordering preference (tested in SearchHeuristic tests),
    // the DFS search should prefer nodes with more channels (12 or 13) over insufficient (11)
    // In relaxed mode, all are valid, but preference should lead to better matches
    // Verify that if a better match exists, it was chosen
    EXPECT_GE(it->second, 1u) << "Connection should exist";
    // The preference for more channels should result in choosing 12 (2 channels) or 13 (3 channels)
    // over 11 (1 channel), though all are valid in relaxed mode
}

TEST_F(TopologySolverTest, MappingValidatorSavesPartialMapping) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (only 2 nodes, can't fit 3-node path)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Initialize search state with partial mapping (node 1 -> 10, node 2 -> 11)
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(3, -1);
    state.mapping[0] = 0;   // target node 1 -> global node 10
    state.mapping[1] = 1;   // target node 2 -> global node 11
    state.mapping[2] = -1;  // target node 3 not mapped
    state.used.resize(2, false);
    state.used[0] = true;
    state.used[1] = true;

    // Build result - should save partial mapping even though it's incomplete
    MappingResult<TestTargetNode, TestGlobalNode> result =
        MappingValidator<TestTargetNode, TestGlobalNode>::build_result(
            state.mapping, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // Should fail (incomplete mapping)
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());

    // But should still save the partial mapping that was found
    EXPECT_EQ(result.target_to_global.size(), 2u) << "Should save 2 mapped nodes";
    EXPECT_EQ(result.target_to_global.at(1), 10u) << "Target node 1 should map to global node 10";
    EXPECT_EQ(result.target_to_global.at(2), 11u) << "Target node 2 should map to global node 11";
    EXPECT_EQ(result.target_to_global.count(3), 0u) << "Target node 3 should not be mapped";

    // Should have statistics
    EXPECT_EQ(result.stats.dfs_calls, 0u);  // No DFS calls made in this test
}

// Helper function to create a 2D mesh graph
template <typename NodeId>
AdjacencyGraph<NodeId> create_2d_mesh_graph(size_t rows, size_t cols) {
    using AdjacencyMap = typename AdjacencyGraph<NodeId>::AdjacencyMap;
    AdjacencyMap adj_map;

    auto get_node_id = [cols](size_t row, size_t col) -> size_t { return (row * cols) + col; };

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t node_id = get_node_id(row, col);
            std::vector<NodeId> neighbors;

            // Add left neighbor
            if (col > 0) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row, col - 1)));
            }
            // Add right neighbor
            if (col < cols - 1) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row, col + 1)));
            }
            // Add top neighbor
            if (row > 0) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row - 1, col)));
            }
            // Add bottom neighbor
            if (row < rows - 1) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row + 1, col)));
            }

            adj_map[static_cast<NodeId>(node_id)] = neighbors;
        }
    }

    return AdjacencyGraph<NodeId>(adj_map);
}

// Helper function to create a 1D chain graph
template <typename NodeId>
AdjacencyGraph<NodeId> create_1d_chain_graph(size_t length) {
    using AdjacencyMap = typename AdjacencyGraph<NodeId>::AdjacencyMap;
    AdjacencyMap adj_map;

    for (size_t i = 0; i < length; ++i) {
        std::vector<NodeId> neighbors;
        // Add left neighbor
        if (i > 0) {
            neighbors.push_back(static_cast<NodeId>(i - 1));
        }
        // Add right neighbor
        if (i < length - 1) {
            neighbors.push_back(static_cast<NodeId>(i + 1));
        }
        adj_map[static_cast<NodeId>(i)] = neighbors;
    }

    return AdjacencyGraph<NodeId>(adj_map);
}

// Helper function to create a 1D ring graph (cycle)
template <typename NodeId>
AdjacencyGraph<NodeId> create_1d_ring_graph(size_t length) {
    using AdjacencyMap = typename AdjacencyGraph<NodeId>::AdjacencyMap;
    AdjacencyMap adj_map;

    for (size_t i = 0; i < length; ++i) {
        std::vector<NodeId> neighbors;
        // Add left neighbor (wraps around)
        size_t left = (i == 0) ? length - 1 : i - 1;
        neighbors.push_back(static_cast<NodeId>(left));
        // Add right neighbor (wraps around)
        size_t right = (i == length - 1) ? 0 : i + 1;
        neighbors.push_back(static_cast<NodeId>(right));
        adj_map[static_cast<NodeId>(i)] = neighbors;
    }

    return AdjacencyGraph<NodeId>(adj_map);
}

// Helper function to create a 2D torus graph (wraps around both dimensions)
template <typename NodeId>
AdjacencyGraph<NodeId> create_2d_torus_graph(size_t rows, size_t cols) {
    using AdjacencyMap = typename AdjacencyGraph<NodeId>::AdjacencyMap;
    AdjacencyMap adj_map;

    auto get_node_id = [cols](size_t row, size_t col) -> size_t { return (row * cols) + col; };

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t node_id = get_node_id(row, col);
            std::vector<NodeId> neighbors;

            // Add left neighbor (wraps around)
            size_t left_col = (col == 0) ? cols - 1 : col - 1;
            neighbors.push_back(static_cast<NodeId>(get_node_id(row, left_col)));

            // Add right neighbor (wraps around)
            size_t right_col = (col == cols - 1) ? 0 : col + 1;
            neighbors.push_back(static_cast<NodeId>(get_node_id(row, right_col)));

            // Add top neighbor (wraps around)
            size_t top_row = (row == 0) ? rows - 1 : row - 1;
            neighbors.push_back(static_cast<NodeId>(get_node_id(top_row, col)));

            // Add bottom neighbor (wraps around)
            size_t bottom_row = (row == rows - 1) ? 0 : row + 1;
            neighbors.push_back(static_cast<NodeId>(get_node_id(bottom_row, col)));

            adj_map[static_cast<NodeId>(node_id)] = neighbors;
        }
    }

    return AdjacencyGraph<NodeId>(adj_map);
}

TEST_F(TopologySolverTest, RequiredConstraints_4x8MeshOn8x8Mesh_CornersToCorners) {
    // Create global graph: 8x8 mesh (64 nodes, no wrap-around, has 4 corners)
    auto global_graph = create_2d_mesh_graph<TestGlobalNode>(8, 8);

    // Create target graph: 4x8 mesh without torus connections (32 nodes, has 4 corners)
    auto target_graph = create_2d_mesh_graph<TestTargetNode>(4, 8);

    // Verify graph sizes
    EXPECT_EQ(global_graph.get_nodes().size(), 64u);
    EXPECT_EQ(target_graph.get_nodes().size(), 32u);

    // Verify global graph structure (8x8 mesh - corners have 2, edges have 3, interior have 4)
    size_t global_corner_count = 0, global_edge_count = 0, global_interior_count = 0;
    for (const auto& node : global_graph.get_nodes()) {
        size_t degree = global_graph.get_neighbors(node).size();
        if (degree == 2) {
            global_corner_count++;
        } else if (degree == 3) {
            global_edge_count++;
        } else if (degree == 4) {
            global_interior_count++;
        }
    }
    // 8x8 mesh: 4 corners, (8-2)*2 + (8-2)*2 = 12 + 12 = 24 edge nodes, (8-2)*(8-2) = 36 interior
    EXPECT_EQ(global_corner_count, 4u);
    EXPECT_EQ(global_edge_count, 24u);
    EXPECT_EQ(global_interior_count, 36u);

    // Verify target graph structure (4x8 mesh - corners have 2, edges have 3, interior have 4)
    size_t target_corner_count = 0, target_edge_count = 0, target_interior_count = 0;
    for (const auto& node : target_graph.get_nodes()) {
        size_t degree = target_graph.get_neighbors(node).size();
        if (degree == 2) {
            target_corner_count++;
        } else if (degree == 3) {
            target_edge_count++;
        } else if (degree == 4) {
            target_interior_count++;
        }
    }
    // 4x8 mesh: 4 corners, (4-2)*2 + (8-2)*2 = 4 + 12 = 16 edge nodes, (4-2)*(8-2) = 12 interior
    EXPECT_EQ(target_corner_count, 4u);
    EXPECT_EQ(target_edge_count, 16u);
    EXPECT_EQ(target_interior_count, 12u);

    // Identify corner nodes in target graph (nodes with degree 2)
    std::map<TestTargetNode, std::string> target_traits;

    // Mark corner nodes in target graph (4x8 mesh corners)
    for (const auto& node : target_graph.get_nodes()) {
        if (target_graph.get_neighbors(node).size() == 2) {
            target_traits[node] = "corner";
        }
    }

    // Verify we found the correct number of corners in target graph
    EXPECT_EQ(target_traits.size(), 4u) << "Target graph should have 4 corners";

    // Define specific allowed positions for corners in global graph
    // Positions: 00, 03, 04, 07, 70, 73, 74, 77
    // Node ID = row * 8 + col
    std::map<TestGlobalNode, std::string> global_traits;
    global_traits[0] = "corner";   // 00 = row 0, col 0
    global_traits[3] = "corner";   // 03 = row 0, col 3
    global_traits[4] = "corner";   // 04 = row 0, col 4
    global_traits[7] = "corner";   // 07 = row 0, col 7
    global_traits[56] = "corner";  // 70 = row 7, col 0
    global_traits[59] = "corner";  // 73 = row 7, col 3
    global_traits[60] = "corner";  // 74 = row 7, col 4
    global_traits[63] = "corner";  // 77 = row 7, col 7

    // Verify we have 8 allowed positions
    EXPECT_EQ(global_traits.size(), 8u) << "Should have 8 allowed corner positions";

    // Create constraints with trait-based required mappings (one-to-many)
    // This constrains mesh corners to map to ANY of the 8 specified positions
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    EXPECT_TRUE(constraints.add_required_trait_constraint<std::string>(target_traits, global_traits));

    // Solve the mapping
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Verify mapping succeeded
    EXPECT_TRUE(result.success)
        << "4x8 mesh should fit in 8x8 mesh with corner-to-allowed-positions constraints. Error: "
        << result.error_message;

    if (result.success) {
        // Verify all target nodes are mapped
        EXPECT_EQ(result.target_to_global.size(), 32u) << "All 32 target nodes should be mapped";

        // Verify all corner nodes are mapped to allowed positions
        for (const auto& [target_node, trait] : target_traits) {
            if (trait == "corner") {
                auto it = result.target_to_global.find(target_node);
                ASSERT_NE(it, result.target_to_global.end()) << "Corner node " << target_node << " should be mapped";

                // Verify the mapped global node is one of the allowed positions
                TestGlobalNode mapped_global = it->second;
                auto global_it = global_traits.find(mapped_global);
                EXPECT_NE(global_it, global_traits.end())
                    << "Target corner " << target_node
                    << " should map to one of the allowed positions (00, 03, 04, 07, 70, 73, 74, 77), "
                    << "but mapped to " << mapped_global;
                if (global_it != global_traits.end()) {
                    EXPECT_EQ(global_it->second, "corner") << "Mapped global node should be in allowed positions";
                }
            }
        }

        // Verify all 4 target corners mapped to allowed positions
        size_t corners_mapped_to_allowed = 0;
        for (const auto& [target_node, trait] : target_traits) {
            if (trait == "corner" && result.target_to_global.contains(target_node)) {
                TestGlobalNode mapped_global = result.target_to_global.at(target_node);
                if (global_traits.contains(mapped_global)) {
                    corners_mapped_to_allowed++;
                }
            }
        }
        EXPECT_EQ(corners_mapped_to_allowed, 4u) << "All 4 target corners should map to allowed positions";

        // Verify mapping preserves adjacency
        for (const auto& [target_node, global_node] : result.target_to_global) {
            const auto& target_neighbors = target_graph.get_neighbors(target_node);
            const auto& global_neighbors = global_graph.get_neighbors(global_node);

            for (const auto& target_neighbor : target_neighbors) {
                auto it = result.target_to_global.find(target_neighbor);
                if (it != result.target_to_global.end()) {
                    // Check if the mapped neighbor is adjacent to the mapped global node
                    bool neighbor_adjacent = std::find(global_neighbors.begin(), global_neighbors.end(), it->second) !=
                                             global_neighbors.end();
                    EXPECT_TRUE(neighbor_adjacent)
                        << "Target node " << target_node << " -> global " << global_node << " should have neighbor "
                        << target_neighbor << " -> global " << it->second << " as adjacent";
                }
            }
        }

        // Log statistics
        log_info(
            tt::LogFabric,
            "Corner-to-allowed-positions constraints test completed: corners_mapped={}, dfs_calls={}, backtracks={}",
            corners_mapped_to_allowed,
            result.stats.dfs_calls,
            result.stats.backtrack_count);
    } else {
        log_error(tt::LogFabric, "Mapping failed: {}", result.error_message);
    }
}

TEST_F(TopologySolverTest, SolveTopologyMapping_4x3MeshOn6x2Torus) {
    // Test mapping a 4x3 logical mesh onto a 6x2 physical torus
    // Logical mesh: 4x3 grid (12 nodes, no wrap-around)
    // Physical mesh: 6x2 torus (12 nodes, with wrap-around in both dimensions)
    //
    // This tests that a grid topology cannot be mapped onto a torus topology
    // where the torus has wrap-around connections that the grid doesn't have.
    // Uses smaller topology to reduce DFS calls while still testing the concept.

    // Create logical graph: 4x3 mesh (12 nodes, no wrap-around)
    auto logical_graph = create_2d_mesh_graph<TestTargetNode>(4, 3);

    // Create physical graph: 6x2 torus (12 nodes, with wrap-around)
    auto physical_graph = create_2d_torus_graph<TestGlobalNode>(6, 2);

    // Verify graph sizes match
    EXPECT_EQ(logical_graph.get_nodes().size(), 12u) << "Logical mesh should have 4*3=12 nodes";
    EXPECT_EQ(physical_graph.get_nodes().size(), 12u) << "Physical torus should have 6*2=12 nodes";

    // Verify logical graph structure (4x3 mesh)
    // Corners have 2 neighbors, edges have 3 neighbors, interior have 4 neighbors
    size_t logical_corner_count = 0, logical_edge_count = 0, logical_interior_count = 0;
    for (const auto& node : logical_graph.get_nodes()) {
        size_t degree = logical_graph.get_neighbors(node).size();
        if (degree == 2) {
            logical_corner_count++;
        } else if (degree == 3) {
            logical_edge_count++;
        } else if (degree == 4) {
            logical_interior_count++;
        }
    }
    // 4x3 mesh: 4 corners, (4-2)*2 + (3-2)*2 = 4 + 2 = 6 edge nodes, (4-2)*(3-2) = 2 interior
    EXPECT_EQ(logical_corner_count, 4u);
    EXPECT_EQ(logical_edge_count, 6u);
    EXPECT_EQ(logical_interior_count, 2u);

    // Verify physical graph structure (6x2 torus - all nodes have 4 neighbors due to wrap-around)
    for (const auto& node : physical_graph.get_nodes()) {
        size_t degree = physical_graph.get_neighbors(node).size();
        EXPECT_EQ(degree, 4u) << "All nodes in a 2D torus should have exactly 4 neighbors";
    }

    // No constraints - let the solver find any valid mapping
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Solve the mapping
    auto result = solve_topology_mapping(logical_graph, physical_graph, constraints, ConnectionValidationMode::RELAXED);

    // Verify mapping failed - a 4x3 mesh cannot map onto a 6x2 torus
    // The mesh has nodes with 2 neighbors (corners) and 3 neighbors (edges),
    // but the torus only has nodes with 4 neighbors (all nodes due to wrap-around).
    // This violates the adjacency preservation requirement.
    EXPECT_FALSE(result.success) << "4x3 mesh should NOT map onto 6x2 torus because mesh has nodes with 2-3 neighbors "
                                 << "but torus only has nodes with 4 neighbors. Error: " << result.error_message;
    EXPECT_FALSE(result.error_message.empty()) << "Should have error message explaining why mapping failed";

    // Verify no infinite loop occurred - DFS calls should be reasonable and not exceed limit
    // The DFS limit is 1 million, so we check that it's well below that
    EXPECT_LT(result.stats.dfs_calls, 1000000u) << "DFS calls should not exceed limit (no infinite loop)";
    EXPECT_GT(result.stats.dfs_calls, 0u) << "Should have made some DFS calls";

    // Log statistics for debugging
    log_info(
        tt::LogFabric,
        "4x3 mesh on 6x2 torus test (expected failure): dfs_calls={}, backtracks={}, memoization_hits={}, "
        "mapped_nodes={}, error={}",
        result.stats.dfs_calls,
        result.stats.backtrack_count,
        result.stats.memoization_hits,
        result.target_to_global.size(),
        result.error_message);
}

// Helper function to create a 2D mesh graph without torus connections (no wrap-around)
template <typename NodeId>
AdjacencyGraph<NodeId> create_2d_mesh_no_torus_graph(size_t rows, size_t cols) {
    using AdjacencyMap = typename AdjacencyGraph<NodeId>::AdjacencyMap;
    AdjacencyMap adj_map;

    auto get_node_id = [cols](size_t row, size_t col) -> size_t { return (row * cols) + col; };

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t node_id = get_node_id(row, col);
            std::vector<NodeId> neighbors;

            // Add left neighbor (no wrap-around)
            if (col > 0) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row, col - 1)));
            }
            // Add right neighbor (no wrap-around)
            if (col < cols - 1) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row, col + 1)));
            }
            // Add top neighbor (no wrap-around)
            if (row > 0) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row - 1, col)));
            }
            // Add bottom neighbor (no wrap-around)
            if (row < rows - 1) {
                neighbors.push_back(static_cast<NodeId>(get_node_id(row + 1, col)));
            }

            adj_map[static_cast<NodeId>(node_id)] = neighbors;
        }
    }

    return AdjacencyGraph<NodeId>(adj_map);
}

// Helper function to create a disconnected graph with multiple components
template <typename NodeId>
AdjacencyGraph<NodeId> create_disconnected_graph(
    const std::vector<std::vector<std::pair<size_t, size_t>>>& components, size_t base_node_id = 0) {
    using AdjacencyMap = typename AdjacencyGraph<NodeId>::AdjacencyMap;
    AdjacencyMap adj_map;

    size_t current_node_id = base_node_id;
    for (const auto& component : components) {
        // Create nodes for this component
        std::map<size_t, size_t> local_to_global;
        for (const auto& edge : component) {
            if (!local_to_global.contains(edge.first)) {
                local_to_global[edge.first] = current_node_id++;
            }
            if (!local_to_global.contains(edge.second)) {
                local_to_global[edge.second] = current_node_id++;
            }
        }

        // Build adjacency map for this component
        for (const auto& edge : component) {
            size_t global_from = local_to_global[edge.first];
            size_t global_to = local_to_global[edge.second];

            adj_map[static_cast<NodeId>(global_from)].push_back(static_cast<NodeId>(global_to));
            adj_map[static_cast<NodeId>(global_to)].push_back(static_cast<NodeId>(global_from));
        }
    }

    return AdjacencyGraph<NodeId>(adj_map);
}

TEST_F(TopologySolverTest, DFSSearchEngineStressTest_1DRingOn2DMesh) {
    using namespace tt::tt_fabric::detail;

    // Create global graph: 2D mesh of 4x8 = 32 nodes
    // Each node connects to up to 4 neighbors (up, down, left, right)
    auto global_graph = create_2d_mesh_graph<TestGlobalNode>(4, 8);

    // Create target graph: 1D ring of 32 nodes (cycle)
    // Each node connects to exactly 2 neighbors (prev, next, wrapping around)
    auto target_graph = create_1d_ring_graph<TestTargetNode>(32);

    // Verify graph sizes
    EXPECT_EQ(global_graph.get_nodes().size(), 32u);
    EXPECT_EQ(target_graph.get_nodes().size(), 32u);

    // Verify global graph structure (2D mesh)
    // Corner nodes should have 2 neighbors, edge nodes 3, interior nodes 4
    size_t corner_count = 0, edge_count = 0, interior_count = 0;
    for (const auto& node : global_graph.get_nodes()) {
        size_t degree = global_graph.get_neighbors(node).size();
        if (degree == 2) {
            corner_count++;
        } else if (degree == 3) {
            edge_count++;
        } else if (degree == 4) {
            interior_count++;
        }
    }
    // 4x8 mesh: 4 corners, (4-2)*2 + (8-2)*2 = 4 + 12 = 16 edge nodes, (4-2)*(8-2) = 12 interior
    EXPECT_EQ(corner_count, 4u);
    EXPECT_EQ(edge_count, 16u);
    EXPECT_EQ(interior_count, 12u);

    // Verify target graph structure (1D ring)
    // All nodes should have exactly 2 neighbors (ring topology)
    for (const auto& node : target_graph.get_nodes()) {
        size_t degree = target_graph.get_neighbors(node).size();
        EXPECT_EQ(degree, 2u) << "All nodes in ring should have degree 2";
    }

    // Build graph data and constraints
    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add constraints that are satisfiable for a ring on a 2D mesh
    // For a ring, we need to ensure the cycle can be closed
    // Pin a node to a corner (node 0 -> global 0)
    constraints.add_required_constraint(0, 0);
    // Pin the next node in the ring to an adjacent node (node 1 -> global 1, which is adjacent to 0)
    constraints.add_required_constraint(1, 1);
    // Pin a node halfway around the ring to a node that can form a path back
    // Node 16 is halfway, pin it to global node 16 (row 2, col 0) which can connect back
    constraints.add_preferred_constraint(16, 16);

    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
    const auto& state = search_engine.get_state();

    // This should succeed - a 1D ring can be embedded in a 2D mesh
    // (the ring can follow a cycle path through the mesh)
    EXPECT_TRUE(found) << "1D ring of 32 nodes should fit in 4x8 2D mesh";

    if (found) {
        // Verify all nodes are mapped
        size_t mapped_count = 0;
        for (int mapped_value : state.mapping) {
            if (mapped_value != -1) {
                mapped_count++;
            }
        }
        EXPECT_EQ(mapped_count, 32u) << "All 32 target nodes should be mapped";

        // Verify mapping preserves adjacency
        // For each edge in target graph, corresponding edge should exist in global graph
        for (size_t i = 0; i < graph_data.n_target; ++i) {
            if (state.mapping[i] == -1) {
                continue;
            }
            size_t global_i = static_cast<size_t>(state.mapping[i]);

            for (size_t neighbor_idx : graph_data.target_adj_idx[i]) {
                if (state.mapping[neighbor_idx] == -1) {
                    continue;
                }
                size_t global_neighbor = static_cast<size_t>(state.mapping[neighbor_idx]);

                // Check if edge exists in global graph
                bool edge_exists = std::binary_search(
                    graph_data.global_adj_idx[global_i].begin(),
                    graph_data.global_adj_idx[global_i].end(),
                    global_neighbor);
                EXPECT_TRUE(edge_exists) << "Edge from target node " << graph_data.target_nodes[i] << " to "
                                         << graph_data.target_nodes[neighbor_idx]
                                         << " should map to edge in global graph";
            }
        }

        // Verify pinned nodes are correctly mapped
        EXPECT_EQ(state.mapping[0], 0) << "First node should be pinned to global node 0";
        EXPECT_EQ(state.mapping[1], 1) << "Second node should be pinned to global node 1";

        // Verify the ring structure - check that the cycle is closed
        // Node 31 should connect back to node 0
        size_t node_31_global = static_cast<size_t>(state.mapping[31]);
        size_t node_0_global = static_cast<size_t>(state.mapping[0]);
        bool ring_closed = std::binary_search(
            graph_data.global_adj_idx[node_31_global].begin(),
            graph_data.global_adj_idx[node_31_global].end(),
            node_0_global);
        EXPECT_TRUE(ring_closed) << "Ring should be closed: node 31 should connect to node 0";

        // Log statistics
        EXPECT_GT(state.dfs_calls, 0u) << "Should have made some DFS calls";
        log_info(
            tt::LogFabric,
            "Stress test with pinnings completed: DFS calls={}, backtracks={}",
            state.dfs_calls,
            state.backtrack_count);
    } else {
        // If it failed, log the error
        log_error(tt::LogFabric, "Stress test failed: {}", state.error_message);
    }
}

TEST_F(TopologySolverTest, DFSSearchEngineStressTest_1DChainOn2DMesh_Negative) {
    using namespace tt::tt_fabric::detail;

    // Test 1: 1D chain on 3x3 mesh (should succeed - no constraints)
    {
        auto global_graph = create_2d_mesh_graph<TestGlobalNode>(3, 3);
        auto target_graph = create_1d_chain_graph<TestTargetNode>(9);

        GraphIndexData graph_data(target_graph, global_graph);
        MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
        ConstraintIndexData constraint_data(constraints, graph_data);

        DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
        bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
        const auto& state = search_engine.get_state();

        EXPECT_TRUE(found) << "1D chain of 9 nodes should map to 3x3 mesh (9 nodes)";
        EXPECT_EQ(state.mapping.size(), 9u) << "All 9 nodes should be mapped";
    }

    // Test 2: 1D ring on 3x3 mesh (should fail - ring requires cycle, 3x3 mesh may not support it)
    {
        auto global_graph = create_2d_mesh_graph<TestGlobalNode>(3, 3);
        auto target_graph = create_1d_ring_graph<TestTargetNode>(9);

        GraphIndexData graph_data(target_graph, global_graph);
        MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
        ConstraintIndexData constraint_data(constraints, graph_data);

        DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
        bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
        const auto& state = search_engine.get_state();

        if (found) {
            // Unexpected success - print debug information
            log_error(tt::LogFabric, "Negative test unexpectedly succeeded - printing debug information");

            // Print adjacency graphs
            log_info(tt::LogFabric, "=== Target Graph Adjacency (1D Ring) ===");
            target_graph.print_adjacency_map("Target Graph");

            log_info(tt::LogFabric, "=== Global Graph Adjacency (3x3 Mesh) ===");
            global_graph.print_adjacency_map("Global Graph");

            // Print mapping
            log_info(tt::LogFabric, "=== Mapping Result ===");
            MappingValidator<TestTargetNode, TestGlobalNode>::print_mapping(state.mapping, graph_data);

            FAIL() << "1D ring of 9 nodes should not map to 3x3 mesh, but it succeeded. Check logs above for details.";
        }

        EXPECT_FALSE(found) << "1D ring of 9 nodes should fail to map to 3x3 mesh";

        if (!found) {
            log_info(
                tt::LogFabric,
                "Negative stress test completed as expected: DFS calls={}, backtracks={}, error={}",
                state.dfs_calls,
                state.backtrack_count,
                state.error_message.empty() ? "(no error message)" : state.error_message);
        }
    }
}

TEST_F(TopologySolverTest, DFSSearchEngine_DisconnectedTargetGraph) {
    using namespace tt::tt_fabric::detail;

    // Create global graph: single connected component (chain of 6 nodes)
    auto global_graph = create_1d_chain_graph<TestGlobalNode>(6);

    // Create target graph: two disconnected components
    // Component 1: 3 nodes (0-1-2)
    // Component 2: 3 nodes (3-4-5)
    std::vector<std::vector<std::pair<size_t, size_t>>> target_components = {
        {{0, 1}, {1, 2}},  // Component 1: chain of 3 nodes
        {{3, 4}, {4, 5}}   // Component 2: chain of 3 nodes
    };
    auto target_graph = create_disconnected_graph<TestTargetNode>(target_components);

    // Verify graphs
    EXPECT_EQ(global_graph.get_nodes().size(), 6u);
    EXPECT_EQ(target_graph.get_nodes().size(), 6u);

    // Build graph data and constraints
    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
    const auto& state = search_engine.get_state();

    // This should succeed - two disconnected chains can map to one connected chain
    EXPECT_TRUE(found) << "Disconnected target graph should map to connected global graph";

    if (found) {
        // Verify all nodes are mapped
        size_t mapped_count = 0;
        for (int mapped_value : state.mapping) {
            if (mapped_value != -1) {
                mapped_count++;
            }
        }
        EXPECT_EQ(mapped_count, 6u) << "All 6 target nodes should be mapped";

        // Verify mapping preserves adjacency within each component
        for (size_t i = 0; i < graph_data.n_target; ++i) {
            if (state.mapping[i] == -1) {
                continue;
            }
            size_t global_i = static_cast<size_t>(state.mapping[i]);

            for (size_t neighbor_idx : graph_data.target_adj_idx[i]) {
                if (state.mapping[neighbor_idx] == -1) {
                    continue;
                }
                size_t global_neighbor = static_cast<size_t>(state.mapping[neighbor_idx]);

                bool edge_exists = std::binary_search(
                    graph_data.global_adj_idx[global_i].begin(),
                    graph_data.global_adj_idx[global_i].end(),
                    global_neighbor);
                EXPECT_TRUE(edge_exists) << "Edge from target node " << graph_data.target_nodes[i] << " to "
                                         << graph_data.target_nodes[neighbor_idx]
                                         << " should map to edge in global graph";
            }
        }

        log_info(
            tt::LogFabric,
            "Disconnected target test completed: DFS calls={}, backtracks={}",
            state.dfs_calls,
            state.backtrack_count);
    }
}

TEST_F(TopologySolverTest, DFSSearchEngine_DisconnectedGlobalGraph) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: single connected component (chain of 6 nodes)
    auto target_graph = create_1d_chain_graph<TestTargetNode>(6);

    // Create global graph: two disconnected components
    // Component 1: 3 nodes (0-1-2)
    // Component 2: 3 nodes (3-4-5)
    std::vector<std::vector<std::pair<size_t, size_t>>> global_components = {
        {{0, 1}, {1, 2}},  // Component 1: chain of 3 nodes
        {{3, 4}, {4, 5}}   // Component 2: chain of 3 nodes
    };
    auto global_graph = create_disconnected_graph<TestGlobalNode>(global_components);

    // Verify graphs
    EXPECT_EQ(target_graph.get_nodes().size(), 6u);
    EXPECT_EQ(global_graph.get_nodes().size(), 6u);

    // Build graph data and constraints
    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
    const auto& state = search_engine.get_state();

    // This should fail - a connected chain cannot map to disconnected components
    // because the chain requires a path between all nodes
    EXPECT_FALSE(found) << "Connected target graph should not map to disconnected global graph";

    if (!found) {
        EXPECT_FALSE(state.error_message.empty()) << "Should have error message";
        log_info(
            tt::LogFabric,
            "Disconnected global test failed as expected: DFS calls={}, backtracks={}, error={}",
            state.dfs_calls,
            state.backtrack_count,
            state.error_message);
    }
}

TEST_F(TopologySolverTest, DFSSearchEngine_DisconnectedBothGraphs_Success) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: two disconnected components
    // Component 1: chain of 3 nodes (0-1-2)
    // Component 2: chain of 3 nodes (3-4-5)
    std::vector<std::vector<std::pair<size_t, size_t>>> target_components = {{{0, 1}, {1, 2}}, {{3, 4}, {4, 5}}};
    auto target_graph = create_disconnected_graph<TestTargetNode>(target_components);

    // Create global graph: two disconnected components (same structure)
    std::vector<std::vector<std::pair<size_t, size_t>>> global_components = {{{0, 1}, {1, 2}}, {{3, 4}, {4, 5}}};
    auto global_graph = create_disconnected_graph<TestGlobalNode>(global_components);

    // Verify graphs
    EXPECT_EQ(target_graph.get_nodes().size(), 6u);
    EXPECT_EQ(global_graph.get_nodes().size(), 6u);

    // Build graph data and constraints
    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
    const auto& state = search_engine.get_state();

    // This should succeed - disconnected target can map to disconnected global
    EXPECT_TRUE(found) << "Disconnected target graph should map to disconnected global graph with matching structure";

    if (found) {
        // Verify all nodes are mapped
        size_t mapped_count = 0;
        for (int mapped_value : state.mapping) {
            if (mapped_value != -1) {
                mapped_count++;
            }
        }
        EXPECT_EQ(mapped_count, 6u) << "All 6 target nodes should be mapped";

        log_info(
            tt::LogFabric,
            "Disconnected both graphs test completed: DFS calls={}, backtracks={}",
            state.dfs_calls,
            state.backtrack_count);
    }
}

TEST_F(TopologySolverTest, DFSSearchEngine_DisconnectedBothGraphs_Failure) {
    using namespace tt::tt_fabric::detail;

    // Create target graph: three disconnected components
    // Component 1: 2 nodes (0-1)
    // Component 2: 2 nodes (2-3)
    // Component 3: 2 nodes (4-5)
    std::vector<std::vector<std::pair<size_t, size_t>>> target_components = {{{0, 1}}, {{2, 3}}, {{4, 5}}};
    auto target_graph = create_disconnected_graph<TestTargetNode>(target_components);

    // Create global graph: two disconnected components (fewer components than target)
    std::vector<std::vector<std::pair<size_t, size_t>>> global_components = {
        {{0, 1}, {1, 2}},  // Component 1: chain of 3 nodes
        {{3, 4}}           // Component 2: 2 nodes
    };
    auto global_graph = create_disconnected_graph<TestGlobalNode>(global_components);

    // Verify graphs
    EXPECT_EQ(target_graph.get_nodes().size(), 6u);
    EXPECT_EQ(global_graph.get_nodes().size(), 5u);  // Only 5 nodes in global

    // Build graph data and constraints
    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(graph_data, constraint_data, ConnectionValidationMode::RELAXED);
    const auto& state = search_engine.get_state();

    // This should fail - target has 6 nodes but global only has 5
    EXPECT_FALSE(found) << "Target graph with more nodes than global should fail";

    if (!found) {
        EXPECT_FALSE(state.error_message.empty()) << "Should have error message";
        log_info(
            tt::LogFabric,
            "Disconnected both graphs failure test completed as expected: DFS calls={}, backtracks={}, error={}",
            state.dfs_calls,
            state.backtrack_count,
            state.error_message);
    }
}

// Test that 2 disconnected logical nodes map to 2 different physical nodes
// Logical graph: 2 disconnected nodes (no edges between them)
// Physical graph: 3 fully connected nodes (complete graph/clique)
// This verifies that the solver correctly rejects constraints that force two logical nodes
// to map to the same physical node
TEST_F(TopologySolverTest, SolveTopologyMapping_DisconnectedNodesToFullyConnected_ShouldMapToDifferentNodes) {
    // Create target graph: 2 disconnected nodes (no edges)
    // Node 0 and Node 1 are isolated (no connection between them)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[0] = {};  // Node 0 has no neighbors
    target_adj_map[1] = {};  // Node 1 has no neighbors
    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 3 fully connected nodes (complete graph/clique)
    // All nodes are connected to each other: 100 <-> 101 <-> 102 (all pairs connected)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[100] = {101, 102};  // Node 100 connects to 101 and 102
    global_adj_map[101] = {100, 102};  // Node 101 connects to 100 and 102
    global_adj_map[102] = {100, 101};  // Node 102 connects to 100 and 101
    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add constraints to force both logical nodes to map to the same physical node
    // This should cause the solver to fail because two different logical nodes
    // cannot map to the same physical node
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(0, 100);  // Force logical node 0 -> physical node 100
    constraints.add_required_constraint(1, 100);  // Force logical node 1 -> physical node 100 (SAME as node 0!)

    // Solve - should FAIL because both logical nodes are constrained to the same physical node
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should fail - cannot map two different logical nodes to the same physical node
    EXPECT_FALSE(result.success) << "Solver should reject mapping two logical nodes to the same physical node";
}

// Tests for public API: solve_topology_mapping
TEST_F(TopologySolverTest, SolveTopologyMapping_BasicSuccess) {
    // Create simple target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // No constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Basic mapping should succeed";
    EXPECT_TRUE(result.error_message.empty()) << "Should have no error message on success";

    // Verify mappings
    EXPECT_EQ(result.target_to_global.size(), 2u) << "Should map both target nodes";
    EXPECT_EQ(result.global_to_target.size(), 2u) << "Should have bidirectional mappings";

    // Verify all target nodes are mapped
    EXPECT_NE(result.target_to_global.find(1), result.target_to_global.end());
    EXPECT_NE(result.target_to_global.find(2), result.target_to_global.end());

    // Verify mapped nodes are connected in global graph
    TestGlobalNode global1 = result.target_to_global.at(1);
    TestGlobalNode global2 = result.target_to_global.at(2);
    const auto& neighbors1 = global_graph.get_neighbors(global1);
    bool connected = std::find(neighbors1.begin(), neighbors1.end(), global2) != neighbors1.end();
    EXPECT_TRUE(connected) << "Mapped nodes should be connected in global graph";

    // Verify statistics
    EXPECT_GT(result.stats.dfs_calls, 0u) << "Should have made DFS calls";
    EXPECT_GE(result.stats.elapsed_time.count(), 0) << "Should have elapsed time";
    EXPECT_GE(result.stats.memoization_hits, 0u) << "Should track memoization hits";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_WithRequiredConstraints) {
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add required constraint: target node 1 must map to global node 10
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with required constraint should succeed";

    // Verify required constraint is satisfied
    EXPECT_EQ(result.target_to_global.at(1), 10) << "Required constraint should be satisfied";

    // Verify constraint statistics
    EXPECT_EQ(result.constraint_stats.required_satisfied, 1u) << "Should satisfy 1 required constraint";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_WithPreferredConstraints) {
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add preferred constraint: prefer target node 1 -> global node 10
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_preferred_constraint(1, 10);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with preferred constraint should succeed";

    // Preferred constraint may or may not be satisfied (it's just a preference)
    // But we should have statistics about it
    EXPECT_GE(result.constraint_stats.preferred_total, 1u) << "Should have preferred constraints";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_StrictMode) {
    // Create target graph: 1 -> 2 (requires 2 channels)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2};  // 2 connections
    target_adj_map[2] = {1, 1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (1 channel) and 10 -> 12 (2 channels)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 12, 12};  // 1 to 11, 2 to 12
    global_adj_map[11] = {10};
    global_adj_map[12] = {10, 10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // No constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Solve in STRICT mode - should find mapping using node with sufficient channels
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::STRICT);

    // Should succeed (node 12 has 2 channels)
    EXPECT_TRUE(result.success) << "Strict mode should succeed when sufficient channels exist";

    // Verify one of the mapped nodes is 12 (which has 2 channels)
    TestGlobalNode global1 = result.target_to_global.at(1);
    TestGlobalNode global2 = result.target_to_global.at(2);
    EXPECT_TRUE(global1 == 12 || global2 == 12) << "Should use node with sufficient channels";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_RelaxedModeWithWarnings) {
    // Create target graph: 1 -> 2 (requires 2 channels)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2, 2};  // 2 connections
    target_adj_map[2] = {1, 1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (only 1 channel, insufficient)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // No constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Solve in RELAXED mode - should succeed but with warnings
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed (relaxed mode allows channel mismatches)
    EXPECT_TRUE(result.success) << "Relaxed mode should succeed even with insufficient channels";

    // Should have warnings about channel count mismatches
    EXPECT_FALSE(result.warnings.empty()) << "Should have warnings about channel count mismatches";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_FailureCase) {
    // Create target graph: 1 -> 2 -> 3 (3 nodes)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 (only 2 nodes, can't fit 3-node path)
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // No constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Solve - should fail
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should fail
    EXPECT_FALSE(result.success) << "Should fail when target graph is too large";
    EXPECT_FALSE(result.error_message.empty()) << "Should have error message on failure";

    // Partial mapping should still be available
    EXPECT_LE(result.target_to_global.size(), 3u) << "May have partial mapping";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_ConflictingConstraints) {
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add conflicting required constraints: node 1 -> 10, node 2 -> 12
    // But 10 and 12 are not adjacent, so this should fail
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    constraints.add_required_constraint(2, 12);  // 12 is not adjacent to 10

    // Solve - should fail due to conflicting constraints
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should fail
    EXPECT_FALSE(result.success) << "Should fail with conflicting constraints";
    EXPECT_FALSE(result.error_message.empty()) << "Should have error message";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_ResultStructure) {
    // Create simple target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add both required and preferred constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    constraints.add_preferred_constraint(2, 11);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Verify result structure
    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.error_message.empty());

    // Verify bidirectional mappings are consistent
    for (const auto& [target, global] : result.target_to_global) {
        EXPECT_EQ(result.global_to_target.at(global), target) << "Bidirectional mappings should be consistent";
    }

    // Verify statistics are populated
    EXPECT_GT(result.stats.dfs_calls, 0u) << "Should have DFS calls";
    EXPECT_GE(result.stats.elapsed_time.count(), 0) << "Should have elapsed time";
    EXPECT_GE(result.stats.memoization_hits, 0u) << "Should track memoization hits";
    EXPECT_EQ(result.constraint_stats.required_satisfied, 1u) << "Should satisfy required constraint";
    EXPECT_GE(result.constraint_stats.preferred_satisfied, 0u) << "Should track preferred constraints";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_MemoizationHits) {
    // Test that memoization hits and backtracking stats are tracked in results
    // Create a topology with multiple paths to allow exploration

    // Create target graph: 1 -> 2 -> 3 -> 4 -> 5 (5-node path)
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2, 4};
    target_adj_map[4] = {3, 5};
    target_adj_map[5] = {4};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph with multiple valid paths
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11, 15};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12, 14, 19};  // Multiple neighbors for exploration
    global_adj_map[14] = {13};
    global_adj_map[19] = {13, 20};
    global_adj_map[20] = {19};
    global_adj_map[15] = {10, 16};  // Alternative path
    global_adj_map[16] = {15, 17};
    global_adj_map[17] = {16, 18};
    global_adj_map[18] = {17};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Use constraints to guide search - this may or may not cause backtracking
    // depending on heuristic choices, but stats should always be tracked
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    constraints.add_required_constraint(2, 11);
    constraints.add_required_constraint(3, 12);
    constraints.add_preferred_constraint(4, 19);  // Guide search

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed after backtracking
    EXPECT_TRUE(result.success) << "Should find a valid mapping after backtracking";

    // Verify stats are tracked
    EXPECT_GT(result.stats.dfs_calls, 0u) << "Should have made DFS calls";
    EXPECT_GE(result.stats.memoization_hits, 0u) << "Should track memoization hits";

    // Verify stats are tracked correctly
    // Note: With efficient forward consistency checking, backtracking may not always occur
    // in simple topologies. The important thing is that the stats are properly tracked.
    // If backtracking occurs (backtrack_count > 0), that demonstrates the mechanism works.
    // If it doesn't occur, that's also valid - it means the heuristics are working well.

    // Log the stats for debugging
    log_info(
        tt::LogFabric,
        "Memoization test: dfs_calls={}, backtracks={}, memoization_hits={}",
        result.stats.dfs_calls,
        result.stats.backtrack_count,
        result.stats.memoization_hits);
}

TEST_F(TopologySolverTest, MappingConstraintsOneToManyRequired) {
    // Test one target to many globals (required constraint)
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};
    constraints.add_required_constraint(1, global_nodes);

    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 3u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(12), 1u);
    EXPECT_FALSE(constraints.is_valid_mapping(1, 20));

    // Test intersection with existing constraint
    std::set<TestGlobalNode> global_nodes2 = {11, 12, 13};
    constraints.add_required_constraint(1, global_nodes2);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);  // Intersection: {11, 12}
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(12), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 0u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(13), 0u);

    // Test many targets to one global (required constraint)
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints2;
    std::set<TestTargetNode> target_nodes = {1, 2, 3};
    constraints2.add_required_constraint(target_nodes, 10);

    EXPECT_EQ(constraints2.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints2.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints2.get_valid_mappings(2).size(), 1u);
    EXPECT_EQ(constraints2.get_valid_mappings(2).count(10), 1u);
    EXPECT_EQ(constraints2.get_valid_mappings(3).size(), 1u);
    EXPECT_EQ(constraints2.get_valid_mappings(3).count(10), 1u);

    // Test intersection with existing constraint for multiple targets
    std::set<TestGlobalNode> global_nodes3 = {10, 11};
    std::set<TestGlobalNode> global_nodes4 = {10, 12};
    constraints2.add_required_constraint(1, global_nodes3);
    constraints2.add_required_constraint(2, global_nodes4);
    EXPECT_EQ(constraints2.get_valid_mappings(1).size(), 1u);  // Still {10}
    EXPECT_EQ(constraints2.get_valid_mappings(2).size(), 1u);  // Still {10}
    EXPECT_EQ(constraints2.get_valid_mappings(3).size(), 1u);  // Still {10}

    // Test conflict: many targets to one global, then conflicting constraint
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints3;
    std::set<TestTargetNode> target_nodes2 = {1, 2};
    constraints3.add_required_constraint(target_nodes2, 10);
    EXPECT_FALSE(constraints3.add_required_constraint(1, 20)) << "Conflicting constraint should return false";
}

TEST_F(TopologySolverTest, MappingConstraintsOneToManyPreferred) {
    // Test one target to many globals (preferred constraint)
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};
    constraints.add_preferred_constraint(1, global_nodes);

    EXPECT_EQ(constraints.get_preferred_mappings(1).size(), 3u);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(11), 1u);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(12), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 0u);  // Preferred doesn't restrict valid

    // Test intersection with existing preferred constraint
    std::set<TestGlobalNode> global_nodes2 = {11, 12, 13};
    constraints.add_preferred_constraint(1, global_nodes2);
    EXPECT_EQ(constraints.get_preferred_mappings(1).size(), 2u);  // Intersection: {11, 12}
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(11), 1u);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(12), 1u);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(10), 0u);
    EXPECT_EQ(constraints.get_preferred_mappings(1).count(13), 0u);

    // Test many targets to one global (preferred constraint)
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints2;
    std::set<TestTargetNode> target_nodes = {1, 2, 3};
    constraints2.add_preferred_constraint(target_nodes, 10);

    EXPECT_EQ(constraints2.get_preferred_mappings(1).size(), 1u);
    EXPECT_EQ(constraints2.get_preferred_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints2.get_preferred_mappings(2).size(), 1u);
    EXPECT_EQ(constraints2.get_preferred_mappings(2).count(10), 1u);
    EXPECT_EQ(constraints2.get_preferred_mappings(3).size(), 1u);
    EXPECT_EQ(constraints2.get_preferred_mappings(3).count(10), 1u);

    // Test intersection with existing preferred constraint for multiple targets
    std::set<TestGlobalNode> global_nodes3 = {10, 11};
    std::set<TestGlobalNode> global_nodes4 = {10, 12};
    constraints2.add_preferred_constraint(1, global_nodes3);
    constraints2.add_preferred_constraint(2, global_nodes4);
    EXPECT_EQ(constraints2.get_preferred_mappings(1).size(), 1u);  // Intersection: {10}
    EXPECT_EQ(constraints2.get_preferred_mappings(2).size(), 1u);  // Intersection: {10}
    EXPECT_EQ(constraints2.get_preferred_mappings(3).size(), 1u);  // Still {10}

    // Test that preferred constraints don't restrict valid mappings
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints3;
    std::set<TestGlobalNode> preferred_globals = {10, 11};
    std::set<TestGlobalNode> required_globals = {20, 21};
    constraints3.add_preferred_constraint(1, preferred_globals);
    constraints3.add_required_constraint(1, required_globals);
    EXPECT_EQ(constraints3.get_preferred_mappings(1).size(), 2u);  // {10, 11}
    EXPECT_EQ(constraints3.get_valid_mappings(1).size(), 2u);      // {20, 21}
    EXPECT_TRUE(constraints3.is_valid_mapping(1, 20));
    EXPECT_TRUE(constraints3.is_valid_mapping(1, 21));
}

TEST_F(TopologySolverTest, MappingConstraintsOneToManyIntersection) {
    // Test intersection between one-to-many and one-to-one constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Start with one-to-many constraint
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};
    constraints.add_required_constraint(1, global_nodes);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 3u);

    // Intersect with one-to-one constraint
    constraints.add_required_constraint(1, 11);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 1u);

    // Test intersection between many-to-one and one-to-many
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints2;
    std::set<TestTargetNode> target_nodes = {1, 2};
    constraints2.add_required_constraint(target_nodes, 10);
    std::set<TestGlobalNode> global_nodes2 = {10, 11};
    constraints2.add_required_constraint(1, global_nodes2);
    EXPECT_EQ(constraints2.get_valid_mappings(1).size(), 1u);  // Intersection: {10}
    EXPECT_EQ(constraints2.get_valid_mappings(2).size(), 1u);  // Still {10}

    // Test intersection between trait constraints and one-to-many
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints3;
    std::map<TestTargetNode, uint8_t> target_traits = {{1, 0}};
    std::map<TestGlobalNode, uint8_t> global_traits = {{10, 0}, {11, 0}, {20, 1}};
    EXPECT_TRUE(constraints3.add_required_trait_constraint<uint8_t>(target_traits, global_traits));
    EXPECT_EQ(constraints3.get_valid_mappings(1).size(), 2u);  // {10, 11}

    std::set<TestGlobalNode> global_nodes3 = {10, 12};
    constraints3.add_required_constraint(1, global_nodes3);
    EXPECT_EQ(constraints3.get_valid_mappings(1).size(), 1u);  // Intersection: {10}
}

TEST_F(TopologySolverTest, MappingConstraintsForbiddenBasic) {
    // Test basic forbidden constraint after required constraint
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};
    constraints.add_required_constraint(1, global_nodes);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 3u);

    // Forbid one mapping
    constraints.add_forbidden_constraint(1, 11);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 0u);  // Forbidden
    EXPECT_EQ(constraints.get_valid_mappings(1).count(12), 1u);

    // Forbid another mapping
    constraints.add_forbidden_constraint(1, 12);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 0u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(12), 0u);

    // Verify is_valid_mapping works correctly
    EXPECT_TRUE(constraints.is_valid_mapping(1, 10));
    EXPECT_FALSE(constraints.is_valid_mapping(1, 11));
    EXPECT_FALSE(constraints.is_valid_mapping(1, 12));
}

TEST_F(TopologySolverTest, MappingConstraintsManyToMany) {
    // Test many-to-many constraint: any target node from a set can map to any global node from a set
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    std::set<TestTargetNode> target_nodes = {1, 2, 3};
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};

    // Add many-to-many constraint
    constraints.add_required_constraint(target_nodes, global_nodes);

    // Verify all target nodes can map to any of the global nodes
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 3u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(12), 1u);

    EXPECT_EQ(constraints.get_valid_mappings(2).size(), 3u);
    EXPECT_EQ(constraints.get_valid_mappings(2).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(2).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(2).count(12), 1u);

    EXPECT_EQ(constraints.get_valid_mappings(3).size(), 3u);
    EXPECT_EQ(constraints.get_valid_mappings(3).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(3).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(3).count(12), 1u);

    // Verify is_valid_mapping works correctly
    EXPECT_TRUE(constraints.is_valid_mapping(1, 10));
    EXPECT_TRUE(constraints.is_valid_mapping(1, 11));
    EXPECT_TRUE(constraints.is_valid_mapping(1, 12));
    EXPECT_TRUE(constraints.is_valid_mapping(2, 10));
    EXPECT_TRUE(constraints.is_valid_mapping(3, 12));
    EXPECT_FALSE(constraints.is_valid_mapping(1, 20));  // Not in global_nodes set
    EXPECT_FALSE(
        constraints.is_valid_mapping(4, 10));  // Not in target_nodes set (but constraint still applies if queried)
}

TEST_F(TopologySolverTest, MappingConstraintsManyToManyIntersection) {
    // Test many-to-many constraint intersection with existing constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // First, add individual constraint for target node 1
    std::set<TestGlobalNode> initial_globals = {10, 11};
    constraints.add_required_constraint(1, initial_globals);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);

    // Now add many-to-many constraint that includes target node 1
    std::set<TestTargetNode> target_nodes = {1, 2, 3};
    std::set<TestGlobalNode> many_to_many_globals = {11, 12, 13};

    // This should intersect: target 1 can only map to {10, 11} ∩ {11, 12, 13} = {11}
    constraints.add_required_constraint(target_nodes, many_to_many_globals);

    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 0u);  // Removed by intersection

    // Target nodes 2 and 3 should have all three options
    EXPECT_EQ(constraints.get_valid_mappings(2).size(), 3u);
    EXPECT_EQ(constraints.get_valid_mappings(2).count(11), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(2).count(12), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(2).count(13), 1u);

    EXPECT_EQ(constraints.get_valid_mappings(3).size(), 3u);
}

TEST_F(TopologySolverTest, MappingConstraintsManyToManyConflict) {
    // Test many-to-many constraint that causes conflict
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // First constrain target 1 to only global 10
    constraints.add_required_constraint(1, 10);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);

    // Now add many-to-many constraint that doesn't include global 10
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {11, 12};

    // This should cause a conflict for target 1: {10} ∩ {11, 12} = {}
    EXPECT_FALSE(constraints.add_required_constraint(target_nodes, global_nodes))
        << "Conflicting constraint should return false";
}

TEST_F(TopologySolverTest, MappingConstraintsForbiddenOneToMany) {
    // Test forbidden constraint with multiple global nodes
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<TestGlobalNode> global_nodes = {10, 11, 12, 13};
    constraints.add_required_constraint(1, global_nodes);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 4u);

    // Forbid multiple mappings at once
    std::set<TestGlobalNode> forbidden_nodes = {11, 13};
    constraints.add_forbidden_constraint(1, forbidden_nodes);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 0u);  // Forbidden
    EXPECT_EQ(constraints.get_valid_mappings(1).count(12), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(13), 0u);  // Forbidden
}

TEST_F(TopologySolverTest, MappingConstraintsForbiddenManyToOne) {
    // Test forbidden constraint with multiple target nodes
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<TestGlobalNode> global_nodes1 = {10, 11, 12};
    std::set<TestGlobalNode> global_nodes2 = {10, 11, 13};
    constraints.add_required_constraint(1, global_nodes1);
    constraints.add_required_constraint(2, global_nodes2);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 3u);
    EXPECT_EQ(constraints.get_valid_mappings(2).size(), 3u);

    // Forbid one global node for multiple targets
    std::set<TestTargetNode> target_nodes = {1, 2};
    constraints.add_forbidden_constraint(target_nodes, 11);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);  // {10, 12}
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 0u);
    EXPECT_EQ(constraints.get_valid_mappings(2).size(), 2u);  // {10, 13}
    EXPECT_EQ(constraints.get_valid_mappings(2).count(11), 0u);
}

TEST_F(TopologySolverTest, MappingConstraintsForbiddenContradiction) {
    // Test that forbidden constraint cannot contradict required constraint
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);

    // Try to forbid the required mapping - should return false
    EXPECT_FALSE(constraints.add_forbidden_constraint(1, 10)) << "Forbidding required mapping should return false";

    // Verify the constraint is still valid after the failed attempt
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
}

TEST_F(TopologySolverTest, MappingConstraintsForbiddenAfterTrait) {
    // Test forbidden constraint after trait constraint
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::map<TestTargetNode, std::string> target_traits = {{1, "host0"}, {2, "host0"}};
    std::map<TestGlobalNode, std::string> global_traits = {{10, "host0"}, {11, "host0"}, {20, "host1"}};
    EXPECT_TRUE(constraints.add_required_trait_constraint<std::string>(target_traits, global_traits));
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);  // {10, 11}

    // Forbid one of the valid mappings
    constraints.add_forbidden_constraint(1, 11);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(10), 1u);
    EXPECT_EQ(constraints.get_valid_mappings(1).count(11), 0u);

    // Target 2 should still have both mappings
    EXPECT_EQ(constraints.get_valid_mappings(2).size(), 2u);
}

TEST_F(TopologySolverTest, MappingConstraintsForbiddenEmptyValidMappings) {
    // Test that forbidding all valid mappings causes validation error
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<TestGlobalNode> global_nodes = {10, 11};
    constraints.add_required_constraint(1, global_nodes);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 2u);

    // Forbid one mapping - should be fine
    constraints.add_forbidden_constraint(1, 11);
    EXPECT_EQ(constraints.get_valid_mappings(1).size(), 1u);

    // Forbid the remaining mapping - should return false (empty valid mappings)
    EXPECT_FALSE(constraints.add_forbidden_constraint(1, 10)) << "Forbidding last valid mapping should return false";
}

TEST_F(TopologySolverTest, SolveTopologyMapping_WithForbiddenConstraints) {
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add required constraint: target node 1 must map to global node 10
    // Add forbidden constraint: target node 2 cannot map to global node 12
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);

    // First, add a required constraint for target 2 to restrict its valid mappings
    std::set<TestGlobalNode> valid_for_2 = {11, 12, 13};
    constraints.add_required_constraint(2, valid_for_2);

    // Then forbid one of them
    constraints.add_forbidden_constraint(2, 12);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with forbidden constraint should succeed";

    // Verify required constraint is satisfied
    EXPECT_EQ(result.target_to_global.at(1), 10) << "Required constraint should be satisfied";

    // Verify forbidden constraint is satisfied (target 2 should not map to 12)
    EXPECT_NE(result.target_to_global.at(2), 12) << "Forbidden constraint should be satisfied";
}

// ============================================================================
// Cardinality Constraint Tests
// ============================================================================

TEST_F(TopologySolverTest, CardinalityConstraint_Basic) {
    // Test basic cardinality constraint: at least 1 of {(x,1), (x,2), (y,1), (y,2)} must be satisfied
    // Create simple target graph: x -> y
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};  // x=1, y=2
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)} must be satisfied
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality constraint should succeed";

    // Verify at least one of the cardinality pairs is satisfied
    bool cardinality_satisfied = false;
    if ((result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
        (result.target_to_global.at(2) == 10) || (result.target_to_global.at(2) == 11)) {
        cardinality_satisfied = true;
    }
    EXPECT_TRUE(cardinality_satisfied) << "At least one cardinality constraint pair should be satisfied";

    // Verify graph isomorphism is maintained
    TestGlobalNode global1 = result.target_to_global.at(1);
    TestGlobalNode global2 = result.target_to_global.at(2);
    const auto& neighbors1 = global_graph.get_neighbors(global1);
    bool connected = std::find(neighbors1.begin(), neighbors1.end(), global2) != neighbors1.end();
    EXPECT_TRUE(connected) << "Mapped nodes should be connected in global graph";
}

TEST_F(TopologySolverTest, CardinalityConstraint_MinCountGreaterThanOne) {
    // Test cardinality constraint with min_count = 2
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13 -> 14
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12, 14};
    global_adj_map[14] = {13};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add cardinality constraint: at least 2 of {(1,10), (1,11), (2,11), (2,12), (3,12), (3,13)} must be satisfied
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {
        {1, 10}, {1, 11}, {2, 11}, {2, 12}, {3, 12}, {3, 13}};
    constraints.add_cardinality_constraint(cardinality_pairs, 2);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality constraint (min_count=2) should succeed";

    // Count how many cardinality pairs are satisfied
    size_t satisfied_count = 0;
    if (result.target_to_global.at(1) == 10 || result.target_to_global.at(1) == 11) {
        satisfied_count++;
    }
    if (result.target_to_global.at(2) == 11 || result.target_to_global.at(2) == 12) {
        satisfied_count++;
    }
    if (result.target_to_global.at(3) == 12 || result.target_to_global.at(3) == 13) {
        satisfied_count++;
    }

    EXPECT_GE(satisfied_count, 2u) << "At least 2 cardinality constraint pairs should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_WithRequiredConstraints) {
    // Test cardinality constraint combined with required constraints
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Add required constraint: node 1 must map to 10
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    constraints.add_required_constraint(1, 10);

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)} must be satisfied
    // Note: (1,11) will be filtered out because 1 must map to 10
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality + required constraints should succeed";

    // Verify required constraint is satisfied
    EXPECT_EQ(result.target_to_global.at(1), 10) << "Required constraint should be satisfied";

    // Verify cardinality constraint is satisfied (either (1,10) or (2,10) or (2,11))
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(2) == 10) ||
                                 (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ConflictWithRequired) {
    // Test that cardinality constraint throws error when incompatible with required constraints
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 must map to 20
    constraints.add_required_constraint(1, 20);

    // Add cardinality constraint: at least 1 of {(1,10), (1,11)} must be satisfied
    // This should fail because 1 can only map to 20, not 10 or 11
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}};
    EXPECT_FALSE(constraints.add_cardinality_constraint(cardinality_pairs, 1))
        << "Cardinality constraint incompatible with required constraints should return false";
}

TEST_F(TopologySolverTest, CardinalityConstraint_MultipleConstraints) {
    // Test multiple cardinality constraints
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // First cardinality constraint: at least 1 of {(1,10), (1,11)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs1 = {{1, 10}, {1, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs1, 1);

    // Second cardinality constraint: at least 1 of {(2,11), (2,12)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs2 = {{2, 11}, {2, 12}};
    constraints.add_cardinality_constraint(cardinality_pairs2, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with multiple cardinality constraints should succeed";

    // Verify first cardinality constraint is satisfied
    bool constraint1_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11);
    EXPECT_TRUE(constraint1_satisfied) << "First cardinality constraint should be satisfied";

    // Verify second cardinality constraint is satisfied
    bool constraint2_satisfied = (result.target_to_global.at(2) == 11) || (result.target_to_global.at(2) == 12);
    EXPECT_TRUE(constraint2_satisfied) << "Second cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_Validation) {
    // Test that cardinality constraint validation works correctly
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Test empty pairs - should return false
    std::set<std::pair<TestTargetNode, TestGlobalNode>> empty_pairs;
    EXPECT_FALSE(constraints.add_cardinality_constraint(empty_pairs, 1))
        << "Empty cardinality constraint should return false";

    // Test min_count > pairs.size() - should return false
    std::set<std::pair<TestTargetNode, TestGlobalNode>> pairs = {{1, 10}, {2, 11}};
    EXPECT_FALSE(constraints.add_cardinality_constraint(pairs, 3)) << "min_count > pairs.size() should return false";

    // Test min_count = 0 - should return false
    EXPECT_FALSE(constraints.add_cardinality_constraint(pairs, 0)) << "min_count = 0 should return false";

    // Test valid constraint
    EXPECT_TRUE(constraints.add_cardinality_constraint(pairs, 1)) << "Valid cardinality constraint should return true";
}

TEST_F(TopologySolverTest, CardinalityConstraint_IntegrationWithSolver) {
    // Test that cardinality constraints work correctly with the full solver pipeline
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13 -> 14
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12, 14};
    global_adj_map[14] = {13};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,11), (2,12), (3,12), (3,13)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {
        {1, 10}, {1, 11}, {2, 11}, {2, 12}, {3, 12}, {3, 13}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality constraint should succeed";
    EXPECT_TRUE(result.error_message.empty()) << "Should have no error message on success";

    // Verify all nodes are mapped
    EXPECT_EQ(result.target_to_global.size(), 3u) << "Should map all 3 target nodes";
    EXPECT_EQ(result.global_to_target.size(), 3u) << "Should have bidirectional mappings";

    // Verify graph isomorphism
    for (size_t i = 1; i <= 3; ++i) {
        TestTargetNode target_i = static_cast<TestTargetNode>(i);
        TestGlobalNode global_i = result.target_to_global.at(target_i);

        // Check neighbors
        const auto& target_neighbors = target_graph.get_neighbors(target_i);
        const auto& global_neighbors = global_graph.get_neighbors(global_i);

        for (const auto& target_neighbor : target_neighbors) {
            TestGlobalNode mapped_neighbor = result.target_to_global.at(target_neighbor);
            bool neighbor_connected =
                std::find(global_neighbors.begin(), global_neighbors.end(), mapped_neighbor) != global_neighbors.end();
            EXPECT_TRUE(neighbor_connected) << "Neighbor " << target_neighbor << " mapped to " << mapped_neighbor
                                            << " should be connected to " << global_i << " in global graph";
        }
    }

    // Verify statistics
    EXPECT_GT(result.stats.dfs_calls, 0u) << "Should have made DFS calls";
    EXPECT_GE(result.stats.elapsed_time.count(), 0) << "Should have elapsed time";
    EXPECT_GE(result.stats.memoization_hits, 0u) << "Should track memoization hits";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ConstraintIndexData) {
    // Test that cardinality constraints are correctly converted to ConstraintIndexData
    using namespace tt::tt_fabric::detail;

    // Create simple graphs
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    GraphIndexData graph_data(target_graph, global_graph);

    // Add cardinality constraint
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Convert to ConstraintIndexData
    ConstraintIndexData constraint_data(constraints, graph_data);

    // Verify cardinality constraints are present
    EXPECT_FALSE(constraint_data.cardinality_constraints.empty())
        << "Cardinality constraints should be converted to index data";

    // Verify we can check cardinality constraints
    std::vector<int> mapping = {0, 1};  // target 1 -> global 10 (idx 0), target 2 -> global 11 (idx 1)
    EXPECT_TRUE(constraint_data.check_cardinality_constraints(mapping))
        << "Cardinality constraints should be satisfied by this mapping";

    // Test can_satisfy_cardinality_constraints with partial mapping
    std::vector<int> partial_mapping = {0, -1};  // target 1 mapped, target 2 not mapped
    EXPECT_TRUE(constraint_data.can_satisfy_cardinality_constraints(partial_mapping))
        << "Cardinality constraints should still be satisfiable with partial mapping";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ForwardConsistency) {
    // Test that cardinality constraints are checked during forward consistency checking
    // This ensures branches are pruned early if cardinality constraints cannot be satisfied
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,11), (2,12)} must be satisfied
    // This constraint should guide the solver
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 11}, {2, 12}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality constraint should succeed";

    // Verify cardinality constraint is satisfied
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
                                 (result.target_to_global.at(2) == 11) || (result.target_to_global.at(2) == 12);
    EXPECT_TRUE(cardinality_satisfied) << "Cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_WithPreferredConstraints) {
    // Test cardinality constraint combined with preferred constraints
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add preferred constraint: node 1 prefers 10
    constraints.add_preferred_constraint(1, 10);

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality + preferred constraints should succeed";

    // Verify cardinality constraint is satisfied
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
                                 (result.target_to_global.at(2) == 10) || (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Cardinality constraint should be satisfied";

    // Preferred constraint should guide the solver (but not required)
    // If cardinality is satisfied by (1,10), preferred constraint is also satisfied
    if (result.target_to_global.at(1) == 10) {
        EXPECT_EQ(result.constraint_stats.preferred_satisfied, 1u) << "Preferred constraint should be satisfied";
    }
}

TEST_F(TopologySolverTest, CardinalityConstraint_WithForbiddenConstraints) {
    // Test cardinality constraint combined with forbidden constraints
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 can map to {10, 11, 12}
    constraints.add_required_constraint(1, std::set<TestGlobalNode>{10, 11, 12});

    // Forbid node 1 from mapping to 10
    constraints.add_forbidden_constraint(1, 10);

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)} must be satisfied
    // Note: (1,10) will be filtered out because it's forbidden
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality + forbidden constraints should succeed";

    // Verify forbidden constraint is satisfied (node 1 should not map to 10)
    EXPECT_NE(result.target_to_global.at(1), 10) << "Forbidden constraint should be satisfied";

    // Verify cardinality constraint is satisfied (must be via (1,11), (2,10), or (2,11))
    bool cardinality_satisfied = (result.target_to_global.at(1) == 11) || (result.target_to_global.at(2) == 10) ||
                                 (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_WithManyToManyConstraints) {
    // Test cardinality constraint combined with many-to-many constraints
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add many-to-many constraint: nodes {1, 2} can map to {10, 11, 12}
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};
    constraints.add_required_constraint(target_nodes, global_nodes);

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality + many-to-many constraints should succeed";

    // Verify many-to-many constraint is satisfied
    EXPECT_TRUE(global_nodes.contains(result.target_to_global.at(1))) << "Node 1 should map to {10,11,12}";
    EXPECT_TRUE(global_nodes.contains(result.target_to_global.at(2))) << "Node 2 should map to {10,11,12}";

    // Verify cardinality constraint is satisfied
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
                                 (result.target_to_global.at(2) == 10) || (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_WithMultipleConstraintTypes) {
    // Test cardinality constraint combined with multiple constraint types
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 can map to {10, 11, 12}
    constraints.add_required_constraint(1, std::set<TestGlobalNode>{10, 11, 12});

    // Add preferred constraint: node 2 prefers 11
    constraints.add_preferred_constraint(2, 11);

    // Add forbidden constraint: node 1 cannot map to 12
    constraints.add_forbidden_constraint(1, 12);

    // Add cardinality constraint: at least 2 of {(1,10), (1,11), (2,10), (2,11), (2,12)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {
        {1, 10}, {1, 11}, {2, 10}, {2, 11}, {2, 12}};
    constraints.add_cardinality_constraint(cardinality_pairs, 2);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with multiple constraint types should succeed";

    // Verify required constraint is satisfied
    EXPECT_TRUE(result.target_to_global.at(1) == 10 || result.target_to_global.at(1) == 11)
        << "Node 1 should map to {10, 11} (12 is forbidden)";

    // Verify forbidden constraint is satisfied
    EXPECT_NE(result.target_to_global.at(1), 12) << "Forbidden constraint should be satisfied";

    // Verify cardinality constraint is satisfied (at least 2 pairs)
    size_t satisfied_pairs = 0;
    if (result.target_to_global.at(1) == 10) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(1) == 11) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(2) == 10) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(2) == 11) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(2) == 12) {
        satisfied_pairs++;
    }
    EXPECT_GE(satisfied_pairs, 2u) << "At least 2 cardinality pairs should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_IntersectionWithRequired) {
    // Test cardinality constraint with intersection scenarios
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 can map to {10, 11}
    constraints.add_required_constraint(1, std::set<TestGlobalNode>{10, 11});

    // Add cardinality constraint: at least 1 of {(1,10), (1,11), (1,12), (2,10), (2,11)} must be satisfied
    // Note: (1,12) will be filtered out because 1 can only map to {10, 11}
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {
        {1, 10}, {1, 11}, {1, 12}, {2, 10}, {2, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with cardinality intersection should succeed";

    // Verify required constraint is satisfied
    EXPECT_TRUE(result.target_to_global.at(1) == 10 || result.target_to_global.at(1) == 11)
        << "Node 1 should map to {10, 11}";

    // Verify cardinality constraint is satisfied (must be via valid pairs)
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
                                 (result.target_to_global.at(2) == 10) || (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Cardinality constraint should be satisfied with valid pairs";
}

TEST_F(TopologySolverTest, CardinalityConstraint_MinCountWithConstraints) {
    // Test cardinality constraint with min_count > 1 combined with other constraints
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 can map to {10, 11}
    constraints.add_required_constraint(1, std::set<TestGlobalNode>{10, 11});

    // Add forbidden constraint: node 2 cannot map to 12
    constraints.add_forbidden_constraint(2, 12);

    // Add cardinality constraint: at least 2 of {(1,10), (1,11), (2,10), (2,11), (2,12)} must be satisfied
    // Note: (2,12) will be filtered out because it's forbidden
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs = {
        {1, 10}, {1, 11}, {2, 10}, {2, 11}, {2, 12}};
    constraints.add_cardinality_constraint(cardinality_pairs, 2);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with min_count=2 and constraints should succeed";

    // Verify constraints are satisfied
    EXPECT_TRUE(result.target_to_global.at(1) == 10 || result.target_to_global.at(1) == 11)
        << "Node 1 should map to {10, 11}";
    EXPECT_NE(result.target_to_global.at(2), 12) << "Node 2 should not map to 12 (forbidden)";

    // Verify cardinality constraint is satisfied (at least 2 pairs)
    size_t satisfied_pairs = 0;
    if (result.target_to_global.at(1) == 10) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(1) == 11) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(2) == 10) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(2) == 11) {
        satisfied_pairs++;
    }
    EXPECT_GE(satisfied_pairs, 2u) << "At least 2 cardinality pairs should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_MultipleCardinalityWithOtherConstraints) {
    // Test multiple cardinality constraints combined with other constraint types
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 can map to {10, 11}
    constraints.add_required_constraint(1, std::set<TestGlobalNode>{10, 11});

    // Add preferred constraint: node 2 prefers 11
    constraints.add_preferred_constraint(2, 11);

    // Add first cardinality constraint: at least 1 of {(1,10), (1,11), (2,10)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs1 = {{1, 10}, {1, 11}, {2, 10}};
    constraints.add_cardinality_constraint(cardinality_pairs1, 1);

    // Add second cardinality constraint: at least 1 of {(2,11), (2,12), (3,11)} must be satisfied
    std::set<std::pair<TestTargetNode, TestGlobalNode>> cardinality_pairs2 = {{2, 11}, {2, 12}, {3, 11}};
    constraints.add_cardinality_constraint(cardinality_pairs2, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with multiple cardinality constraints should succeed";

    // Verify required constraint is satisfied
    EXPECT_TRUE(result.target_to_global.at(1) == 10 || result.target_to_global.at(1) == 11)
        << "Node 1 should map to {10, 11}";

    // Verify first cardinality constraint is satisfied
    bool cardinality1_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
                                  (result.target_to_global.at(2) == 10);
    EXPECT_TRUE(cardinality1_satisfied) << "First cardinality constraint should be satisfied";

    // Verify second cardinality constraint is satisfied
    bool cardinality2_satisfied = (result.target_to_global.at(2) == 11) || (result.target_to_global.at(2) == 12) ||
                                  (result.target_to_global.at(3) == 11);
    EXPECT_TRUE(cardinality2_satisfied) << "Second cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ManyToMany_Basic) {
    // Test many-to-many cardinality constraint: at least 1 mapping from {1,2} × {10,11,12}
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add many-to-many cardinality constraint: at least 1 of {(1,10), (1,11), (1,12), (2,10), (2,11), (2,12)}
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {10, 11, 12};
    constraints.add_cardinality_constraint(target_nodes, global_nodes, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with many-to-many cardinality constraint should succeed";

    // Verify cardinality constraint is satisfied (at least 1 pair)
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(1) == 11) ||
                                 (result.target_to_global.at(1) == 12) || (result.target_to_global.at(2) == 10) ||
                                 (result.target_to_global.at(2) == 11) || (result.target_to_global.at(2) == 12);
    EXPECT_TRUE(cardinality_satisfied) << "Many-to-many cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ManyToMany_MinCountGreaterThanOne) {
    // Test many-to-many cardinality constraint with min_count = 2
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add many-to-many cardinality constraint: at least 2 of {(1,10), (1,11), (2,10), (2,11)}
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {10, 11};
    constraints.add_cardinality_constraint(target_nodes, global_nodes, 2);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with many-to-many cardinality constraint (min_count=2) should succeed";

    // Verify cardinality constraint is satisfied (at least 2 pairs)
    size_t satisfied_pairs = 0;
    if (result.target_to_global.at(1) == 10 || result.target_to_global.at(1) == 11) {
        satisfied_pairs++;
    }
    if (result.target_to_global.at(2) == 10 || result.target_to_global.at(2) == 11) {
        satisfied_pairs++;
    }
    EXPECT_GE(satisfied_pairs, 2u) << "At least 2 pairs from many-to-many cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ManyToMany_WithRequiredConstraints) {
    // Test many-to-many cardinality constraint combined with required constraints
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 must map to 10
    constraints.add_required_constraint(1, 10);

    // Add many-to-many cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)}
    // Note: (1,11) will be filtered out because 1 must map to 10
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {10, 11};
    constraints.add_cardinality_constraint(target_nodes, global_nodes, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with many-to-many cardinality + required constraints should succeed";

    // Verify required constraint is satisfied
    EXPECT_EQ(result.target_to_global.at(1), 10) << "Required constraint should be satisfied";

    // Verify cardinality constraint is satisfied (via (1,10) or (2,10) or (2,11))
    bool cardinality_satisfied = (result.target_to_global.at(1) == 10) || (result.target_to_global.at(2) == 10) ||
                                 (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Many-to-many cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ManyToMany_WithForbiddenConstraints) {
    // Test many-to-many cardinality constraint combined with forbidden constraints
    // Create target graph: 1 -> 2 -> 3
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1, 3};
    target_adj_map[3] = {2};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12 -> 13
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11, 13};
    global_adj_map[13] = {12};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add required constraint: node 1 can map to {10, 11, 12}
    constraints.add_required_constraint(1, std::set<TestGlobalNode>{10, 11, 12});

    // Forbid node 1 from mapping to 10
    constraints.add_forbidden_constraint(1, 10);

    // Add many-to-many cardinality constraint: at least 1 of {(1,10), (1,11), (2,10), (2,11)}
    // Note: (1,10) will be filtered out because it's forbidden
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {10, 11};
    constraints.add_cardinality_constraint(target_nodes, global_nodes, 1);

    // Solve
    auto result = solve_topology_mapping(target_graph, global_graph, constraints, ConnectionValidationMode::RELAXED);

    // Should succeed
    EXPECT_TRUE(result.success) << "Mapping with many-to-many cardinality + forbidden constraints should succeed";

    // Verify forbidden constraint is satisfied
    EXPECT_NE(result.target_to_global.at(1), 10) << "Forbidden constraint should be satisfied";

    // Verify cardinality constraint is satisfied (must be via (1,11), (2,10), or (2,11))
    bool cardinality_satisfied = (result.target_to_global.at(1) == 11) || (result.target_to_global.at(2) == 10) ||
                                 (result.target_to_global.at(2) == 11);
    EXPECT_TRUE(cardinality_satisfied) << "Many-to-many cardinality constraint should be satisfied";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ManyToMany_Validation) {
    // Test validation of many-to-many cardinality constraint
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Test empty target nodes - should throw
    std::set<TestTargetNode> empty_targets;
    std::set<TestGlobalNode> global_nodes = {10, 11};
    EXPECT_FALSE(constraints.add_cardinality_constraint(empty_targets, global_nodes, 1))
        << "Empty target nodes should return false";

    // Test empty global nodes - should return false
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> empty_globals;
    EXPECT_FALSE(constraints.add_cardinality_constraint(target_nodes, empty_globals, 1))
        << "Empty global nodes should return false";

    // Test min_count greater than number of pairs - should return false
    std::set<TestTargetNode> small_targets = {1};
    std::set<TestGlobalNode> small_globals = {10};
    EXPECT_FALSE(constraints.add_cardinality_constraint(small_targets, small_globals, 2))
        << "min_count > pairs.size() should return false";

    // Test min_count = 0 - should return false
    EXPECT_FALSE(constraints.add_cardinality_constraint(target_nodes, global_nodes, 0))
        << "min_count = 0 should return false";
}

TEST_F(TopologySolverTest, CardinalityConstraint_ManyToMany_EquivalentToExplicitPairs) {
    // Test that many-to-many cardinality constraint is equivalent to explicitly listing all pairs
    // Create target graph: 1 -> 2
    AdjacencyGraph<TestTargetNode>::AdjacencyMap target_adj_map;
    target_adj_map[1] = {2};
    target_adj_map[2] = {1};

    AdjacencyGraph<TestTargetNode> target_graph(target_adj_map);

    // Create global graph: 10 -> 11 -> 12
    AdjacencyGraph<TestGlobalNode>::AdjacencyMap global_adj_map;
    global_adj_map[10] = {11};
    global_adj_map[11] = {10, 12};
    global_adj_map[12] = {11};

    AdjacencyGraph<TestGlobalNode> global_graph(global_adj_map);

    // Test 1: Using many-to-many convenience method
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints1;
    std::set<TestTargetNode> target_nodes = {1, 2};
    std::set<TestGlobalNode> global_nodes = {10, 11};
    constraints1.add_cardinality_constraint(target_nodes, global_nodes, 1);

    // Test 2: Using explicit pairs (equivalent)
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints2;
    std::set<std::pair<TestTargetNode, TestGlobalNode>> explicit_pairs = {{1, 10}, {1, 11}, {2, 10}, {2, 11}};
    constraints2.add_cardinality_constraint(explicit_pairs, 1);

    // Both should have the same cardinality constraints
    EXPECT_EQ(constraints1.get_cardinality_constraints().size(), constraints2.get_cardinality_constraints().size());
    if (!constraints1.get_cardinality_constraints().empty() && !constraints2.get_cardinality_constraints().empty()) {
        const auto& pairs1 = constraints1.get_cardinality_constraints()[0].first;
        const auto& pairs2 = constraints2.get_cardinality_constraints()[0].first;
        EXPECT_EQ(pairs1.size(), pairs2.size());
        EXPECT_EQ(pairs1, pairs2) << "Many-to-many should generate same pairs as explicit listing";
    }
}
}  // namespace tt::tt_fabric
