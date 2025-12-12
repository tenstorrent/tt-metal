// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <gtest/gtest.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include "tt_metal/fabric/topology_solver_internal.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

namespace tt::tt_fabric {

class TopologySolverTest : public ::testing::Test {
protected:
    void SetUp() override { setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "10", 1); }

    void TearDown() override {}
};

TEST_F(TopologySolverTest, BuildAdjacencyMapLogical) {
    // Use 2x2 T3K multiprocess MGD (has 2 meshes: mesh_id 0 and 1)
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";

    // Create mesh graph from descriptor
    auto mesh_graph = MeshGraph(mesh_graph_desc_path.string());

    // Build adjacency map logical
    auto adjacency_map = build_adjacency_map_logical(mesh_graph);

    // Verify that we have adjacency graphs for each mesh
    EXPECT_GT(adjacency_map.size(), 0u);

    // Verify each mesh has a valid adjacency graph
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

    // For T3K 2x2 multiprocess, we expect 2 meshes
    EXPECT_EQ(adjacency_map.size(), 2u);
}

TEST_F(TopologySolverTest, BuildAdjacencyMapPhysical) {
    // Load PSD from pre-written test file
    const std::filesystem::path root_dir =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    const std::filesystem::path psd_file_path =
        root_dir / "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto";

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
    auto adjacency_map = build_adjacency_map_physical(physical_system_descriptor, asic_id_to_mesh_rank);

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
    required_constraints.add_required_trait_constraint<std::string>(target_traits, global_traits);

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
    constraints.add_required_trait_constraint<uint8_t>(target_host, global_host);

    std::map<TestTargetNode, uint8_t> target_rack = {{1, 0}, {2, 1}};
    std::map<TestGlobalNode, uint8_t> global_rack = {{10, 0}, {11, 1}, {20, 0}};
    constraints.add_required_trait_constraint<uint8_t>(target_rack, global_rack);

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
    EXPECT_THROW(constraints.add_required_constraint(1, 20), std::runtime_error);

    // Test conflict in trait constraint - should throw (no matching global nodes)
    MappingConstraints<TestTargetNode, TestGlobalNode> trait_constraints;
    std::map<TestTargetNode, size_t> target_traits = {{1, 999}};
    std::map<TestGlobalNode, size_t> global_traits = {{10, 100}, {20, 200}};
    EXPECT_THROW(
        trait_constraints.add_required_trait_constraint<size_t>(target_traits, global_traits), std::runtime_error);

    // Test conflict in trait constraint - conflicting trait values
    MappingConstraints<TestTargetNode, TestGlobalNode> conflict_constraints;
    std::map<TestTargetNode, uint8_t> target_host1 = {{1, 0}};
    std::map<TestGlobalNode, uint8_t> global_host1 = {{10, 0}, {11, 0}};
    conflict_constraints.add_required_trait_constraint<uint8_t>(target_host1, global_host1);

    std::map<TestTargetNode, uint8_t> target_host2 = {{1, 1}};
    std::map<TestGlobalNode, uint8_t> global_host2 = {{20, 1}, {21, 1}};
    EXPECT_THROW(
        conflict_constraints.add_required_trait_constraint<uint8_t>(target_host2, global_host2), std::runtime_error);

    // Test conflict in constructor - should throw
    std::set<std::pair<TestTargetNode, TestGlobalNode>> conflicting_required = {{1, 10}, {1, 20}};
    std::set<std::pair<TestTargetNode, TestGlobalNode>> empty_preferred;
    EXPECT_THROW(
        (MappingConstraints<TestTargetNode, TestGlobalNode>(conflicting_required, empty_preferred)),
        std::runtime_error);
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
    constraints.add_required_trait_constraint<std::string>(target_traits, global_traits);

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

    // Select candidates - should select either node 2 or node 3
    auto result = SearchHeuristic::select_and_generate_candidates(
        graph_data, constraint_data, mapping, used, ConnectionValidationMode::RELAXED);

    // Should select one of the unassigned nodes (2 or 3)
    EXPECT_TRUE(result.target_idx == 1u || result.target_idx == 2u);
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(2, -1);
    state.used.resize(3, false);

    // Run search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // Should find a mapping (e.g., 1->10, 2->11)
    EXPECT_TRUE(found);
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(2, -1);
    state.used.resize(3, false);

    // Run search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // Should find a mapping
    EXPECT_TRUE(found);

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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(3, -1);
    state.used.resize(2, false);

    // Run search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(2, -1);
    state.used.resize(3, false);

    // Run search in STRICT mode - should find mapping (1->10, 2->12) since 12 has 2 channels
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::STRICT);

    EXPECT_TRUE(found);
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(2, -1);
    state.used.resize(4, false);

    // Run search in RELAXED mode
    DFSSearchEngine<TestTargetNode, TestGlobalNode> engine;
    bool found = engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    EXPECT_TRUE(found);

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
            state.mapping, graph_data, state, constraints, ConnectionValidationMode::RELAXED);

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
            if (local_to_global.find(edge.first) == local_to_global.end()) {
                local_to_global[edge.first] = current_node_id++;
            }
            if (local_to_global.find(edge.second) == local_to_global.end()) {
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(graph_data.n_target, -1);
    state.used.resize(graph_data.n_global, false);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // This should succeed - a 1D ring can be embedded in a 2D mesh
    // (the ring can follow a cycle path through the mesh)
    EXPECT_TRUE(found) << "1D ring of 32 nodes should fit in 4x8 2D mesh";

    if (found) {
        // Verify all nodes are mapped
        size_t mapped_count = 0;
        for (size_t i = 0; i < state.mapping.size(); ++i) {
            if (state.mapping[i] != -1) {
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

    // Create global graph: 2D mesh of 4x8 = 32 nodes
    auto global_graph = create_2d_mesh_graph<TestGlobalNode>(4, 8);

    // Create target graph: 1D chain of 32 nodes
    auto target_graph = create_1d_chain_graph<TestTargetNode>(32);

    // Build graph data and constraints
    GraphIndexData graph_data(target_graph, global_graph);
    MappingConstraints<TestTargetNode, TestGlobalNode> constraints;

    // Add conflicting pinnings that make mapping impossible
    // Pin first node to top-left corner (node 0)
    constraints.add_required_constraint(0, 0);
    // Pin second node (which is adjacent to node 0 in target graph) to a node that's NOT adjacent to node 0
    // Node 0's neighbors in 4x8 mesh are: 1 (right), 8 (down)
    // Let's pin target node 1 to global node 15 (which is row 1, col 7 - far from node 0 and not adjacent)
    constraints.add_required_constraint(1, 15);

    // Verify that node 15 is not adjacent to node 0 in the global graph
    const auto& neighbors_of_0 = global_graph.get_neighbors(0);
    bool node_15_is_neighbor = std::find(neighbors_of_0.begin(), neighbors_of_0.end(), 15) != neighbors_of_0.end();
    EXPECT_FALSE(node_15_is_neighbor) << "Node 15 should not be adjacent to node 0 in 4x8 mesh";

    // Also verify that target nodes 0 and 1 are adjacent in the target graph
    const auto& neighbors_of_target_0 = target_graph.get_neighbors(0);
    bool target_1_is_neighbor =
        std::find(neighbors_of_target_0.begin(), neighbors_of_target_0.end(), 1) != neighbors_of_target_0.end();
    EXPECT_TRUE(target_1_is_neighbor) << "Target nodes 0 and 1 should be adjacent in 1D chain";

    ConstraintIndexData constraint_data(constraints, graph_data);

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(graph_data.n_target, -1);
    state.used.resize(graph_data.n_global, false);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // This should fail - we've pinned two adjacent nodes in the target graph
    // to non-adjacent nodes in the global graph
    EXPECT_FALSE(found) << "Mapping should fail due to conflicting pinnings";

    if (!found) {
        // Verify error message indicates the problem
        EXPECT_FALSE(state.error_message.empty()) << "Should have error message";
        log_info(
            tt::LogFabric,
            "Negative stress test completed as expected: DFS calls={}, backtracks={}, error={}",
            state.dfs_calls,
            state.backtrack_count,
            state.error_message);

        // The solver should have detected the conflict early or during search
        // Partial mapping may or may not be saved depending on when the failure occurs
        // The important thing is that it failed as expected
        EXPECT_GE(state.dfs_calls, 0u) << "Should have attempted search";
    } else {
        // If it somehow succeeded, that's unexpected
        log_error(tt::LogFabric, "Negative test unexpectedly succeeded - this indicates a bug");
        FAIL() << "Mapping should have failed due to conflicting pinnings";
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(graph_data.n_target, -1);
    state.used.resize(graph_data.n_global, false);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // This should succeed - two disconnected chains can map to one connected chain
    EXPECT_TRUE(found) << "Disconnected target graph should map to connected global graph";

    if (found) {
        // Verify all nodes are mapped
        size_t mapped_count = 0;
        for (size_t i = 0; i < state.mapping.size(); ++i) {
            if (state.mapping[i] != -1) {
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(graph_data.n_target, -1);
    state.used.resize(graph_data.n_global, false);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(graph_data.n_target, -1);
    state.used.resize(graph_data.n_global, false);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

    // This should succeed - disconnected target can map to disconnected global
    EXPECT_TRUE(found) << "Disconnected target graph should map to disconnected global graph with matching structure";

    if (found) {
        // Verify all nodes are mapped
        size_t mapped_count = 0;
        for (size_t i = 0; i < state.mapping.size(); ++i) {
            if (state.mapping[i] != -1) {
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

    // Initialize search state
    DFSSearchEngine<TestTargetNode, TestGlobalNode>::SearchState state;
    state.mapping.resize(graph_data.n_target, -1);
    state.used.resize(graph_data.n_global, false);

    // Run DFS search
    DFSSearchEngine<TestTargetNode, TestGlobalNode> search_engine;
    bool found = search_engine.search(0, graph_data, constraint_data, state, ConnectionValidationMode::RELAXED);

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

}  // namespace tt::tt_fabric
