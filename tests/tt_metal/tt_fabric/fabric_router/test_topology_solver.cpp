// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <memory>
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
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt_metal_home) /
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
    constraints.add_required_trait_constraint<std::string>(target_traits, global_traits);

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

}  // namespace tt::tt_fabric
