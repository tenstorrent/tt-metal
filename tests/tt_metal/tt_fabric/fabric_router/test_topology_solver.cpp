// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <memory>
#include <cstdlib>
#include <gtest/gtest.h>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
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

}  // namespace tt::tt_fabric
