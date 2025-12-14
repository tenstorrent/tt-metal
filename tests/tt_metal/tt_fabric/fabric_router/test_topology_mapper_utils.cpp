// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>

namespace tt::tt_metal::experimental::tt_fabric {
namespace {

// =============================================================================
// Test Fixture with Helper Methods
// =============================================================================

class TopologyMapperUtilsTest : public ::testing::Test {
protected:
    // Default test mesh and ranks
    static constexpr uint32_t kDefaultMeshId = 0;
    static constexpr uint64_t kAsicIdBase = 100;

    const MeshId mesh_id_{kDefaultMeshId};
    const MeshHostRankId rank0_{0};
    const MeshHostRankId rank1_{1};

    // -------------------------------------------------------------------------
    // Factory helpers
    // -------------------------------------------------------------------------

    static FabricNodeId make_node(uint32_t mesh_id, uint32_t chip_id) {
        return FabricNodeId(MeshId{mesh_id}, chip_id);
    }

    static tt::tt_metal::AsicID make_asic(uint64_t id) { return tt::tt_metal::AsicID{id}; }

    // Create N nodes for the default mesh
    static std::vector<FabricNodeId> make_nodes(size_t count) {
        std::vector<FabricNodeId> nodes;
        nodes.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            nodes.push_back(make_node(kDefaultMeshId, i));
        }
        return nodes;
    }

    // Create N ASICs with sequential IDs starting from base
    static std::vector<tt::tt_metal::AsicID> make_asics(size_t count, uint64_t base = kAsicIdBase) {
        std::vector<tt::tt_metal::AsicID> asics;
        asics.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            asics.push_back(make_asic(base + i));
        }
        return asics;
    }

    // -------------------------------------------------------------------------
    // Topology builders
    // -------------------------------------------------------------------------

    // Build a linear chain: n0 -- n1 -- n2 -- ... -- n(count-1)
    template <typename NodeType>
    static auto build_chain_adjacency(const std::vector<NodeType>& nodes) {
        std::map<NodeType, std::vector<NodeType>> adj;
        for (size_t i = 0; i < nodes.size(); ++i) {
            adj[nodes[i]] = {};
            if (i > 0) {
                adj[nodes[i]].push_back(nodes[i - 1]);
            }
            if (i + 1 < nodes.size()) {
                adj[nodes[i]].push_back(nodes[i + 1]);
            }
        }
        return adj;
    }

    // Build a fully connected graph (clique)
    template <typename NodeType>
    static auto build_clique_adjacency(const std::vector<NodeType>& nodes) {
        std::map<NodeType, std::vector<NodeType>> adj;
        for (size_t i = 0; i < nodes.size(); ++i) {
            adj[nodes[i]] = {};
            for (size_t j = 0; j < nodes.size(); ++j) {
                if (i != j) {
                    adj[nodes[i]].push_back(nodes[j]);
                }
            }
        }
        return adj;
    }

    // Build a 2D grid: rows x cols with 4-connectivity
    template <typename NodeType>
    static auto build_grid_adjacency(const std::vector<NodeType>& nodes, size_t rows, size_t cols) {
        std::map<NodeType, std::vector<NodeType>> adj;
        auto idx = [cols](size_t r, size_t c) { return r * cols + c; };

        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                adj[nodes[idx(r, c)]] = {};
                if (r > 0) {
                    adj[nodes[idx(r, c)]].push_back(nodes[idx(r - 1, c)]);
                }
                if (r + 1 < rows) {
                    adj[nodes[idx(r, c)]].push_back(nodes[idx(r + 1, c)]);
                }
                if (c > 0) {
                    adj[nodes[idx(r, c)]].push_back(nodes[idx(r, c - 1)]);
                }
                if (c + 1 < cols) {
                    adj[nodes[idx(r, c)]].push_back(nodes[idx(r, c + 1)]);
                }
            }
        }
        return adj;
    }

    // -------------------------------------------------------------------------
    // Rank map builders
    // -------------------------------------------------------------------------

    // Assign all nodes to the same rank
    static std::map<FabricNodeId, MeshHostRankId> make_uniform_node_ranks(
        const std::vector<FabricNodeId>& nodes, MeshHostRankId rank = MeshHostRankId{0}) {
        std::map<FabricNodeId, MeshHostRankId> result;
        for (const auto& node : nodes) {
            result[node] = rank;
        }
        return result;
    }

    // Assign all ASICs to the same rank
    static std::map<tt::tt_metal::AsicID, MeshHostRankId> make_uniform_asic_ranks(
        const std::vector<tt::tt_metal::AsicID>& asics, MeshHostRankId rank = MeshHostRankId{0}) {
        std::map<tt::tt_metal::AsicID, MeshHostRankId> result;
        for (const auto& asic : asics) {
            result[asic] = rank;
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Assertion helpers
    // -------------------------------------------------------------------------

    // Verify bidirectional mapping consistency
    static void verify_bidirectional_consistency(const TopologyMappingResult& result) {
        for (const auto& [node, asic] : result.fabric_node_to_asic) {
            ASSERT_TRUE(result.asic_to_fabric_node.count(asic) > 0)
                << "ASIC " << asic.get() << " not found in reverse mapping";
            EXPECT_EQ(result.asic_to_fabric_node.at(asic), node)
                << "Bidirectional mapping inconsistent for ASIC " << asic.get();
        }
        for (const auto& [asic, node] : result.asic_to_fabric_node) {
            ASSERT_TRUE(result.fabric_node_to_asic.count(node) > 0)
                << "Node not found in forward mapping";
            EXPECT_EQ(result.fabric_node_to_asic.at(node), asic) << "Bidirectional mapping inconsistent for node";
        }
    }

    // Verify that logical connectivity is preserved in physical mapping
    static void verify_connectivity_preserved(
        const TopologyMappingResult& result,
        const LogicalAdjacencyMap& logical_adj,
        const PhysicalAdjacencyMap& physical_adj) {
        for (const auto& [node, neighbors] : logical_adj) {
            const auto mapped_asic = result.fabric_node_to_asic.at(node);
            const auto& physical_neighbors = physical_adj.at(mapped_asic);

            for (const auto& neighbor : neighbors) {
                const auto neighbor_asic = result.fabric_node_to_asic.at(neighbor);
                bool found = std::find(physical_neighbors.begin(), physical_neighbors.end(), neighbor_asic) !=
                             physical_neighbors.end();
                EXPECT_TRUE(found) << "Logical edge not preserved in physical mapping";
            }
        }
    }

    // Verify rank constraints are satisfied
    static void verify_rank_constraints(
        const TopologyMappingResult& result,
        const std::map<FabricNodeId, MeshHostRankId>& node_ranks,
        const std::map<tt::tt_metal::AsicID, MeshHostRankId>& asic_ranks) {
        for (const auto& [node, asic] : result.fabric_node_to_asic) {
            EXPECT_EQ(node_ranks.at(node), asic_ranks.at(asic))
                << "Rank constraint violated: node mapped to ASIC on different rank";
        }
    }
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, EmptyGraph_ReturnsSuccess) {
    const LogicalAdjacencyMap logical_adj;
    const PhysicalAdjacencyMap physical_adj;
    const std::map<FabricNodeId, MeshHostRankId> node_ranks;
    const std::map<tt::tt_metal::AsicID, MeshHostRankId> asic_ranks;

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.error_message.empty());
    EXPECT_TRUE(result.fabric_node_to_asic.empty());
    EXPECT_TRUE(result.asic_to_fabric_node.empty());
}

TEST_F(TopologyMapperUtilsTest, SingleNode_MapsCorrectly) {
    const auto nodes = make_nodes(1);
    const auto asics = make_asics(1);

    LogicalAdjacencyMap logical_adj;
    logical_adj[nodes[0]] = {};

    PhysicalAdjacencyMap physical_adj;
    physical_adj[asics[0]] = {};

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), 1u);
    EXPECT_EQ(result.fabric_node_to_asic.at(nodes[0]), asics[0]);
    verify_bidirectional_consistency(result);
}

TEST_F(TopologyMapperUtilsTest, TwoConnectedNodes_MapsCorrectly) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), 2u);
    verify_bidirectional_consistency(result);
    verify_connectivity_preserved(result, logical_adj, physical_adj);
}

// =============================================================================
// Chain Topology Tests (exercises fast-path algorithm)
// =============================================================================

TEST_F(TopologyMapperUtilsTest, ChainTopology_FourNodes_PreservesConnectivity) {
    const auto nodes = make_nodes(4);
    const auto asics = make_asics(4);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), 4u);
    verify_bidirectional_consistency(result);
    verify_connectivity_preserved(result, logical_adj, physical_adj);
}

TEST_F(TopologyMapperUtilsTest, ChainTopology_EightNodes_PreservesConnectivity) {
    const auto nodes = make_nodes(8);
    const auto asics = make_asics(8);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), 8u);
    verify_bidirectional_consistency(result);
    verify_connectivity_preserved(result, logical_adj, physical_adj);
}

// =============================================================================
// Grid Topology Tests (exercises general DFS algorithm)
// =============================================================================

TEST_F(TopologyMapperUtilsTest, GridTopology_2x2_PreservesConnectivity) {
    constexpr size_t kRows = 2;
    constexpr size_t kCols = 2;
    const auto nodes = make_nodes(kRows * kCols);
    const auto asics = make_asics(kRows * kCols);

    const auto logical_adj = build_grid_adjacency(nodes, kRows, kCols);
    const auto physical_adj = build_grid_adjacency(asics, kRows, kCols);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), kRows * kCols);
    verify_bidirectional_consistency(result);
    verify_connectivity_preserved(result, logical_adj, physical_adj);
}

TEST_F(TopologyMapperUtilsTest, GridTopology_3x3_PreservesConnectivity) {
    constexpr size_t kRows = 3;
    constexpr size_t kCols = 3;
    const auto nodes = make_nodes(kRows * kCols);
    const auto asics = make_asics(kRows * kCols);

    const auto logical_adj = build_grid_adjacency(nodes, kRows, kCols);
    const auto physical_adj = build_grid_adjacency(asics, kRows, kCols);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), kRows * kCols);
    verify_bidirectional_consistency(result);
    verify_connectivity_preserved(result, logical_adj, physical_adj);
}

// =============================================================================
// Rank Constraint Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, MultiRank_NodesMapToCorrectRanks) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    // Assign different ranks
    std::map<FabricNodeId, MeshHostRankId> node_ranks;
    node_ranks[nodes[0]] = rank0_;
    node_ranks[nodes[1]] = rank1_;

    std::map<tt::tt_metal::AsicID, MeshHostRankId> asic_ranks;
    asic_ranks[asics[0]] = rank0_;
    asic_ranks[asics[1]] = rank1_;

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    verify_bidirectional_consistency(result);
    verify_rank_constraints(result, node_ranks, asic_ranks);
}

TEST_F(TopologyMapperUtilsTest, MultiRank_FourNodesAcrossTwoRanks) {
    const auto nodes = make_nodes(4);
    const auto asics = make_asics(4);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    // First two nodes on rank0, last two on rank1
    std::map<FabricNodeId, MeshHostRankId> node_ranks;
    std::map<tt::tt_metal::AsicID, MeshHostRankId> asic_ranks;
    for (size_t i = 0; i < 4; ++i) {
        const auto rank = (i < 2) ? rank0_ : rank1_;
        node_ranks[nodes[i]] = rank;
        asic_ranks[asics[i]] = rank;
    }

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    verify_rank_constraints(result, node_ranks, asic_ranks);
}

// =============================================================================
// Strict Mode Tests (channel count validation)
// =============================================================================

TEST_F(TopologyMapperUtilsTest, StrictMode_SufficientChannels_Succeeds) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    // Two logical channels between nodes
    LogicalAdjacencyMap logical_adj;
    logical_adj[nodes[0]] = {nodes[1], nodes[1]};
    logical_adj[nodes[1]] = {nodes[0], nodes[0]};

    // Two physical channels between ASICs
    PhysicalAdjacencyMap physical_adj;
    physical_adj[asics[0]] = {asics[1], asics[1]};
    physical_adj[asics[1]] = {asics[0], asics[0]};

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    TopologyMappingConfig config;
    config.strict_mode = true;

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    ASSERT_TRUE(result.success) << result.error_message;
}

TEST_F(TopologyMapperUtilsTest, StrictMode_InsufficientChannels_Fails) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    // Two logical channels required
    LogicalAdjacencyMap logical_adj;
    logical_adj[nodes[0]] = {nodes[1], nodes[1]};
    logical_adj[nodes[1]] = {nodes[0], nodes[0]};

    // Only one physical channel available
    PhysicalAdjacencyMap physical_adj;
    physical_adj[asics[0]] = {asics[1]};
    physical_adj[asics[1]] = {asics[0]};

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    TopologyMappingConfig config;
    config.strict_mode = true;

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    EXPECT_FALSE(result.success);
    EXPECT_THAT(result.error_message, ::testing::HasSubstr("channel"));
}

TEST_F(TopologyMapperUtilsTest, RelaxedMode_InsufficientChannels_Succeeds) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    // Two logical channels required
    LogicalAdjacencyMap logical_adj;
    logical_adj[nodes[0]] = {nodes[1], nodes[1]};
    logical_adj[nodes[1]] = {nodes[0], nodes[0]};

    // Only one physical channel available
    PhysicalAdjacencyMap physical_adj;
    physical_adj[asics[0]] = {asics[1]};
    physical_adj[asics[1]] = {asics[0]};

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    TopologyMappingConfig config;
    config.strict_mode = false;

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    ASSERT_TRUE(result.success) << result.error_message;
}

// =============================================================================
// Pinning Constraint Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, Pinning_SingleNodePinned_RespectsPinning) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    // Pin node0 to asic0's position
    const tt::tt_metal::TrayID tray{1};
    const tt::tt_metal::ASICLocation loc{0};
    const AsicPosition pos{tray, loc};

    TopologyMappingConfig config;
    config.pinnings.emplace_back(pos, nodes[0]);
    config.asic_positions[asics[0]] = pos;
    config.asic_positions[asics[1]] = {tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{1}};

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.at(nodes[0]), asics[0]) << "Pinning constraint not respected";
}

TEST_F(TopologyMapperUtilsTest, Pinning_InvalidPosition_Fails) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    // Pin to a position that doesn't exist
    const tt::tt_metal::TrayID nonexistent_tray{99};
    const tt::tt_metal::ASICLocation nonexistent_loc{99};
    const AsicPosition invalid_pos{nonexistent_tray, nonexistent_loc};

    TopologyMappingConfig config;
    config.pinnings.emplace_back(invalid_pos, nodes[0]);
    config.asic_positions[asics[0]] = {tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{0}};
    config.asic_positions[asics[1]] = {tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{1}};

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    EXPECT_FALSE(result.success);
    EXPECT_THAT(result.error_message, ::testing::HasSubstr("not found"));
}

TEST_F(TopologyMapperUtilsTest, Pinning_DuplicatePinningsSameNode_Fails) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    // Pin the same node to two different positions
    const AsicPosition pos1{tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{0}};
    const AsicPosition pos2{tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{1}};

    TopologyMappingConfig config;
    config.pinnings.emplace_back(pos1, nodes[0]);
    config.pinnings.emplace_back(pos2, nodes[0]);  // Same node, different position
    config.asic_positions[asics[0]] = pos1;
    config.asic_positions[asics[1]] = pos2;

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    EXPECT_FALSE(result.success);
    EXPECT_THAT(result.error_message, ::testing::HasSubstr("multiple"));
}

TEST_F(TopologyMapperUtilsTest, Pinning_NodeNotInMesh_Fails) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    // Pin a node that doesn't exist in the mesh
    const auto nonexistent_node = make_node(kDefaultMeshId, 999);
    const AsicPosition pos{tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{0}};

    TopologyMappingConfig config;
    config.pinnings.emplace_back(pos, nonexistent_node);
    config.asic_positions[asics[0]] = pos;
    config.asic_positions[asics[1]] = {tt::tt_metal::TrayID{1}, tt::tt_metal::ASICLocation{1}};

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks, config);

    EXPECT_FALSE(result.success);
    EXPECT_THAT(result.error_message, ::testing::HasSubstr("not found"));
}

// =============================================================================
// Failure Case Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, LogicalLargerThanPhysical_Fails) {
    const auto nodes = make_nodes(3);
    const auto asics = make_asics(2);  // Fewer ASICs than nodes

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    EXPECT_FALSE(result.success);
    EXPECT_THAT(result.error_message, ::testing::HasSubstr("larger"));
}

TEST_F(TopologyMapperUtilsTest, ConnectivityMismatch_TriangleToChain_Fails) {
    const auto nodes = make_nodes(3);
    const auto asics = make_asics(3);

    // Logical: fully connected triangle
    const auto logical_adj = build_clique_adjacency(nodes);

    // Physical: linear chain (can't form triangle)
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    EXPECT_FALSE(result.success);
}

TEST_F(TopologyMapperUtilsTest, RankMismatch_NoValidMapping_Fails) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(2);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    // All nodes on rank0, but all ASICs on rank1 - impossible to satisfy
    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank1_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    EXPECT_FALSE(result.success);
}

// =============================================================================
// Physical Larger Than Logical Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, PhysicalLargerThanLogical_SelectsValidSubset) {
    const auto nodes = make_nodes(2);
    const auto asics = make_asics(4);  // More ASICs than needed

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), 2u);
    verify_bidirectional_consistency(result);
    verify_connectivity_preserved(result, logical_adj, physical_adj);
}

// =============================================================================
// Stress/Edge Case Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, LargeChain_SixteenNodes_Succeeds) {
    constexpr size_t kNodeCount = 16;
    const auto nodes = make_nodes(kNodeCount);
    const auto asics = make_asics(kNodeCount);

    const auto logical_adj = build_chain_adjacency(nodes);
    const auto physical_adj = build_chain_adjacency(asics);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), kNodeCount);
    verify_bidirectional_consistency(result);
}

TEST_F(TopologyMapperUtilsTest, LargeGrid_4x4_Succeeds) {
    constexpr size_t kRows = 4;
    constexpr size_t kCols = 4;
    const auto nodes = make_nodes(kRows * kCols);
    const auto asics = make_asics(kRows * kCols);

    const auto logical_adj = build_grid_adjacency(nodes, kRows, kCols);
    const auto physical_adj = build_grid_adjacency(asics, kRows, kCols);

    const auto node_ranks = make_uniform_node_ranks(nodes, rank0_);
    const auto asic_ranks = make_uniform_asic_ranks(asics, rank0_);

    const auto result = map_mesh_to_physical(mesh_id_, logical_adj, physical_adj, node_ranks, asic_ranks);

    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_EQ(result.fabric_node_to_asic.size(), kRows * kCols);
    verify_bidirectional_consistency(result);
}

}  // namespace
}  // namespace tt::tt_metal::experimental::tt_fabric
