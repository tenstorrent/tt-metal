// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"

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
    std::vector<FabricNodeId> make_nodes(size_t count) const {
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
    std::map<FabricNodeId, MeshHostRankId> make_uniform_node_ranks(
        const std::vector<FabricNodeId>& nodes, MeshHostRankId rank = MeshHostRankId{0}) const {
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
            ASSERT_TRUE(result.asic_to_fabric_node.contains(asic))
                << "ASIC " << asic.get() << " not found in reverse mapping";
            EXPECT_EQ(result.asic_to_fabric_node.at(asic), node)
                << "Bidirectional mapping inconsistent for ASIC " << asic.get();
        }
        for (const auto& [asic, node] : result.asic_to_fabric_node) {
            ASSERT_TRUE(result.fabric_node_to_asic.contains(node)) << "Node not found in forward mapping";
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

// =============================================================================
// Multi-Mesh Graph Tests
// =============================================================================

TEST_F(TopologyMapperUtilsTest, BuildLogicalMultiMeshGraph_ClosetboxSuperpod) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_superpod_mgd.textproto";

    ::tt::tt_fabric::MeshGraph mesh_graph(mesh_graph_desc_path.string());
    const auto multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 4u);

    // Verify each mesh has correct structure and internal connectivity
    for (const auto& [mesh_id, adjacency_graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& internal_nodes = adjacency_graph.get_nodes();
        EXPECT_EQ(internal_nodes.size(), 8u);

        // Check a few internal connections
        FabricNodeId node0(mesh_id, 0);
        FabricNodeId node1(mesh_id, 1);
        FabricNodeId node4(mesh_id, 4);

        if (std::find(internal_nodes.begin(), internal_nodes.end(), node0) != internal_nodes.end()) {
            const auto& neighbors0 = adjacency_graph.get_neighbors(node0);
            EXPECT_GT(neighbors0.size(), 0u);
            if (std::find(internal_nodes.begin(), internal_nodes.end(), node1) != internal_nodes.end()) {
                EXPECT_TRUE(std::find(neighbors0.begin(), neighbors0.end(), node1) != neighbors0.end());
            }
            if (std::find(internal_nodes.begin(), internal_nodes.end(), node4) != internal_nodes.end()) {
                EXPECT_TRUE(std::find(neighbors0.begin(), neighbors0.end(), node4) != neighbors0.end());
            }
        }
    }

    // Verify ALL_TO_ALL inter-mesh connectivity
    // MGD specifies channels { count: 2 } for ALL_TO_ALL topology
    for (const auto& [mesh_id, _] : multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(mesh_id);
        if (neighbors.size() > 0) {
            EXPECT_EQ(neighbors.size(), 6u);
            std::unordered_set<MeshId> neighbor_ids;
            std::map<MeshId, uint32_t> neighbor_counts;
            for (const auto& n : neighbors) {
                EXPECT_NE(n, mesh_id);
                neighbor_ids.insert(n);
                neighbor_counts[n]++;
            }
            for (const auto& [other_mesh_id, _] : multi_mesh_graph.mesh_adjacency_graphs_) {
                if (other_mesh_id != mesh_id) {
                    EXPECT_TRUE(neighbor_ids.contains(other_mesh_id));
                    // Verify channel count: each mesh should appear 2 times (channels.count from MGD)
                    EXPECT_EQ(neighbor_counts.at(other_mesh_id), 2u);
                }
            }
        }
    }
}

TEST_F(TopologyMapperUtilsTest, BuildPhysicalMultiMeshGraph_MultiHostMultiMesh) {
    // ASCII diagram illustrating the ASIC connectivity structure:
    //
    //         Host0 (rank 0)
    //   +-------------------------+
    //   |  Mesh 0    |  Mesh 1    |
    //   |  1 === 2 -- 5 === 6     |
    //   |  ||   ||    ||   ||     |
    //   +--+---+------+---+-------+
    //      |   |      |   |
    //   +--+---+------+---+-------+
    //   |  ||   ||    ||   ||     |
    //   |  3 === 4 -- 7 === 8     |
    //   |  Mesh 0    |  Mesh 1    |
    //   +-------------------------+
    //         Host1 (rank 1)
    //
    // Legend:
    //   ===  : 2 local ethernet connections (within same host, same mesh pair)
    //   --   : 1 local ethernet connection (within same host, between mesh pairs)
    //   ||   : 2 cross-host ethernet connections (between hosts, same mesh)
    //
    // Mesh assignments:
    //   Mesh 0: ASICs 1, 2 (host0) <-> ASICs 3, 4 (host1)
    //   Mesh 1: ASICs 5, 6 (host0) <-> ASICs 7, 8 (host1)
    //
    // Connectivity details:
    //   - Local pairs (2 links): 1-2, 5-6, 3-4, 7-8
    //   - Cross-host pairs (2 links): 1-3, 2-4, 5-7, 6-8
    //   - Inter-group local (1 link): 2-5, 4-7

    // Load PSD from pre-written test file
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path psd_file_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_multihost_multimesh.textproto";

    // Verify the file exists
    ASSERT_TRUE(std::filesystem::exists(psd_file_path)) << "PSD test file not found: " << psd_file_path.string();

    // Load PhysicalSystemDescriptor from file
    tt::tt_metal::PhysicalSystemDescriptor physical_system_descriptor(psd_file_path.string());

    // Hand-craft the asic_id_to_mesh_rank mapping
    // Mesh 0: ASICs 1, 2 (host0) and ASICs 3, 4 (host1) - cross-host mesh
    // Mesh 1: ASICs 5, 6 (host0) and ASICs 7, 8 (host1) - cross-host mesh
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;

    // Mesh 0: spans both hosts
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{1}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{2}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{3}] = MeshHostRankId{1};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{4}] = MeshHostRankId{1};

    // Mesh 1: spans both hosts
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{5}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{6}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{7}] = MeshHostRankId{1};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{8}] = MeshHostRankId{1};

    // Build physical multi-mesh adjacency graph
    const auto multi_mesh_graph =
        build_physical_multi_mesh_adjacency_graph(physical_system_descriptor, asic_id_to_mesh_rank);

    // Verify we have 2 mesh nodes
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 2u) << "Should have 2 meshes";

    // Verify each mesh has correct structure and internal connectivity
    for (const auto& [mesh_id, adjacency_graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& internal_nodes = adjacency_graph.get_nodes();
        EXPECT_EQ(internal_nodes.size(), 4u) << "Each mesh should have 4 ASICs (2 per host)";

        // Verify internal connections within each mesh
        // Mesh 0: ASICs 1-2 (host0) and 3-4 (host1)
        // Mesh 1: ASICs 5-6 (host0) and 7-8 (host1)
        if (mesh_id == MeshId{0}) {
            // Check local connections on host0
            auto asic1 = tt::tt_metal::AsicID{1};
            auto asic2 = tt::tt_metal::AsicID{2};
            if (std::find(internal_nodes.begin(), internal_nodes.end(), asic1) != internal_nodes.end()) {
                const auto& neighbors1 = adjacency_graph.get_neighbors(asic1);
                EXPECT_GT(neighbors1.size(), 0u) << "ASIC 1 should have neighbors";
                EXPECT_TRUE(std::find(neighbors1.begin(), neighbors1.end(), asic2) != neighbors1.end())
                    << "ASIC 1 should be connected to ASIC 2";
            }

            // Check local connections on host1
            auto asic3 = tt::tt_metal::AsicID{3};
            auto asic4 = tt::tt_metal::AsicID{4};
            if (std::find(internal_nodes.begin(), internal_nodes.end(), asic3) != internal_nodes.end()) {
                const auto& neighbors3 = adjacency_graph.get_neighbors(asic3);
                EXPECT_GT(neighbors3.size(), 0u) << "ASIC 3 should have neighbors";
                EXPECT_TRUE(std::find(neighbors3.begin(), neighbors3.end(), asic4) != neighbors3.end())
                    << "ASIC 3 should be connected to ASIC 4";
            }
        } else if (mesh_id == MeshId{1}) {
            // Check local connections on host0
            auto asic5 = tt::tt_metal::AsicID{5};
            auto asic6 = tt::tt_metal::AsicID{6};
            if (std::find(internal_nodes.begin(), internal_nodes.end(), asic5) != internal_nodes.end()) {
                const auto& neighbors5 = adjacency_graph.get_neighbors(asic5);
                EXPECT_GT(neighbors5.size(), 0u) << "ASIC 5 should have neighbors";
                EXPECT_TRUE(std::find(neighbors5.begin(), neighbors5.end(), asic6) != neighbors5.end())
                    << "ASIC 5 should be connected to ASIC 6";
            }

            // Check local connections on host1
            auto asic7 = tt::tt_metal::AsicID{7};
            auto asic8 = tt::tt_metal::AsicID{8};
            if (std::find(internal_nodes.begin(), internal_nodes.end(), asic7) != internal_nodes.end()) {
                const auto& neighbors7 = adjacency_graph.get_neighbors(asic7);
                EXPECT_GT(neighbors7.size(), 0u) << "ASIC 7 should have neighbors";
                EXPECT_TRUE(std::find(neighbors7.begin(), neighbors7.end(), asic8) != neighbors7.end())
                    << "ASIC 7 should be connected to ASIC 8";
            }
        }
    }

    // Verify inter-mesh connectivity (if established)
    // Since meshes are on different hosts but connected via exit nodes,
    // they should be connected in the multi-mesh graph
    for (const auto& [mesh_id, _] : multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(mesh_id);
        // Each mesh should be connected to the other mesh via cross-host connections
        if (neighbors.size() > 0) {
            EXPECT_EQ(neighbors.size(), 2u) << "Each mesh should be connected to the other mesh";
            std::unordered_set<MeshId> neighbor_ids;
            for (const auto& n : neighbors) {
                EXPECT_NE(n, mesh_id) << "Mesh should not be connected to itself";
                neighbor_ids.insert(n);
            }
            // Verify bidirectional connectivity
            for (const auto& [other_mesh_id, _] : multi_mesh_graph.mesh_adjacency_graphs_) {
                if (other_mesh_id != mesh_id) {
                    EXPECT_TRUE(neighbor_ids.contains(other_mesh_id))
                        << "Mesh " << mesh_id.get() << " should be connected to mesh " << other_mesh_id.get();
                }
            }
        }
    }
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_TwoMeshes_Succeeds) {
    // Test the map_multi_mesh_to_physical function
    // This test manually creates both logical and physical multi-mesh graphs from adjacency maps
    // and verifies that the mapping function correctly maps logical meshes to physical meshes
    // and fabric nodes to ASICs within each mesh.
    //
    // Logical Topology (2 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 (line: 0-1)
    //
    // Physical Topology (3 meshes):
    //   Mesh 0: Two disconnected 1x2 chains (4 ASICs: 2 + 2)
    //   Mesh 1: 2x4 grid (8 ASICs)
    //   Mesh 2: 2x2 grid (4 ASICs)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 <-> Mesh 2 (line: 0-1-2)

    using namespace ::tt::tt_fabric;

    const MeshId logical_mesh0{0};
    const MeshId logical_mesh1{1};
    // Physical meshes: disconnected chains first (0), then grids (1,2)
    const MeshId physical_mesh0{0};  // Disconnected chains
    const MeshId physical_mesh1{1};  // 2x4 grid
    const MeshId physical_mesh2{2};  // 2x2 grid

    // =========================================================================
    // Create Logical Multi-Mesh Graph (2 meshes, both 2x2)
    // =========================================================================

    // Logical Mesh 0: 2x2 grid
    std::vector<FabricNodeId> logical_nodes_m0;
    for (uint32_t i = 0; i < 4; ++i) {
        logical_nodes_m0.push_back(FabricNodeId(logical_mesh0, i));
    }
    auto logical_adj_m0 = build_grid_adjacency(logical_nodes_m0, 2, 2);

    // Logical Mesh 1: 2x2 grid
    std::vector<FabricNodeId> logical_nodes_m1;
    for (uint32_t i = 0; i < 4; ++i) {
        logical_nodes_m1.push_back(FabricNodeId(logical_mesh1, i));
    }
    auto logical_adj_m1 = build_grid_adjacency(logical_nodes_m1, 2, 2);

    // Create logical multi-mesh graph
    LogicalMultiMeshGraph logical_multi_mesh_graph;
    logical_multi_mesh_graph.mesh_adjacency_graphs_[logical_mesh0] = AdjacencyGraph<FabricNodeId>(logical_adj_m0);
    logical_multi_mesh_graph.mesh_adjacency_graphs_[logical_mesh1] = AdjacencyGraph<FabricNodeId>(logical_adj_m1);

    // Create mesh-level adjacency map (line: 0-1)
    AdjacencyGraph<MeshId>::AdjacencyMap logical_mesh_level_adj_map;
    logical_mesh_level_adj_map[logical_mesh0] = {logical_mesh1};
    logical_mesh_level_adj_map[logical_mesh1] = {logical_mesh0};
    logical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(logical_mesh_level_adj_map);

    // =========================================================================
    // Create Physical Multi-Mesh Graph (3 meshes)
    // =========================================================================

    // Physical Mesh 0: Two disconnected 1x2 chains (4 ASICs total)
    // Chain 1: ASICs 300-301
    // Chain 2: ASICs 302-303
    PhysicalAdjacencyMap physical_adj_m0;
    tt::tt_metal::AsicID asic300{300};
    tt::tt_metal::AsicID asic301{301};
    tt::tt_metal::AsicID asic302{302};
    tt::tt_metal::AsicID asic303{303};

    // First disconnected chain: 300-301
    physical_adj_m0[asic300] = {asic301};
    physical_adj_m0[asic301] = {asic300};

    // Second disconnected chain: 302-303
    physical_adj_m0[asic302] = {asic303};
    physical_adj_m0[asic303] = {asic302};

    // Physical Mesh 1: 2x4 grid (8 ASICs)
    std::vector<tt::tt_metal::AsicID> physical_asics_m1;
    for (uint64_t i = 0; i < 8; ++i) {
        physical_asics_m1.push_back(tt::tt_metal::AsicID{100 + i});
    }
    auto physical_adj_m1 = build_grid_adjacency(physical_asics_m1, 2, 4);

    // Physical Mesh 2: 2x2 grid (4 ASICs)
    std::vector<tt::tt_metal::AsicID> physical_asics_m2;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m2.push_back(tt::tt_metal::AsicID{200 + i});
    }
    auto physical_adj_m2 = build_grid_adjacency(physical_asics_m2, 2, 2);

    // Create physical multi-mesh graph
    PhysicalMultiMeshGraph physical_multi_mesh_graph;
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh0] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m0);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh1] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m1);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh2] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m2);

    // Create mesh-level adjacency map (line: 0-1-2)
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;
    physical_mesh_level_adj_map[physical_mesh0] = {physical_mesh1};
    physical_mesh_level_adj_map[physical_mesh1] = {physical_mesh0, physical_mesh2};
    physical_mesh_level_adj_map[physical_mesh2] = {physical_mesh1};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

    // =========================================================================
    // Run mapping and verify results
    // =========================================================================

    // Create mapping config
    TopologyMappingConfig config;
    config.strict_mode = true;            // Use strict mode for testing
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    // Call map_multi_mesh_to_physical (rank mappings omitted since disable_rank_bindings is true)
    const auto results = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Verify results
    EXPECT_EQ(results.size(), 2u) << "Should have a result for each logical mesh";

    // Both logical meshes should succeed
    EXPECT_TRUE(results.at(logical_mesh0).success)
        << "Logical Mesh 0 mapping should succeed: " << results.at(logical_mesh0).error_message;
    EXPECT_TRUE(results.at(logical_mesh1).success)
        << "Logical Mesh 1 mapping should succeed: " << results.at(logical_mesh1).error_message;

    // Verify Mesh 0 mapping (2x2 logical -> should map to 2x2 physical mesh 1)
    const auto& result0 = results.at(logical_mesh0);
    verify_bidirectional_consistency(result0);
    EXPECT_EQ(result0.fabric_node_to_asic.size(), 4u) << "Logical Mesh 0 should map all 4 nodes";

    // Verify Mesh 1 mapping (2x2 logical -> should map to 2x2 physical mesh 1 or another suitable mesh)
    const auto& result1 = results.at(logical_mesh1);
    verify_bidirectional_consistency(result1);
    EXPECT_EQ(result1.fabric_node_to_asic.size(), 4u) << "Logical Mesh 1 should map all 4 nodes";

    // Verify connectivity for all successful mappings (rank constraints disabled)
    for (const auto& [mesh_id, result] : results) {
        if (result.success) {
            // Find which physical mesh this logical mesh mapped to
            MeshId physical_mesh_id = physical_mesh0;  // Default, will be updated
            if (!result.fabric_node_to_asic.empty()) {
                const auto& first_asic = result.fabric_node_to_asic.begin()->second;
                // Determine which physical mesh this ASIC belongs to by checking all physical meshes
                for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
                    const auto& nodes = adjacency_graph.get_nodes();
                    if (std::find(nodes.begin(), nodes.end(), first_asic) != nodes.end()) {
                        physical_mesh_id = pm_id;
                        break;
                    }
                }
            }

            // Verify connectivity is preserved
            const auto& logical_graph = logical_multi_mesh_graph.mesh_adjacency_graphs_.at(mesh_id);
            const auto& physical_graph = physical_multi_mesh_graph.mesh_adjacency_graphs_.at(physical_mesh_id);
            const auto& logical_nodes = logical_graph.get_nodes();

            for (const auto& node : logical_nodes) {
                const auto mapped_asic = result.fabric_node_to_asic.at(node);
                const auto& logical_neighbors = logical_graph.get_neighbors(node);
                const auto& physical_neighbors = physical_graph.get_neighbors(mapped_asic);

                for (const auto& neighbor : logical_neighbors) {
                    const auto neighbor_asic = result.fabric_node_to_asic.at(neighbor);
                    EXPECT_TRUE(
                        std::find(physical_neighbors.begin(), physical_neighbors.end(), neighbor_asic) !=
                        physical_neighbors.end())
                        << "Logical edge not preserved in physical mapping for logical mesh " << mesh_id.get();
                }
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental::tt_fabric
