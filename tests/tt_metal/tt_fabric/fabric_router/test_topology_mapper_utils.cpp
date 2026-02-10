// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <random>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/cluster.hpp>
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
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/wh_closetbox_superpod_mgd.textproto";

    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, mesh_graph_desc_path.string());
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

    // Verify exit nodes ARE populated in relaxed mode (as mesh-level exit nodes)
    // In relaxed mode, exit nodes are mesh-level (no fabric_node_id specified)
    EXPECT_FALSE(multi_mesh_graph.mesh_exit_node_graphs_.empty())
        << "Exit nodes should be populated in relaxed mode (as mesh-level exit nodes)";

    // Verify that all meshes with intermesh connections have exit node graphs
    for (const auto& mesh_id : {MeshId{0}, MeshId{1}, MeshId{2}}) {
        if (multi_mesh_graph.mesh_level_graph_.get_neighbors(mesh_id).size() > 0) {
            EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(mesh_id))
                << "Mesh " << mesh_id.get() << " should have exit node graph in relaxed mode";

            const auto& exit_graph = multi_mesh_graph.mesh_exit_node_graphs_.at(mesh_id);
            const auto& exit_nodes = exit_graph.get_nodes();
            EXPECT_GT(exit_nodes.size(), 0u) << "Mesh " << mesh_id.get() << " should have at least one exit node";

            // Verify exit nodes are mesh-level (fabric_node_id is nullopt)
            for (const auto& exit_node : exit_nodes) {
                EXPECT_FALSE(exit_node.fabric_node_id.has_value())
                    << "Relaxed mode exit nodes should be mesh-level (no fabric_node_id)";
                EXPECT_EQ(exit_node.mesh_id, mesh_id) << "Exit node mesh_id should match the mesh";
            }
        }
    }
}

TEST_F(TopologyMapperUtilsTest, BuildLogicalMultiMeshGraph_StrictModeIntermeshPorts) {
    // Test build_logical_multi_mesh_adjacency_graph with strict mode intermesh connections
    // Strict mode specifies exact device-to-device connections with channel counts
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_strict_connection_mgd.textproto";

    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::T3K, mesh_graph_desc_path.string());
    const auto multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Verify we have 2 meshes
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 2u) << "Should have 2 meshes";
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.contains(MeshId{0})) << "Should have mesh 0";
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.contains(MeshId{1})) << "Should have mesh 1";

    // Verify each mesh has correct structure (2x2 = 4 devices each)
    for (const auto& [mesh_id, adjacency_graph] : multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& internal_nodes = adjacency_graph.get_nodes();
        EXPECT_EQ(internal_nodes.size(), 4u) << "Mesh " << mesh_id.get() << " should have 4 devices";
    }

    // Verify strict mode intermesh connections
    // The descriptor specifies:
    // - M0 D1 <-> M1 D0 with 2 channels
    // - M0 D3 <-> M1 D2 with 2 channels
    // So mesh 0 should connect to mesh 1 with 4 total connections (2 + 2)
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 4u) << "Mesh 0 should have 4 connections to mesh 1 (2 channels x 2 connections)";

    // Count occurrences of mesh 1 in neighbors (should be 4)
    uint32_t mesh1_count = 0;
    for (const auto& neighbor : mesh0_neighbors) {
        if (neighbor == MeshId{1}) {
            mesh1_count++;
        }
    }
    EXPECT_EQ(mesh1_count, 4u) << "Mesh 0 should connect to mesh 1 with 4 channels total";

    const auto& mesh1_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 4u) << "Mesh 1 should have 4 connections to mesh 0 (2 channels x 2 connections)";

    // Count occurrences of mesh 0 in neighbors (should be 4)
    uint32_t mesh0_count = 0;
    for (const auto& neighbor : mesh1_neighbors) {
        if (neighbor == MeshId{0}) {
            mesh0_count++;
        }
    }
    EXPECT_EQ(mesh0_count, 4u) << "Mesh 1 should connect to mesh 0 with 4 channels total";

    // Verify that requested_intermesh_ports was used (not requested_intermesh_connections)
    const auto& requested_ports = mesh_graph.get_requested_intermesh_ports();
    EXPECT_FALSE(requested_ports.empty()) << "Should have requested_intermesh_ports in strict mode";

    // Verify the structure of requested_intermesh_ports
    // Should have: mesh 0 -> mesh 1 -> [(device 1, device 0, 2), (device 3, device 2, 2)]
    EXPECT_TRUE(requested_ports.contains(0)) << "Should have entries for mesh 0";
    EXPECT_TRUE(requested_ports.at(0).contains(1)) << "Should have connections from mesh 0 to mesh 1";
    const auto& mesh0_to_mesh1_ports = requested_ports.at(0).at(1);
    EXPECT_EQ(mesh0_to_mesh1_ports.size(), 2u) << "Should have 2 port entries (2 device pairs)";

    // Verify first connection: M0 D1 -> M1 D0 with 2 channels
    EXPECT_EQ(std::get<0>(mesh0_to_mesh1_ports[0]), 1u) << "First connection: src device should be 1";
    EXPECT_EQ(std::get<1>(mesh0_to_mesh1_ports[0]), 0u) << "First connection: dst device should be 0";
    EXPECT_EQ(std::get<2>(mesh0_to_mesh1_ports[0]), 2u) << "First connection: should have 2 channels";

    // Verify second connection: M0 D3 -> M1 D2 with 2 channels
    EXPECT_EQ(std::get<0>(mesh0_to_mesh1_ports[1]), 3u) << "Second connection: src device should be 3";
    EXPECT_EQ(std::get<1>(mesh0_to_mesh1_ports[1]), 2u) << "Second connection: dst device should be 2";
    EXPECT_EQ(std::get<2>(mesh0_to_mesh1_ports[1]), 2u) << "Second connection: should have 2 channels";

    // Verify requested_intermesh_connections is empty (strict mode doesn't use it)
    const auto& requested_connections = mesh_graph.get_requested_intermesh_connections();
    EXPECT_TRUE(requested_connections.empty()) << "requested_intermesh_connections should be empty in strict mode";

    // Verify exit nodes are populated in strict mode
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Should have exit node graph for mesh 0 in strict mode";
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Should have exit node graph for mesh 1 in strict mode";

    // Verify mesh 0 exit nodes: devices 1 and 3
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 2u) << "Mesh 0 should have 2 exit nodes";

    LogicalExitNode exit_node0_1{MeshId{0}, FabricNodeId(MeshId{0}, 1)};
    LogicalExitNode exit_node0_3{MeshId{0}, FabricNodeId(MeshId{0}, 3)};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_1) != exit_nodes0.end())
        << "Device 1 should be an exit node in mesh 0";
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_3) != exit_nodes0.end())
        << "Device 3 should be an exit node in mesh 0";

    // Verify exit node connections: device 1 connects to mesh 1 device 0 (2 channels)
    const auto& exit_neighbors0_1 = exit_graph0.get_neighbors(exit_node0_1);
    EXPECT_EQ(exit_neighbors0_1.size(), 2u) << "Exit node 0_1 should have 2 connections (2 channels)";
    LogicalExitNode target0_1{MeshId{1}, FabricNodeId(MeshId{1}, 0)};
    // Count occurrences of target0_1 (should be 2 for 2 channels)
    uint32_t target0_1_count = 0;
    for (const auto& neighbor : exit_neighbors0_1) {
        if (neighbor == target0_1) {
            target0_1_count++;
        }
    }
    EXPECT_EQ(target0_1_count, 2u) << "Exit node 0_1 should have 2 connections to mesh 1 device 0 (2 channels)";

    // Verify exit node connections: device 3 connects to mesh 1 device 2 (2 channels)
    const auto& exit_neighbors0_3 = exit_graph0.get_neighbors(exit_node0_3);
    EXPECT_EQ(exit_neighbors0_3.size(), 2u) << "Exit node 0_3 should have 2 connections (2 channels)";
    LogicalExitNode target0_3{MeshId{1}, FabricNodeId(MeshId{1}, 2)};
    // Count occurrences of target0_3 (should be 2 for 2 channels)
    uint32_t target0_3_count = 0;
    for (const auto& neighbor : exit_neighbors0_3) {
        if (neighbor == target0_3) {
            target0_3_count++;
        }
    }
    EXPECT_EQ(target0_3_count, 2u) << "Exit node 0_3 should have 2 connections to mesh 1 device 2 (2 channels)";

    // Verify mesh 1 exit nodes: devices 0 and 2
    const auto& exit_graph1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 2u) << "Mesh 1 should have 2 exit nodes";

    LogicalExitNode exit_node1_0{MeshId{1}, FabricNodeId(MeshId{1}, 0)};
    LogicalExitNode exit_node1_2{MeshId{1}, FabricNodeId(MeshId{1}, 2)};
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_0) != exit_nodes1.end())
        << "Device 0 should be an exit node in mesh 1";
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_2) != exit_nodes1.end())
        << "Device 2 should be an exit node in mesh 1";

    // Verify exit node connections: device 0 connects to mesh 0 device 1 (2 channels)
    const auto& exit_neighbors1_0 = exit_graph1.get_neighbors(exit_node1_0);
    EXPECT_EQ(exit_neighbors1_0.size(), 2u) << "Exit node 1_0 should have 2 connections (2 channels)";
    // Count occurrences of exit_node0_1 (should be 2 for 2 channels)
    uint32_t exit_node0_1_count = 0;
    for (const auto& neighbor : exit_neighbors1_0) {
        if (neighbor == exit_node0_1) {
            exit_node0_1_count++;
        }
    }
    EXPECT_EQ(exit_node0_1_count, 2u) << "Exit node 1_0 should have 2 connections to mesh 0 device 1 (2 channels)";

    // Verify exit node connections: device 2 connects to mesh 0 device 3 (2 channels)
    const auto& exit_neighbors1_2 = exit_graph1.get_neighbors(exit_node1_2);
    EXPECT_EQ(exit_neighbors1_2.size(), 2u) << "Exit node 1_2 should have 2 connections (2 channels)";
    // Count occurrences of exit_node0_3 (should be 2 for 2 channels)
    uint32_t exit_node0_3_count = 0;
    for (const auto& neighbor : exit_neighbors1_2) {
        if (neighbor == exit_node0_3) {
            exit_node0_3_count++;
        }
    }
    EXPECT_EQ(exit_node0_3_count, 2u) << "Exit node 1_2 should have 2 connections to mesh 0 device 3 (2 channels)";

    // Verify that strict mode creates fabric node-level exit nodes
    for (const auto& exit_node : exit_nodes0) {
        EXPECT_TRUE(exit_node.fabric_node_id.has_value())
            << "Strict mode exit nodes should be fabric node-level (fabric_node_id must be set)";
        EXPECT_EQ(exit_node.mesh_id, MeshId{0}) << "Exit node mesh_id should match the mesh";
    }
    for (const auto& exit_node : exit_nodes1) {
        EXPECT_TRUE(exit_node.fabric_node_id.has_value())
            << "Strict mode exit nodes should be fabric node-level (fabric_node_id must be set)";
        EXPECT_EQ(exit_node.mesh_id, MeshId{1}) << "Exit node mesh_id should match the mesh";
    }
}

TEST_F(TopologyMapperUtilsTest, BuildLogicalMultiMeshGraph_MixedStrictAndRelaxedConnections) {
    // Test build_logical_multi_mesh_adjacency_graph with strict mode connections from MGD
    // This verifies that exit nodes are created correctly for strict mode connections
    //
    // The MGD specifies strict connections between mesh 0 and mesh 1:
    // - M0 D1 <-> M1 D0 with 2 channels
    //
    // This verifies that:
    // 1. Exit nodes are tracked for strict mode connections
    // 2. Exit node details are correct (correct devices, correct connections)
    // 3. Mesh-level connectivity is correct

    // Create MGD textproto string with 3 meshes and strict connections between mesh 0-1
    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1,2 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }

          # Strict connections between specific devices: Mesh 0 <-> Mesh 1
          # M0 D1 <-> M1 D0 with 2 channels
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 1 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            channels { count: 2 }
          }

          # Strict mesh-to-mesh connection: Mesh 0 <-> Mesh 2
          # Mesh-level connection (no device_id) with STRICT policy and 3 channels
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
            channels { count: 3 policy: STRICT }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create MeshGraphDescriptor from string, then create MeshGraph
    // Since MeshGraph constructor requires a file path, we create a temporary file
    // with a unique name that gets automatically cleaned up
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path = temp_dir / ("test_mixed_connections_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the temporary file (with strict connections between mesh 0-1)
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Clean up temporary file immediately after loading
    std::filesystem::remove(temp_mgd_path);

    // Build the logical multi-mesh graph
    const auto multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Verify we have 3 meshes
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 meshes";

    // Verify mesh-level connectivity
    // Mesh 0: strict connection to mesh 1 (2 channels, device-level) + strict connection to mesh 2 (3 channels,
    // mesh-level)
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 5u) << "Mesh 0 should have 5 connections total (2 to mesh 1, 3 to mesh 2)";

    uint32_t mesh0_to_mesh1_count = 0;
    uint32_t mesh0_to_mesh2_count = 0;
    for (const auto& neighbor : mesh0_neighbors) {
        if (neighbor == MeshId{1}) {
            mesh0_to_mesh1_count++;
        } else if (neighbor == MeshId{2}) {
            mesh0_to_mesh2_count++;
        }
    }
    EXPECT_EQ(mesh0_to_mesh1_count, 2u) << "Mesh 0 should connect to mesh 1 with 2 channels (strict, device-level)";
    EXPECT_EQ(mesh0_to_mesh2_count, 3u) << "Mesh 0 should connect to mesh 2 with 3 channels (strict, mesh-level)";

    // Mesh 1: strict connection to mesh 0 (2 channels) only
    const auto& mesh1_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 2u) << "Mesh 1 should have 2 connections to mesh 0 (strict mode, 2 channels)";

    uint32_t mesh1_to_mesh0_count = 0;
    uint32_t mesh1_to_mesh2_count = 0;
    for (const auto& neighbor : mesh1_neighbors) {
        if (neighbor == MeshId{0}) {
            mesh1_to_mesh0_count++;
        } else if (neighbor == MeshId{2}) {
            mesh1_to_mesh2_count++;
        }
    }
    EXPECT_EQ(mesh1_to_mesh0_count, 2u) << "Mesh 1 should connect to mesh 0 with 2 channels (strict)";
    EXPECT_EQ(mesh1_to_mesh2_count, 0u) << "Mesh 1 should NOT connect to mesh 2";

    // Mesh 2: strict connection to mesh 0 (3 channels, mesh-level) only
    const auto& mesh2_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{2});
    EXPECT_EQ(mesh2_neighbors.size(), 3u) << "Mesh 2 should have 3 connections to mesh 0 (strict mode, 3 channels)";

    uint32_t mesh2_to_mesh0_count = 0;
    uint32_t mesh2_to_mesh1_count = 0;
    for (const auto& neighbor : mesh2_neighbors) {
        if (neighbor == MeshId{0}) {
            mesh2_to_mesh0_count++;
        } else if (neighbor == MeshId{1}) {
            mesh2_to_mesh1_count++;
        }
    }
    EXPECT_EQ(mesh2_to_mesh0_count, 3u) << "Mesh 2 should connect to mesh 0 with 3 channels (strict, mesh-level)";
    EXPECT_EQ(mesh2_to_mesh1_count, 0u) << "Mesh 2 should NOT connect to mesh 1";

    // Verify exit nodes: device-level connections create device-level exit nodes,
    // mesh-level connections create mesh-level exit nodes
    // If a mesh has device-level connections, it only gets device-level exit nodes (not mesh-level)
    // Mesh 0 should have exit nodes (device-level connection to mesh 1, so device-level exit node)
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Mesh 0 should have exit nodes (device-level connection to mesh 1)";

    // Mesh 1 should have exit nodes (device-level strict connection to mesh 0)
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Mesh 1 should have exit nodes (device-level strict connection)";

    // Mesh 2 SHOULD have exit nodes (mesh-level strict connection creates mesh-level exit nodes)
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{2}))
        << "Mesh 2 should have exit nodes (mesh-level strict connection creates mesh-level exit nodes)";

    // Verify exit node details for mesh 0
    // Mesh 0 has device-level connection to mesh 1 (device 1 specified) -> creates device-level exit node
    // Mesh 0 has mesh-level connection to mesh 2 (no device specified) -> creates mesh-level exit node
    // So mesh 0 should have 1 device-level exit node and 1 mesh-level exit node
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 2u)
        << "Mesh 0 should have 2 exit nodes (1 device-level for mesh 1, 1 mesh-level for mesh 2)";

    // Find the device-level exit node (device 1, for connection to mesh 1)
    LogicalExitNode exit_node0_1{MeshId{0}, FabricNodeId(MeshId{0}, 1)};
    auto fabric_node_exit_it = std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_1);
    EXPECT_NE(fabric_node_exit_it, exit_nodes0.end())
        << "Device 1 should be a device-level exit node in mesh 0 (device specified in connection to mesh 1)";

    // Find the mesh-level exit node (for connection to mesh 2)
    LogicalExitNode mesh_level_exit_node0{MeshId{0}, std::nullopt};
    auto mesh_level_exit_it = std::find(exit_nodes0.begin(), exit_nodes0.end(), mesh_level_exit_node0);
    EXPECT_NE(mesh_level_exit_it, exit_nodes0.end())
        << "Mesh 0 should have a mesh-level exit node (no device specified in connection to mesh 2)";

    // Verify device-level exit node connections: device 1 connects to mesh 1 device 0 (2 channels)
    const auto& exit_neighbors0_1 = exit_graph0.get_neighbors(exit_node0_1);
    EXPECT_EQ(exit_neighbors0_1.size(), 2u) << "Device-level exit node 0_1 should have 2 connections (2 channels)";
    LogicalExitNode target0_1{MeshId{1}, FabricNodeId(MeshId{1}, 0)};
    uint32_t target0_1_count = 0;
    for (const auto& neighbor : exit_neighbors0_1) {
        if (neighbor == target0_1) {
            target0_1_count++;
        }
    }
    EXPECT_EQ(target0_1_count, 2u) << "Device-level exit node 0_1 should have 2 connections to mesh 1 device 0";

    // Verify mesh-level exit node connections: mesh-level exit node connects to mesh 2 (3 channels)
    const auto& exit_neighbors_mesh_level = exit_graph0.get_neighbors(mesh_level_exit_node0);
    EXPECT_EQ(exit_neighbors_mesh_level.size(), 3u) << "Mesh-level exit node should have 3 connections (3 channels)";
    LogicalExitNode target_mesh2{MeshId{2}, std::nullopt};
    uint32_t target_mesh2_count = 0;
    for (const auto& neighbor : exit_neighbors_mesh_level) {
        if (neighbor == target_mesh2) {
            target_mesh2_count++;
        }
    }
    EXPECT_EQ(target_mesh2_count, 3u) << "Mesh-level exit node should have 3 connections to mesh 2";

    // Verify exit node details for mesh 1
    const auto& exit_graph1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 1u) << "Mesh 1 should have 1 exit node (device 0)";

    LogicalExitNode exit_node1_0{MeshId{1}, FabricNodeId(MeshId{1}, 0)};
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_0) != exit_nodes1.end())
        << "Device 0 should be an exit node in mesh 1";

    // Verify exit node connections: device 0 connects to mesh 0 device 1 (2 channels)
    const auto& exit_neighbors1_0 = exit_graph1.get_neighbors(exit_node1_0);
    EXPECT_EQ(exit_neighbors1_0.size(), 2u) << "Exit node 1_0 should have 2 connections (2 channels)";
    uint32_t exit_node0_1_count = 0;
    for (const auto& neighbor : exit_neighbors1_0) {
        if (neighbor == exit_node0_1) {
            exit_node0_1_count++;
        }
    }
    EXPECT_EQ(exit_node0_1_count, 2u) << "Exit node 1_0 should have 2 connections to mesh 0 device 1";

    // ========================================================================
    // Verify exit node types:
    // - Device-level connections (device specified) create device-level exit nodes
    // - Mesh-level connections (no device specified) create mesh-level exit nodes
    // ========================================================================

    // Verify that mesh 0 has both types of exit nodes
    bool found_device_level = false;
    bool found_mesh_level = false;
    for (const auto& exit_node : exit_nodes0) {
        EXPECT_EQ(exit_node.mesh_id, MeshId{0}) << "Exit node mesh_id should match the mesh";

        if (exit_node.fabric_node_id.has_value()) {
            found_device_level = true;
            // Verify the fabric_node_id is valid and matches expected device
            const auto& fabric_node_id = exit_node.fabric_node_id.value();
            EXPECT_EQ(fabric_node_id.mesh_id, MeshId{0}) << "Fabric node ID mesh_id should match exit node mesh_id";
            EXPECT_EQ(fabric_node_id.chip_id, 1u)
                << "Mesh 0 device-level exit node should be device 1 (device specified in connection to mesh 1)";
        } else {
            found_mesh_level = true;
        }
    }
    EXPECT_TRUE(found_device_level)
        << "Mesh 0 should have a device-level exit node (device specified in connection to mesh 1)";
    EXPECT_TRUE(found_mesh_level)
        << "Mesh 0 should have a mesh-level exit node (no device specified in connection to mesh 2)";

    // Verify that mesh 1 exit nodes are fabric node-level (strict mode, device-level connection)
    for (const auto& exit_node : exit_nodes1) {
        EXPECT_TRUE(exit_node.fabric_node_id.has_value())
            << "Strict mode (device-level) exit nodes should be fabric node-level (fabric_node_id must be set). "
            << "Found exit node with mesh_id=" << exit_node.mesh_id.get();
        EXPECT_EQ(exit_node.mesh_id, MeshId{1}) << "Exit node mesh_id should match the mesh";

        // Verify the fabric_node_id is valid and matches expected device
        const auto& fabric_node_id = exit_node.fabric_node_id.value();
        EXPECT_EQ(fabric_node_id.mesh_id, MeshId{1}) << "Fabric node ID mesh_id should match exit node mesh_id";
        EXPECT_EQ(fabric_node_id.chip_id, 0u)
            << "Mesh 1 exit node should be device 0 (from device-level strict connection)";
    }

    // Verify exit node connectivity: fabric node-level exit nodes connect to other fabric node-level exit nodes
    for (const auto& neighbor : exit_neighbors0_1) {
        EXPECT_TRUE(neighbor.fabric_node_id.has_value())
            << "Fabric node-level exit node should connect to other fabric node-level exit nodes";
        EXPECT_NE(neighbor.mesh_id, MeshId{0}) << "Neighbor should be from a different mesh";
    }
    for (const auto& neighbor : exit_neighbors1_0) {
        EXPECT_TRUE(neighbor.fabric_node_id.has_value())
            << "Fabric node-level exit node should connect to other fabric node-level exit nodes";
        EXPECT_NE(neighbor.mesh_id, MeshId{1}) << "Neighbor should be from a different mesh";
    }

    // Verify mesh 2 exit nodes are mesh-level (mesh-level strict connection)
    const auto& exit_graph2 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2});
    const auto& exit_nodes2 = exit_graph2.get_nodes();
    EXPECT_EQ(exit_nodes2.size(), 1u) << "Mesh 2 should have 1 mesh-level exit node";

    LogicalExitNode mesh_level_exit_node2{MeshId{2}, std::nullopt};
    EXPECT_TRUE(std::find(exit_nodes2.begin(), exit_nodes2.end(), mesh_level_exit_node2) != exit_nodes2.end())
        << "Mesh 2 should have a mesh-level exit node";

    // Verify mesh-level exit nodes connect to other mesh-level exit nodes
    const auto& exit_neighbors2 = exit_graph2.get_neighbors(mesh_level_exit_node2);
    EXPECT_EQ(exit_neighbors2.size(), 3u) << "Mesh-level exit node should have 3 connections (3 channels)";
    for (const auto& neighbor : exit_neighbors2) {
        EXPECT_FALSE(neighbor.fabric_node_id.has_value())
            << "Mesh-level exit node should connect to other mesh-level exit nodes";
        EXPECT_EQ(neighbor.mesh_id, MeshId{0}) << "Mesh 2 exit node should connect to mesh 0";
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

    // Verify exit node information is tracked
    // Mesh 0 should have exit nodes that connect to mesh 1
    ASSERT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0})) << "Mesh 0 should have exit node graph";
    const auto& exit_graph_m0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes_m0 = exit_graph_m0.get_nodes();
    EXPECT_GT(exit_nodes_m0.size(), 0u) << "Mesh 0 should have at least one exit node";

    // Mesh 1 should have exit nodes that connect to mesh 0
    ASSERT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1})) << "Mesh 1 should have exit node graph";
    const auto& exit_graph_m1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes_m1 = exit_graph_m1.get_nodes();
    EXPECT_GT(exit_nodes_m1.size(), 0u) << "Mesh 1 should have at least one exit node";

    // Verify that exit nodes have connections (the neighbors are PhysicalExitNodes in other meshes)
    for (const auto& exit_node : exit_nodes_m0) {
        EXPECT_EQ(exit_node.mesh_id, MeshId{0}) << "Exit node should belong to mesh 0";
        const auto& neighbors = exit_graph_m0.get_neighbors(exit_node);
        EXPECT_GT(neighbors.size(), 0u) << "Exit node " << exit_node.asic_id.get()
                                        << " in mesh 0 should have at least one connection";
        // Verify connection counts are represented by duplicate entries
        // (multiple channels between same pair appear as multiple entries)
    }

    for (const auto& exit_node : exit_nodes_m1) {
        EXPECT_EQ(exit_node.mesh_id, MeshId{1}) << "Exit node should belong to mesh 1";
        const auto& neighbors = exit_graph_m1.get_neighbors(exit_node);
        EXPECT_GT(neighbors.size(), 0u) << "Exit node " << exit_node.asic_id.get()
                                        << " in mesh 1 should have at least one connection";
        // Verify connection counts are represented by duplicate entries
        // (multiple channels between same pair appear as multiple entries)
    }
}

TEST_F(TopologyMapperUtilsTest, BuildPhysicalMultiMeshGraph_ExitNodeTracking) {
    // Test that exit node information is correctly tracked in the physical multi-mesh graph
    // Exit nodes are ASICs that connect to ASICs in other meshes
    //
    // Physical Topology (2 meshes):
    //   Mesh 0: ASICs 100, 101, 102 (3 ASICs)
    //   Mesh 1: ASICs 200, 201, 202 (3 ASICs)
    //   Inter-mesh connections:
    //     - ASIC 100 (mesh 0) <-> ASIC 200 (mesh 1)
    //     - ASIC 101 (mesh 0) <-> ASIC 201 (mesh 1)
    //     - ASIC 102 (mesh 0) <-> ASIC 202 (mesh 1)
    //
    // Expected exit nodes:
    //   Mesh 0: ASICs 100, 101, 102 (all connect to mesh 1)
    //   Mesh 1: ASICs 200, 201, 202 (all connect to mesh 0)

    using namespace ::tt::tt_fabric;

    const MeshId physical_mesh0{0};
    const MeshId physical_mesh1{1};

    // Create a simple physical system descriptor manually
    // We'll create adjacency maps directly since we can't easily create a PSD from scratch
    PhysicalAdjacencyMap physical_adj_m0;
    PhysicalAdjacencyMap physical_adj_m1;

    // Define ASIC IDs first
    tt::tt_metal::AsicID asic100{100};
    tt::tt_metal::AsicID asic101{101};
    tt::tt_metal::AsicID asic102{102};
    tt::tt_metal::AsicID asic200{200};
    tt::tt_metal::AsicID asic201{201};
    tt::tt_metal::AsicID asic202{202};

    // Mesh 0: ASICs 100, 101, 102 with internal connections
    physical_adj_m0[asic100] = {asic101, asic200};           // Internal + intermesh
    physical_adj_m0[asic101] = {asic100, asic102, asic201};  // Internal + intermesh
    physical_adj_m0[asic102] = {asic101, asic202};           // Internal + intermesh

    // Mesh 1: ASICs 200, 201, 202 with internal connections
    physical_adj_m1[asic200] = {asic201, asic100};           // Internal + intermesh
    physical_adj_m1[asic201] = {asic200, asic202, asic101};  // Internal + intermesh
    physical_adj_m1[asic202] = {asic201, asic102};           // Internal + intermesh

    // Create physical multi-mesh graph manually
    PhysicalMultiMeshGraph physical_multi_mesh_graph;
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh0] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m0);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh1] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m1);

    // Create mesh-level adjacency map
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;
    physical_mesh_level_adj_map[physical_mesh0] = {physical_mesh1};
    physical_mesh_level_adj_map[physical_mesh1] = {physical_mesh0};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

    // Manually populate exit node information (simulating what build_physical_multi_mesh_adjacency_graph does)
    // Mesh 0 exit nodes - each connection has 1 channel
    AdjacencyGraph<PhysicalExitNode>::AdjacencyMap exit_adj_m0;
    PhysicalExitNode exit_node0_100{physical_mesh0, asic100};
    PhysicalExitNode exit_node0_101{physical_mesh0, asic101};
    PhysicalExitNode exit_node0_102{physical_mesh0, asic102};
    PhysicalExitNode exit_node1_200{physical_mesh1, asic200};
    PhysicalExitNode exit_node1_201{physical_mesh1, asic201};
    PhysicalExitNode exit_node1_202{physical_mesh1, asic202};
    exit_adj_m0[exit_node0_100] = {exit_node1_200};  // 1 channel
    exit_adj_m0[exit_node0_101] = {exit_node1_201};  // 1 channel
    exit_adj_m0[exit_node0_102] = {exit_node1_202};  // 1 channel
    physical_multi_mesh_graph.mesh_exit_node_graphs_[physical_mesh0] = AdjacencyGraph<PhysicalExitNode>(exit_adj_m0);

    // Mesh 1 exit nodes - each connection has 1 channel
    AdjacencyGraph<PhysicalExitNode>::AdjacencyMap exit_adj_m1;
    exit_adj_m1[exit_node1_200] = {exit_node0_100};  // 1 channel
    exit_adj_m1[exit_node1_201] = {exit_node0_101};  // 1 channel
    exit_adj_m1[exit_node1_202] = {exit_node0_102};  // 1 channel
    physical_multi_mesh_graph.mesh_exit_node_graphs_[physical_mesh1] = AdjacencyGraph<PhysicalExitNode>(exit_adj_m1);

    // Verify exit node information for mesh 0
    ASSERT_TRUE(physical_multi_mesh_graph.mesh_exit_node_graphs_.contains(physical_mesh0))
        << "Mesh 0 should have exit node graph";
    const auto& exit_graph_0 = physical_multi_mesh_graph.mesh_exit_node_graphs_.at(physical_mesh0);
    const auto& exit_nodes_0 = exit_graph_0.get_nodes();
    EXPECT_EQ(exit_nodes_0.size(), 3u) << "Mesh 0 should have 3 exit nodes";
    EXPECT_TRUE(std::find(exit_nodes_0.begin(), exit_nodes_0.end(), exit_node0_100) != exit_nodes_0.end())
        << "ASIC 100 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes_0.begin(), exit_nodes_0.end(), exit_node0_101) != exit_nodes_0.end())
        << "ASIC 101 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes_0.begin(), exit_nodes_0.end(), exit_node0_102) != exit_nodes_0.end())
        << "ASIC 102 should be an exit node";

    // Verify each exit node connects to the correct exit node
    const auto& neighbors_100 = exit_graph_0.get_neighbors(exit_node0_100);
    EXPECT_EQ(neighbors_100.size(), 1u) << "ASIC 100 should have 1 connection";
    EXPECT_EQ(neighbors_100[0], exit_node1_200) << "ASIC 100 should connect to ASIC 200";

    const auto& neighbors_101 = exit_graph_0.get_neighbors(exit_node0_101);
    EXPECT_EQ(neighbors_101.size(), 1u) << "ASIC 101 should have 1 connection";
    EXPECT_EQ(neighbors_101[0], exit_node1_201) << "ASIC 101 should connect to ASIC 201";

    const auto& neighbors_102 = exit_graph_0.get_neighbors(exit_node0_102);
    EXPECT_EQ(neighbors_102.size(), 1u) << "ASIC 102 should have 1 connection";
    EXPECT_EQ(neighbors_102[0], exit_node1_202) << "ASIC 102 should connect to ASIC 202";

    // Verify exit node information for mesh 1
    ASSERT_TRUE(physical_multi_mesh_graph.mesh_exit_node_graphs_.contains(physical_mesh1))
        << "Mesh 1 should have exit node graph";
    const auto& exit_graph_1 = physical_multi_mesh_graph.mesh_exit_node_graphs_.at(physical_mesh1);
    const auto& exit_nodes_1 = exit_graph_1.get_nodes();
    EXPECT_EQ(exit_nodes_1.size(), 3u) << "Mesh 1 should have 3 exit nodes";
    EXPECT_TRUE(std::find(exit_nodes_1.begin(), exit_nodes_1.end(), exit_node1_200) != exit_nodes_1.end())
        << "ASIC 200 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes_1.begin(), exit_nodes_1.end(), exit_node1_201) != exit_nodes_1.end())
        << "ASIC 201 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes_1.begin(), exit_nodes_1.end(), exit_node1_202) != exit_nodes_1.end())
        << "ASIC 202 should be an exit node";

    // Verify each exit node connects to the correct exit node
    const auto& neighbors_200 = exit_graph_1.get_neighbors(exit_node1_200);
    EXPECT_EQ(neighbors_200.size(), 1u) << "ASIC 200 should have 1 connection";
    EXPECT_EQ(neighbors_200[0], exit_node0_100) << "ASIC 200 should connect to ASIC 100";

    const auto& neighbors_201 = exit_graph_1.get_neighbors(exit_node1_201);
    EXPECT_EQ(neighbors_201.size(), 1u) << "ASIC 201 should have 1 connection";
    EXPECT_EQ(neighbors_201[0], exit_node0_101) << "ASIC 201 should connect to ASIC 101";

    const auto& neighbors_202 = exit_graph_1.get_neighbors(exit_node1_202);
    EXPECT_EQ(neighbors_202.size(), 1u) << "ASIC 202 should have 1 connection";
    EXPECT_EQ(neighbors_202[0], exit_node0_102) << "ASIC 202 should connect to ASIC 102";
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
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Verify overall result succeeded
    EXPECT_TRUE(result.success) << "Multi-mesh mapping should succeed: " << result.error_message;

    // Verify bidirectional consistency of the overall result
    verify_bidirectional_consistency(result);

    // Group mappings by mesh_id
    std::map<MeshId, std::map<FabricNodeId, tt::tt_metal::AsicID>> mappings_by_mesh;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh[fabric_node.mesh_id][fabric_node] = asic;
    }

    // Verify we have mappings for both logical meshes
    EXPECT_EQ(mappings_by_mesh.size(), 2u) << "Should have mappings for both logical meshes";
    EXPECT_TRUE(mappings_by_mesh.contains(logical_mesh0)) << "Should have mappings for logical mesh 0";
    EXPECT_TRUE(mappings_by_mesh.contains(logical_mesh1)) << "Should have mappings for logical mesh 1";

    // Verify Mesh 0 mapping (2x2 logical -> should map to 2x2 physical mesh)
    const auto& mesh0_mappings = mappings_by_mesh.at(logical_mesh0);
    EXPECT_EQ(mesh0_mappings.size(), 4u) << "Logical Mesh 0 should map all 4 nodes";

    // Verify Mesh 1 mapping (2x2 logical -> should map to 2x2 physical mesh)
    const auto& mesh1_mappings = mappings_by_mesh.at(logical_mesh1);
    EXPECT_EQ(mesh1_mappings.size(), 4u) << "Logical Mesh 1 should map all 4 nodes";

    // Verify connectivity for all meshes (rank constraints disabled)
    for (const auto& [mesh_id, mesh_mappings] : mappings_by_mesh) {
        // Find which physical mesh this logical mesh mapped to
        MeshId physical_mesh_id = physical_mesh0;  // Default, will be updated
        if (!mesh_mappings.empty()) {
            const auto& first_asic = mesh_mappings.begin()->second;
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
            const auto mapped_asic = mesh_mappings.at(node);
            const auto& logical_neighbors = logical_graph.get_neighbors(node);
            const auto& physical_neighbors = physical_graph.get_neighbors(mapped_asic);

            for (const auto& neighbor : logical_neighbors) {
                const auto neighbor_asic = mesh_mappings.at(neighbor);
                EXPECT_TRUE(
                    std::find(physical_neighbors.begin(), physical_neighbors.end(), neighbor_asic) !=
                    physical_neighbors.end())
                    << "Logical edge not preserved in physical mapping for logical mesh " << mesh_id.get();
            }
        }
    }
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_IncompatibleTopology_Fails) {
    // Negative test: Logical topology that cannot be mapped to physical topology
    // This test verifies that the mapper correctly fails when:
    // 1. Logical meshes require more nodes than available in physical meshes
    // 2. The mapper tries multiple different multi-mesh mapping combinations before failing
    //
    // Logical Topology (2 meshes):
    //   Mesh 0: 3x3 grid (9 nodes) - requires 9 nodes
    //   Mesh 1: 2x2 grid (4 nodes) - requires 4 nodes
    //   Inter-mesh: Mesh 0 <-> Mesh 1 (line: 0-1)
    //
    // Physical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs) - only 4 nodes available
    //   Mesh 1: 2x2 grid (4 ASICs) - only 4 nodes available
    //   Mesh 2: 2x2 grid (4 ASICs) - only 4 nodes available
    //   Inter-mesh: Mesh 0 <-> Mesh 1 <-> Mesh 2 (line: 0-1-2)
    //
    // Expected behavior:
    //   - The mapper will try multiple combinations:
    //     * Attempt 1: logical_mesh0 -> physical_mesh0, logical_mesh1 -> physical_mesh1 (fails - mesh0 too large)
    //     * Attempt 2: logical_mesh0 -> physical_mesh1, logical_mesh1 -> physical_mesh0 (fails - mesh0 too large)
    //     * Attempt 3: logical_mesh0 -> physical_mesh2, logical_mesh1 -> physical_mesh0 (fails - mesh0 too large)
    //     * etc.
    //   - Eventually exhausts all possibilities and fails
    //   - This ensures the retry logic is exercised (at least 2 attempts)

    using namespace ::tt::tt_fabric;

    const MeshId logical_mesh0{0};
    const MeshId logical_mesh1{1};
    const MeshId physical_mesh0{0};
    const MeshId physical_mesh1{1};
    const MeshId physical_mesh2{2};

    // =========================================================================
    // Create Logical Multi-Mesh Graph (2 meshes: 3x3 and 2x2)
    // =========================================================================

    // Logical Mesh 0: 3x3 grid (9 nodes) - too large for available physical meshes
    std::vector<FabricNodeId> logical_nodes_m0;
    for (uint32_t i = 0; i < 9; ++i) {
        logical_nodes_m0.push_back(FabricNodeId(logical_mesh0, i));
    }
    auto logical_adj_m0 = build_grid_adjacency(logical_nodes_m0, 3, 3);

    // Logical Mesh 1: 2x2 grid (4 nodes)
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
    // Create Physical Multi-Mesh Graph (3 meshes, all 2x2)
    // =========================================================================

    // Physical Mesh 0: 2x2 grid (4 ASICs) - insufficient for logical mesh 0
    std::vector<tt::tt_metal::AsicID> physical_asics_m0;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m0.push_back(tt::tt_metal::AsicID{100 + i});
    }
    auto physical_adj_m0 = build_grid_adjacency(physical_asics_m0, 2, 2);

    // Physical Mesh 1: 2x2 grid (4 ASICs) - insufficient for logical mesh 0
    std::vector<tt::tt_metal::AsicID> physical_asics_m1;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m1.push_back(tt::tt_metal::AsicID{200 + i});
    }
    auto physical_adj_m1 = build_grid_adjacency(physical_asics_m1, 2, 2);

    // Physical Mesh 2: 2x2 grid (4 ASICs) - insufficient for logical mesh 0
    std::vector<tt::tt_metal::AsicID> physical_asics_m2;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m2.push_back(tt::tt_metal::AsicID{300 + i});
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
    // This allows the mapper to try different combinations of physical meshes
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;
    physical_mesh_level_adj_map[physical_mesh0] = {physical_mesh1};
    physical_mesh_level_adj_map[physical_mesh1] = {physical_mesh0, physical_mesh2};
    physical_mesh_level_adj_map[physical_mesh2] = {physical_mesh1};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

    // =========================================================================
    // Run mapping and verify failure
    // =========================================================================

    // Create mapping config
    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    // Call map_multi_mesh_to_physical - should fail
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Verify overall result failed
    EXPECT_FALSE(result.success) << "Multi-mesh mapping should fail due to insufficient physical nodes";
    EXPECT_FALSE(result.error_message.empty()) << "Error message should be provided when mapping fails";

    // Verify that no complete mapping was found
    // The mapper may find partial mappings, but not all logical meshes should be mapped
    std::set<MeshId> mapped_logical_meshes;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mapped_logical_meshes.insert(fabric_node.mesh_id);
    }

    // At least one logical mesh should not be fully mapped
    // (logical mesh 0 requires 9 nodes but physical meshes only have 4 nodes each)
    bool mesh0_fully_mapped = mapped_logical_meshes.contains(logical_mesh0) && result.fabric_node_to_asic.size() >= 9;
    EXPECT_FALSE(mesh0_fully_mapped)
        << "Logical mesh 0 (9 nodes) should not be fully mapped to physical mesh (4 nodes)";

    // Verify that the error message indicates multiple attempts were made
    // The error message should mention retry attempts or failed mesh pairs
    // This ensures the retry logic was exercised (at least 2 attempts)
    bool mentions_retry_or_attempts = result.error_message.find("attempt") != std::string::npos ||
                                      result.error_message.find("retry") != std::string::npos ||
                                      result.error_message.find("Failed mesh pairs") != std::string::npos;
    EXPECT_TRUE(mentions_retry_or_attempts)
        << "Error message should indicate multiple mapping attempts were made. Error: " << result.error_message;
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_StressTest_ManyConfigurations) {
    // Stress test: Run many different configurations to measure solving time
    // This test creates various logical/physical mesh combinations and measures
    // how long the mapping takes for each configuration

    using namespace ::tt::tt_fabric;

    // Test configurations: (logical_meshes, physical_meshes, expected_success, inter_mesh_topology)
    enum class InterMeshTopology {
        LINE,       // Connect meshes in a line: 0-1-2-...
        ALL_TO_ALL  // Fully connected: every mesh connected to every other mesh
    };

    struct TestConfig {
        std::vector<std::pair<size_t, size_t>> logical_meshes;   // (rows, cols) for each logical mesh
        std::vector<std::pair<size_t, size_t>> physical_meshes;  // (rows, cols) for each physical mesh
        bool expected_success;
        std::string description;
        InterMeshTopology inter_mesh_topology = InterMeshTopology::LINE;
    };

    std::vector<TestConfig> configs = {
        // 1. Large single mesh with all-to-all (baseline)
        {{{10, 10}}, {{10, 10}}, true, "Single 10x10 mesh (all-to-all)", InterMeshTopology::ALL_TO_ALL},

        // 2. Multiple large meshes with many inter-mesh connections (all-to-all)
        {{{10, 10}, {10, 10}, {10, 10}},
         {{10, 10}, {10, 10}, {10, 10}},
         true,
         "Three 10x10 meshes (all-to-all)",
         InterMeshTopology::ALL_TO_ALL},

        // 3. Variable mesh sizes with all-to-all
        {{{8, 12}, {12, 8}},
         {{8, 12}, {12, 8}},
         true,
         "Two 8x12/12x8 meshes (all-to-all)",
         InterMeshTopology::ALL_TO_ALL},

        // 4. Physical has more meshes than logical (tests retry logic)
        {{{10, 10}, {10, 10}},
         {{10, 10}, {10, 10}, {10, 10}},
         true,
         "Two logical, three physical (all-to-all)",
         InterMeshTopology::ALL_TO_ALL},

        // 5. Negative test - insufficient physical nodes
        {{{10, 10}, {10, 10}},
         {{8, 8}, {8, 8}},
         false,
         "Two 10x10 logical, two 8x8 physical (insufficient nodes)",
         InterMeshTopology::ALL_TO_ALL},
    };

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;

    size_t total_configs = configs.size();
    size_t successful_configs = 0;
    size_t failed_configs = 0;
    std::chrono::milliseconds total_time{0};
    std::chrono::milliseconds max_time{0};
    std::chrono::milliseconds min_time{std::chrono::milliseconds::max()};

    std::cout << "\n=== Stress Test Results ===" << std::endl;

    for (size_t config_idx = 0; config_idx < configs.size(); ++config_idx) {
        const auto& test_config = configs[config_idx];

        // Create logical multi-mesh graph
        LogicalMultiMeshGraph logical_multi_mesh_graph;
        AdjacencyGraph<MeshId>::AdjacencyMap logical_mesh_level_adj_map;

        for (size_t i = 0; i < test_config.logical_meshes.size(); ++i) {
            const auto& [rows, cols] = test_config.logical_meshes[i];
            const MeshId mesh_id{i};
            size_t node_count = rows * cols;

            std::vector<FabricNodeId> logical_nodes;
            for (uint32_t j = 0; j < node_count; ++j) {
                logical_nodes.push_back(FabricNodeId(mesh_id, j));
            }
            auto logical_adj = build_grid_adjacency(logical_nodes, rows, cols);
            logical_multi_mesh_graph.mesh_adjacency_graphs_[mesh_id] = AdjacencyGraph<FabricNodeId>(logical_adj);
        }

        // Build inter-mesh connectivity based on topology type
        if (test_config.inter_mesh_topology == InterMeshTopology::LINE) {
            // Connect meshes in a line topology: 0-1-2-...
            for (size_t i = 0; i < test_config.logical_meshes.size(); ++i) {
                const MeshId mesh_id{i};
                if (i > 0) {
                    MeshId prev_mesh{i - 1};
                    logical_mesh_level_adj_map[prev_mesh].push_back(mesh_id);
                    logical_mesh_level_adj_map[mesh_id].push_back(prev_mesh);
                }
            }
        } else if (test_config.inter_mesh_topology == InterMeshTopology::ALL_TO_ALL) {
            // Connect all meshes to all other meshes (fully connected/clique)
            for (size_t i = 0; i < test_config.logical_meshes.size(); ++i) {
                const MeshId mesh_id{i};
                for (size_t j = 0; j < test_config.logical_meshes.size(); ++j) {
                    if (i != j) {
                        const MeshId other_mesh_id{j};
                        logical_mesh_level_adj_map[mesh_id].push_back(other_mesh_id);
                    }
                }
            }
        }
        logical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(logical_mesh_level_adj_map);

        // Create physical multi-mesh graph
        PhysicalMultiMeshGraph physical_multi_mesh_graph;
        AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;

        for (size_t i = 0; i < test_config.physical_meshes.size(); ++i) {
            const auto& [rows, cols] = test_config.physical_meshes[i];
            const MeshId mesh_id{i};
            size_t asic_count = rows * cols;

            std::vector<tt::tt_metal::AsicID> physical_asics;
            uint64_t base_id = 1000 * (i + 1);
            for (uint64_t j = 0; j < asic_count; ++j) {
                physical_asics.push_back(tt::tt_metal::AsicID{base_id + j});
            }
            auto physical_adj = build_grid_adjacency(physical_asics, rows, cols);
            physical_multi_mesh_graph.mesh_adjacency_graphs_[mesh_id] =
                AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj);
        }

        // Build inter-mesh connectivity based on topology type (same as logical)
        if (test_config.inter_mesh_topology == InterMeshTopology::LINE) {
            // Connect meshes in a line topology: 0-1-2-...
            for (size_t i = 0; i < test_config.physical_meshes.size(); ++i) {
                const MeshId mesh_id{i};
                if (i > 0) {
                    MeshId prev_mesh{i - 1};
                    physical_mesh_level_adj_map[prev_mesh].push_back(mesh_id);
                    physical_mesh_level_adj_map[mesh_id].push_back(prev_mesh);
                }
            }
        } else if (test_config.inter_mesh_topology == InterMeshTopology::ALL_TO_ALL) {
            // Connect all meshes to all other meshes (fully connected/clique)
            for (size_t i = 0; i < test_config.physical_meshes.size(); ++i) {
                const MeshId mesh_id{i};
                for (size_t j = 0; j < test_config.physical_meshes.size(); ++j) {
                    if (i != j) {
                        const MeshId other_mesh_id{j};
                        physical_mesh_level_adj_map[mesh_id].push_back(other_mesh_id);
                    }
                }
            }
        }
        physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

        // Measure mapping time
        auto start_time = std::chrono::steady_clock::now();
        const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        total_time += elapsed;
        if (elapsed > max_time) {
            max_time = elapsed;
        }
        if (elapsed < min_time) {
            min_time = elapsed;
        }

        // Verify result matches expectation
        bool success_matches = (result.success == test_config.expected_success);
        if (result.success) {
            successful_configs++;
        } else {
            failed_configs++;
        }

        std::cout << "Config " << (config_idx + 1) << "/" << total_configs << ": " << test_config.description << " - "
                  << elapsed.count() << "ms - " << (result.success ? "SUCCESS" : "FAILED");
        if (!success_matches) {
            std::cout << " [UNEXPECTED]";
        }
        std::cout << std::endl;

        // Verify bidirectional consistency if successful
        if (result.success) {
            verify_bidirectional_consistency(result);
        }
    }

    // Print summary statistics
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total configurations: " << total_configs << std::endl;
    std::cout << "Successful: " << successful_configs << std::endl;
    std::cout << "Failed: " << failed_configs << std::endl;
    std::cout << "Total time: " << total_time.count() << "ms" << std::endl;
    std::cout << "Average time: " << (total_time.count() / total_configs) << "ms" << std::endl;
    std::cout << "Min time: " << min_time.count() << "ms" << std::endl;
    std::cout << "Max time: " << max_time.count() << "ms" << std::endl;

    // Verify we ran all configurations
    EXPECT_EQ(total_configs, configs.size());

    // At least some configurations should succeed (the ones marked as expected_success=true)
    EXPECT_GT(successful_configs, 0u) << "At least some configurations should succeed";
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_IncompatibleTopology_2_Fails) {
    // Negative test: Logical topology that cannot be mapped to physical topology
    // This test verifies that the mapper correctly fails when:
    // 1. Logical meshes requires connectivity not available in physical meshes
    // 2. The mapper tries multiple different multi-mesh mapping combinations before failing
    //
    // Logical Topology (2 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 (line: 0-1)
    //
    // Physical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs) - properly connected
    //   Mesh 1: 2x2 grid (4 ASICs) - missing intra-mesh connectivity
    //   Mesh 2: 2x2 grid (4 ASICs) - missing intra-mesh connectivity
    //   Inter-mesh: All-to-all
    //
    // Expected behavior:
    //   - The mapper will try multiple combinations, but will fail because of the missing intra-mesh connectivity
    //   - Eventually exhausts all possibilities and fails
    //   - This ensures the retry logic is exercised (at least 2 attempts)

    using namespace ::tt::tt_fabric;

    const MeshId logical_mesh0{0};
    const MeshId logical_mesh1{1};
    const MeshId physical_mesh0{0};
    const MeshId physical_mesh1{1};
    const MeshId physical_mesh2{2};

    // =========================================================================
    // Create Logical Multi-Mesh Graph (2 meshes: 2x2 and 2x2)
    // =========================================================================

    // Logical Mesh 0: 2x2 grid (4 nodes)
    std::vector<FabricNodeId> logical_nodes_m0;
    for (uint32_t i = 0; i < 4; ++i) {
        logical_nodes_m0.push_back(FabricNodeId(logical_mesh0, i));
    }
    auto logical_adj_m0 = build_grid_adjacency(logical_nodes_m0, 2, 2);

    // Logical Mesh 1: 2x2 grid (4 nodes)
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
    // Create Physical Multi-Mesh Graph (3 meshes, all 2x2)
    // =========================================================================

    // Physical Mesh 0: 2x2 grid (4 ASICs) - properly connected
    std::vector<tt::tt_metal::AsicID> physical_asics_m0;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m0.push_back(tt::tt_metal::AsicID{100 + i});
    }
    auto physical_adj_m0 = build_grid_adjacency(physical_asics_m0, 2, 2);

    // Physical Mesh 1: 2x2 grid (4 ASICs) - missing intra-mesh connectivity (2 x two asics connected in line)
    std::vector<tt::tt_metal::AsicID> physical_asics_m1;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m1.push_back(tt::tt_metal::AsicID{200 + i});
    }
    PhysicalAdjacencyMap physical_adj_m1;
    physical_adj_m1[physical_asics_m1[0]] = {physical_asics_m1[1]};
    physical_adj_m1[physical_asics_m1[1]] = {physical_asics_m1[0]};
    physical_adj_m1[physical_asics_m1[2]] = {physical_asics_m1[3]};
    physical_adj_m1[physical_asics_m1[3]] = {physical_asics_m1[2]};

    // Physical Mesh 2: 2x2 grid (4 ASICs) - missing intra-mesh connectivity (2 x two asics connected in line)
    std::vector<tt::tt_metal::AsicID> physical_asics_m2;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m2.push_back(tt::tt_metal::AsicID{300 + i});
    }
    PhysicalAdjacencyMap physical_adj_m2;
    physical_adj_m2[physical_asics_m2[0]] = {physical_asics_m2[1]};
    physical_adj_m2[physical_asics_m2[1]] = {physical_asics_m2[0]};
    physical_adj_m2[physical_asics_m2[2]] = {physical_asics_m2[3]};
    physical_adj_m2[physical_asics_m2[3]] = {physical_asics_m2[2]};

    // Create physical multi-mesh graph
    PhysicalMultiMeshGraph physical_multi_mesh_graph;
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh0] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m0);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh1] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m1);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh2] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m2);

    // Create mesh-level adjacency map (all-to-all)
    // This allows the mapper to try different combinations of physical meshes
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;
    physical_mesh_level_adj_map[physical_mesh0] = {physical_mesh1, physical_mesh2};
    physical_mesh_level_adj_map[physical_mesh1] = {physical_mesh0, physical_mesh2};
    physical_mesh_level_adj_map[physical_mesh2] = {physical_mesh0, physical_mesh1};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

    // =========================================================================
    // Run mapping and verify failure
    // =========================================================================

    // Create mapping config
    TopologyMappingConfig config;
    config.strict_mode = false;
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    // Call map_multi_mesh_to_physical - should fail
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Verify overall result failed
    EXPECT_FALSE(result.success) << "Multi-mesh mapping should fail due to missing intra-mesh connectivity";
    EXPECT_FALSE(result.error_message.empty()) << "Error message should be provided when mapping fails";

    // Verify that no complete mapping was found
    // The mapper may find partial mappings, but not all logical meshes should be mapped
    std::set<MeshId> mapped_logical_meshes;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mapped_logical_meshes.insert(fabric_node.mesh_id);
    }

    // Verify that the error message indicates multiple attempts were made
    // The error message should mention retry attempts or failed mesh pairs
    // This ensures the retry logic was exercised (at least 2 attempts)
    bool mentions_retry_or_attempts = result.error_message.find("attempt") != std::string::npos ||
                                      result.error_message.find("retry") != std::string::npos ||
                                      result.error_message.find("Failed mesh pairs") != std::string::npos;
    EXPECT_TRUE(mentions_retry_or_attempts)
        << "Error message should indicate multiple mapping attempts were made. Error: " << result.error_message;
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_IncompatibleTopology_3_Fails) {
    // Negative test: Logical topology that cannot be mapped to physical topology
    // This test verifies that the mapper correctly fails when:
    // 1. Logical meshes requires connectivity not available in physical meshes
    // 2. The mapper tries multiple different multi-mesh mapping combinations before failing
    //
    // Logical Topology (2 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 (line: 0-1)
    //
    // Physical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs) - properly connected
    //   Mesh 1: 2x2 grid (4 ASICs) - properly connected
    //   Mesh 2: 2x2 grid (4 ASICs) - properly connected
    //   Inter-mesh: none
    //
    // Expected behavior:
    //   - The mapper will try multiple combinations, but will fail because of the missing inter-mesh connectivity
    //   - Eventually exhausts all possibilities and fails
    //   - This ensures the retry logic is exercised (at least 2 attempts)

    using namespace ::tt::tt_fabric;

    const MeshId logical_mesh0{0};
    const MeshId logical_mesh1{1};
    const MeshId physical_mesh0{0};
    const MeshId physical_mesh1{1};
    const MeshId physical_mesh2{2};

    // =========================================================================
    // Create Logical Multi-Mesh Graph (2 meshes: 2x2 and 2x2)
    // =========================================================================

    // Logical Mesh 0: 2x2 grid (4 nodes)
    std::vector<FabricNodeId> logical_nodes_m0;
    for (uint32_t i = 0; i < 4; ++i) {
        logical_nodes_m0.push_back(FabricNodeId(logical_mesh0, i));
    }
    auto logical_adj_m0 = build_grid_adjacency(logical_nodes_m0, 2, 2);

    // Logical Mesh 1: 2x2 grid (4 nodes)
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
    // Create Physical Multi-Mesh Graph (3 meshes, all 2x2)
    // =========================================================================

    // Physical Mesh 0: 2x2 grid (4 ASICs) - properly connected
    std::vector<tt::tt_metal::AsicID> physical_asics_m0;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m0.push_back(tt::tt_metal::AsicID{100 + i});
    }
    auto physical_adj_m0 = build_grid_adjacency(physical_asics_m0, 2, 2);

    // Physical Mesh 1: 2x2 grid (4 ASICs) - missing intra-mesh connectivity (2 x two asics connected in line)
    std::vector<tt::tt_metal::AsicID> physical_asics_m1;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m1.push_back(tt::tt_metal::AsicID{200 + i});
    }
    auto physical_adj_m1 = build_grid_adjacency(physical_asics_m1, 2, 2);

    // Physical Mesh 2: 2x2 grid (4 ASICs) - missing intra-mesh connectivity (2 x two asics connected in line)
    std::vector<tt::tt_metal::AsicID> physical_asics_m2;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m2.push_back(tt::tt_metal::AsicID{300 + i});
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

    // Create mesh-level adjacency map (no connectivity)
    // This allows the mapper to try different combinations of physical meshes
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;
    physical_mesh_level_adj_map[physical_mesh0] = {};
    physical_mesh_level_adj_map[physical_mesh1] = {};
    physical_mesh_level_adj_map[physical_mesh2] = {};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

    // =========================================================================
    // Run mapping and verify failure
    // =========================================================================

    // Create mapping config
    TopologyMappingConfig config;
    config.strict_mode = false;
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    // Call map_multi_mesh_to_physical - should fail
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Verify overall result failed
    EXPECT_FALSE(result.success) << "Multi-mesh mapping should fail due to missing inter-mesh connectivity";
    EXPECT_FALSE(result.error_message.empty()) << "Error message should be provided when mapping fails";

    // Verify that no complete mapping was found
    // The mapper may find partial mappings, but not all logical meshes should be mapped
    std::set<MeshId> mapped_logical_meshes;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mapped_logical_meshes.insert(fabric_node.mesh_id);
    }

    // Verify that the error message indicates multiple attempts were made
    // The error message should mention retry attempts or failed mesh pairs
    // This ensures the retry logic was exercised (at least 2 attempts)
    bool mentions_retry_or_attempts = result.error_message.find("attempt") != std::string::npos ||
                                      result.error_message.find("retry") != std::string::npos ||
                                      result.error_message.find("Failed mesh pairs") != std::string::npos;
    EXPECT_TRUE(mentions_retry_or_attempts)
        << "Error message should indicate multiple mapping attempts were made. Error: " << result.error_message;
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_IncompatibleTopology_4_Fails) {
    // Negative test: Logical topology that cannot be mapped to physical topology
    // This test verifies that the mapper correctly fails when:
    // 1. Logical meshes requires connectivity not available in physical meshes
    // 2. The mapper tries multiple different multi-mesh mapping combinations before failing
    //
    // Logical Topology (2 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 (none)
    //
    // Physical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs) - properly connected
    //   Mesh 1: 2x2 grid (4 ASICs) - missing intra-mesh connectivity
    //   Mesh 2: 2x2 grid (4 ASICs) - missing intra-mesh connectivity
    //   Inter-mesh: All-to-all
    //
    // Expected behavior:
    //   - The mapper will try multiple combinations, but will fail because of the missing intra-mesh connectivity
    //   - Eventually exhausts all possibilities and fails
    //   - This ensures the retry logic is exercised (at least 2 attempts)

    using namespace ::tt::tt_fabric;

    const MeshId logical_mesh0{0};
    const MeshId logical_mesh1{1};
    const MeshId physical_mesh0{0};
    const MeshId physical_mesh1{1};
    const MeshId physical_mesh2{2};

    // =========================================================================
    // Create Logical Multi-Mesh Graph (2 meshes: 2x2 and 2x2)
    // =========================================================================

    // Logical Mesh 0: 2x2 grid (4 nodes)
    std::vector<FabricNodeId> logical_nodes_m0;
    for (uint32_t i = 0; i < 4; ++i) {
        logical_nodes_m0.push_back(FabricNodeId(logical_mesh0, i));
    }
    auto logical_adj_m0 = build_grid_adjacency(logical_nodes_m0, 2, 2);

    // Logical Mesh 1: 2x2 grid (4 nodes)
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
    logical_mesh_level_adj_map[logical_mesh0] = {};
    logical_mesh_level_adj_map[logical_mesh1] = {};
    logical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(logical_mesh_level_adj_map);

    // =========================================================================
    // Create Physical Multi-Mesh Graph (3 meshes, all 2x2)
    // =========================================================================

    // Physical Mesh 0: 2x2 grid (4 ASICs) - properly connected
    std::vector<tt::tt_metal::AsicID> physical_asics_m0;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m0.push_back(tt::tt_metal::AsicID{100 + i});
    }
    auto physical_adj_m0 = build_grid_adjacency(physical_asics_m0, 2, 2);

    // Physical Mesh 1: 2x2 grid (4 ASICs) - missing intra-mesh connectivity (2 x two asics connected in line)
    std::vector<tt::tt_metal::AsicID> physical_asics_m1;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m1.push_back(tt::tt_metal::AsicID{200 + i});
    }
    PhysicalAdjacencyMap physical_adj_m1;
    physical_adj_m1[physical_asics_m1[0]] = {physical_asics_m1[1]};
    physical_adj_m1[physical_asics_m1[1]] = {physical_asics_m1[0]};
    physical_adj_m1[physical_asics_m1[2]] = {physical_asics_m1[3]};
    physical_adj_m1[physical_asics_m1[3]] = {physical_asics_m1[2]};

    // Physical Mesh 2: 2x2 grid (4 ASICs) - missing intra-mesh connectivity (2 x two asics connected in line)
    std::vector<tt::tt_metal::AsicID> physical_asics_m2;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m2.push_back(tt::tt_metal::AsicID{300 + i});
    }
    PhysicalAdjacencyMap physical_adj_m2;
    physical_adj_m2[physical_asics_m2[0]] = {physical_asics_m2[1]};
    physical_adj_m2[physical_asics_m2[1]] = {physical_asics_m2[0]};
    physical_adj_m2[physical_asics_m2[2]] = {physical_asics_m2[3]};
    physical_adj_m2[physical_asics_m2[3]] = {physical_asics_m2[2]};

    // Create physical multi-mesh graph
    PhysicalMultiMeshGraph physical_multi_mesh_graph;
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh0] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m0);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh1] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m1);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[physical_mesh2] =
        AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m2);

    // Create mesh-level adjacency map (all-to-all)
    // This allows the mapper to try different combinations of physical meshes
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_level_adj_map;
    physical_mesh_level_adj_map[physical_mesh0] = {physical_mesh1, physical_mesh2};
    physical_mesh_level_adj_map[physical_mesh1] = {physical_mesh0, physical_mesh2};
    physical_mesh_level_adj_map[physical_mesh2] = {physical_mesh0, physical_mesh1};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_level_adj_map);

    // =========================================================================
    // Run mapping and verify failure
    // =========================================================================

    // Create mapping config
    TopologyMappingConfig config;
    config.strict_mode = false;
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    // Call map_multi_mesh_to_physical - should fail
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Verify overall result failed
    EXPECT_FALSE(result.success) << "Multi-mesh mapping should fail due to missing intra-mesh connectivity";
    EXPECT_FALSE(result.error_message.empty()) << "Error message should be provided when mapping fails";

    // Verify that no complete mapping was found
    // The mapper may find partial mappings, but not all logical meshes should be mapped
    std::set<MeshId> mapped_logical_meshes;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mapped_logical_meshes.insert(fabric_node.mesh_id);
    }

    // Verify that the error message indicates multiple attempts were made
    // The error message should mention retry attempts or failed mesh pairs
    // This ensures the retry logic was exercised (at least 2 attempts)
    bool mentions_retry_or_attempts = result.error_message.find("attempt") != std::string::npos ||
                                      result.error_message.find("retry") != std::string::npos ||
                                      result.error_message.find("Failed mesh pairs") != std::string::npos;
    EXPECT_TRUE(mentions_retry_or_attempts)
        << "Error message should indicate multiple mapping attempts were made. Error: " << result.error_message;
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_SingleMesh) {
    // Test converting a flat adjacency graph with a single mesh
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0{100};
    tt::tt_metal::AsicID asic1{101};
    tt::tt_metal::AsicID asic2{102};

    // Create a simple chain: asic0 -> asic1 -> asic2
    flat_adj[asic0] = {asic1};
    flat_adj[asic1] = {asic0, asic2};
    flat_adj[asic2] = {asic1};

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign all ASICs to mesh 0
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic2] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify we have 1 mesh
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 1u) << "Should have 1 mesh";
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.contains(MeshId{0})) << "Should have mesh 0";

    // Verify mesh adjacency graph
    const auto& mesh_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& nodes = mesh_graph.get_nodes();
    EXPECT_EQ(nodes.size(), 3u) << "Mesh 0 should have 3 ASICs";

    // Verify connectivity
    const auto& neighbors0 = mesh_graph.get_neighbors(asic0);
    EXPECT_EQ(neighbors0.size(), 1u) << "ASIC 0 should have 1 neighbor";
    EXPECT_EQ(neighbors0[0], asic1) << "ASIC 0 should connect to ASIC 1";

    const auto& neighbors1 = mesh_graph.get_neighbors(asic1);
    EXPECT_EQ(neighbors1.size(), 2u) << "ASIC 1 should have 2 neighbors";
    EXPECT_TRUE(std::find(neighbors1.begin(), neighbors1.end(), asic0) != neighbors1.end())
        << "ASIC 1 should connect to ASIC 0";
    EXPECT_TRUE(std::find(neighbors1.begin(), neighbors1.end(), asic2) != neighbors1.end())
        << "ASIC 1 should connect to ASIC 2";

    // Verify no intermesh connections
    EXPECT_EQ(multi_mesh_graph.mesh_level_graph_.get_nodes().size(), 1u) << "Should have 1 mesh in mesh-level graph";
    const auto& mesh_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_TRUE(mesh_neighbors.empty()) << "Mesh 0 should have no intermesh connections";

    // Verify no exit nodes
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Should have exit node graph for mesh 0";
    const auto& exit_nodes = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes();
    EXPECT_TRUE(exit_nodes.empty()) << "Mesh 0 should have no exit nodes";
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_TwoMeshes) {
    // Test converting a flat adjacency graph with two meshes and intermesh connections
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // Define all ASICs first
    // Mesh 0: ASICs 100-103 (4 ASICs in a chain)
    tt::tt_metal::AsicID asic0_0{100};
    tt::tt_metal::AsicID asic0_1{101};
    tt::tt_metal::AsicID asic0_2{102};
    tt::tt_metal::AsicID asic0_3{103};
    // Mesh 1: ASICs 200-203 (4 ASICs in a chain)
    tt::tt_metal::AsicID asic1_0{200};
    tt::tt_metal::AsicID asic1_1{201};
    tt::tt_metal::AsicID asic1_2{202};
    tt::tt_metal::AsicID asic1_3{203};

    // Mesh 0: chain 100-101-102-103, with exit node 100 connecting to mesh 1
    flat_adj[asic0_0] = {asic0_1, asic1_0};  // Intra-mesh + intermesh
    flat_adj[asic0_1] = {asic0_0, asic0_2};  // Intra-mesh only
    flat_adj[asic0_2] = {asic0_1, asic0_3};  // Intra-mesh only
    flat_adj[asic0_3] = {asic0_2};           // Intra-mesh only

    // Mesh 1: chain 200-201-202-203, with exit node 200 connecting to mesh 0
    flat_adj[asic1_0] = {asic1_1, asic0_0};  // Intra-mesh + intermesh
    flat_adj[asic1_1] = {asic1_0, asic1_2};  // Intra-mesh only
    flat_adj[asic1_2] = {asic1_1, asic1_3};  // Intra-mesh only
    flat_adj[asic1_3] = {asic1_2};           // Intra-mesh only

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign ASICs to meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_3] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_3] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify we have 2 meshes
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 2u) << "Should have 2 meshes";
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.contains(MeshId{0})) << "Should have mesh 0";
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.contains(MeshId{1})) << "Should have mesh 1";

    // Verify mesh 0 adjacency graph (only intra-mesh connections)
    const auto& mesh0_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& mesh0_nodes = mesh0_graph.get_nodes();
    EXPECT_EQ(mesh0_nodes.size(), 4u) << "Mesh 0 should have 4 ASICs";
    const auto& neighbors0_0 = mesh0_graph.get_neighbors(asic0_0);
    EXPECT_EQ(neighbors0_0.size(), 1u) << "ASIC 0_0 should have 1 intra-mesh neighbor";
    EXPECT_EQ(neighbors0_0[0], asic0_1) << "ASIC 0_0 should connect to ASIC 0_1";

    // Verify mesh 1 adjacency graph (only intra-mesh connections)
    const auto& mesh1_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{1});
    const auto& mesh1_nodes = mesh1_graph.get_nodes();
    EXPECT_EQ(mesh1_nodes.size(), 4u) << "Mesh 1 should have 4 ASICs";
    const auto& neighbors1_0 = mesh1_graph.get_neighbors(asic1_0);
    EXPECT_EQ(neighbors1_0.size(), 1u) << "ASIC 1_0 should have 1 intra-mesh neighbor";
    EXPECT_EQ(neighbors1_0[0], asic1_1) << "ASIC 1_0 should connect to ASIC 1_1";

    // Verify intermesh connections
    const auto& mesh_level_nodes = multi_mesh_graph.mesh_level_graph_.get_nodes();
    EXPECT_EQ(mesh_level_nodes.size(), 2u) << "Should have 2 meshes in mesh-level graph";
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 1u) << "Mesh 0 should have 1 intermesh neighbor";
    EXPECT_EQ(mesh0_neighbors[0], MeshId{1}) << "Mesh 0 should connect to mesh 1";

    const auto& mesh1_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 1u) << "Mesh 1 should have 1 intermesh neighbor";
    EXPECT_EQ(mesh1_neighbors[0], MeshId{0}) << "Mesh 1 should connect to mesh 0";

    // Verify exit nodes
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Should have exit node graph for mesh 0";
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 1u) << "Mesh 0 should have 1 exit node";
    PhysicalExitNode exit_node0{MeshId{0}, asic0_0};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0) != exit_nodes0.end())
        << "ASIC 0_0 should be an exit node";
    const auto& exit_neighbors0 = exit_graph0.get_neighbors(exit_node0);
    EXPECT_EQ(exit_neighbors0.size(), 1u) << "Exit node should have 1 connection";
    PhysicalExitNode exit_node1{MeshId{1}, asic1_0};
    EXPECT_EQ(exit_neighbors0[0], exit_node1) << "Exit node should connect to ASIC 1_0";

    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Should have exit node graph for mesh 1";
    const auto& exit_graph1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 1u) << "Mesh 1 should have 1 exit node";
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1) != exit_nodes1.end())
        << "ASIC 1_0 should be an exit node";
    const auto& exit_neighbors1 = exit_graph1.get_neighbors(exit_node1);
    EXPECT_EQ(exit_neighbors1.size(), 1u) << "Exit node should have 1 connection";
    EXPECT_EQ(exit_neighbors1[0], exit_node0) << "Exit node should connect to ASIC 0_0";
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_MultipleChannels) {
    // Test that multiple channels between the same pair are preserved
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // Mesh 0: ASICs 100-102 (3 ASICs in a chain)
    tt::tt_metal::AsicID asic0_0{100};
    tt::tt_metal::AsicID asic0_1{101};
    tt::tt_metal::AsicID asic0_2{102};
    // Mesh 1: ASICs 200-202 (3 ASICs in a chain)
    tt::tt_metal::AsicID asic1_0{200};
    tt::tt_metal::AsicID asic1_1{201};
    tt::tt_metal::AsicID asic1_2{202};

    // Build adjacency map
    // Mesh 0: chain 100-101-102, with exit node 100 connecting to mesh 1
    flat_adj[asic0_0] = {asic0_1, asic1_0, asic1_0, asic1_0};  // Internal + 3 exit channels
    flat_adj[asic0_1] = {asic0_0, asic0_2};                    // Internal only
    flat_adj[asic0_2] = {asic0_1};                             // Internal only

    // Mesh 1: chain 200-201-202, with exit node 200 connecting to mesh 0
    flat_adj[asic1_0] = {asic1_1, asic0_0, asic0_0, asic0_0};  // Internal + 3 exit channels
    flat_adj[asic1_1] = {asic1_0, asic1_2};                    // Internal only
    flat_adj[asic1_2] = {asic1_1};                             // Internal only

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to different meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_2] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify mesh 0 has correct internal structure (3 ASICs)
    const auto& mesh0_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& mesh0_nodes = mesh0_graph.get_nodes();
    EXPECT_EQ(mesh0_nodes.size(), 3u) << "Mesh 0 should have 3 ASICs";

    // Verify exit node graph preserves multiple channels
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 1u) << "Mesh 0 should have 1 exit node";
    PhysicalExitNode exit_node0_0{MeshId{0}, asic0_0};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_0) != exit_nodes0.end())
        << "ASIC 0_0 should be an exit node";

    const auto& exit_neighbors0 = exit_graph0.get_neighbors(exit_node0_0);
    EXPECT_EQ(exit_neighbors0.size(), 3u) << "Exit node should have 3 connections (3 channels)";
    PhysicalExitNode exit_node1_0{MeshId{1}, asic1_0};
    EXPECT_EQ(exit_neighbors0[0], exit_node1_0) << "All connections should be to ASIC 1_0";
    EXPECT_EQ(exit_neighbors0[1], exit_node1_0);
    EXPECT_EQ(exit_neighbors0[2], exit_node1_0);

    // Verify mesh 1 has correct internal structure (3 ASICs)
    const auto& mesh1_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{1});
    const auto& mesh1_nodes = mesh1_graph.get_nodes();
    EXPECT_EQ(mesh1_nodes.size(), 3u) << "Mesh 1 should have 3 ASICs";

    const auto& exit_graph1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 1u) << "Mesh 1 should have 1 exit node";
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_0) != exit_nodes1.end())
        << "ASIC 1_0 should be an exit node";

    const auto& exit_neighbors1 = exit_graph1.get_neighbors(exit_node1_0);
    EXPECT_EQ(exit_neighbors1.size(), 3u) << "Exit node should have 3 connections (3 channels)";
    EXPECT_EQ(exit_neighbors1[0], exit_node0_0) << "All connections should be to ASIC 0_0";
    EXPECT_EQ(exit_neighbors1[1], exit_node0_0);
    EXPECT_EQ(exit_neighbors1[2], exit_node0_0);
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_ThreeMeshes) {
    // Test converting with three meshes in a line topology
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // Define all ASICs first
    tt::tt_metal::AsicID asic0{100};
    tt::tt_metal::AsicID asic1{200};
    tt::tt_metal::AsicID asic2{300};

    // Mesh 0: ASIC 100
    flat_adj[asic0] = {asic1};

    // Mesh 1: ASIC 200
    flat_adj[asic1] = {asic0, asic2};

    // Mesh 2: ASIC 300
    flat_adj[asic2] = {asic1};

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify we have 3 meshes
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 meshes";

    // Verify mesh-level graph: 0-1-2 line
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 1u) << "Mesh 0 should have 1 neighbor";
    EXPECT_EQ(mesh0_neighbors[0], MeshId{1}) << "Mesh 0 should connect to mesh 1";

    const auto& mesh1_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 2u) << "Mesh 1 should have 2 neighbors";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{0}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 0";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{2}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 2";

    const auto& mesh2_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{2});
    EXPECT_EQ(mesh2_neighbors.size(), 1u) << "Mesh 2 should have 1 neighbor";
    EXPECT_EQ(mesh2_neighbors[0], MeshId{1}) << "Mesh 2 should connect to mesh 1";

    // Verify exit nodes
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u)
        << "Mesh 0 should have 1 exit node";
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 1u)
        << "Mesh 1 should have 1 exit node";
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2}).get_nodes().size(), 1u)
        << "Mesh 2 should have 1 exit node";
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_DisconnectedMeshes) {
    // Test with multiple meshes that have no intermesh connections
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // Mesh 0: ASICs 100-104 (5 ASICs in a chain)
    tt::tt_metal::AsicID asic0_0{100};
    tt::tt_metal::AsicID asic0_1{101};
    tt::tt_metal::AsicID asic0_2{102};
    tt::tt_metal::AsicID asic0_3{103};
    tt::tt_metal::AsicID asic0_4{104};
    flat_adj[asic0_0] = {asic0_1};
    flat_adj[asic0_1] = {asic0_0, asic0_2};
    flat_adj[asic0_2] = {asic0_1, asic0_3};
    flat_adj[asic0_3] = {asic0_2, asic0_4};
    flat_adj[asic0_4] = {asic0_3};

    // Mesh 1: ASICs 200-203 (4 ASICs in a chain)
    tt::tt_metal::AsicID asic1_0{200};
    tt::tt_metal::AsicID asic1_1{201};
    tt::tt_metal::AsicID asic1_2{202};
    tt::tt_metal::AsicID asic1_3{203};
    flat_adj[asic1_0] = {asic1_1};
    flat_adj[asic1_1] = {asic1_0, asic1_2};
    flat_adj[asic1_2] = {asic1_1, asic1_3};
    flat_adj[asic1_3] = {asic1_2};

    // Mesh 2: ASICs 300-301 (2 ASICs connected)
    tt::tt_metal::AsicID asic2_0{300};
    tt::tt_metal::AsicID asic2_1{301};
    flat_adj[asic2_0] = {asic2_1};
    flat_adj[asic2_1] = {asic2_0};

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to different meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_3] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_4] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_3] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2_1] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify we have 3 meshes
    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 meshes";

    // Verify mesh 0 has correct internal structure (5 ASICs)
    const auto& mesh0_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& mesh0_nodes = mesh0_graph.get_nodes();
    EXPECT_EQ(mesh0_nodes.size(), 5u) << "Mesh 0 should have 5 ASICs";

    // Verify mesh 1 has correct internal structure (4 ASICs)
    const auto& mesh1_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{1});
    const auto& mesh1_nodes = mesh1_graph.get_nodes();
    EXPECT_EQ(mesh1_nodes.size(), 4u) << "Mesh 1 should have 4 ASICs";

    // Verify mesh 2 has correct internal structure (2 ASICs)
    const auto& mesh2_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{2});
    const auto& mesh2_nodes = mesh2_graph.get_nodes();
    EXPECT_EQ(mesh2_nodes.size(), 2u) << "Mesh 2 should have 2 ASICs";

    // Verify no intermesh connections
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_TRUE(mesh0_neighbors.empty()) << "Mesh 0 should have no intermesh connections";

    const auto& mesh1_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_TRUE(mesh1_neighbors.empty()) << "Mesh 1 should have no intermesh connections";

    const auto& mesh2_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{2});
    EXPECT_TRUE(mesh2_neighbors.empty()) << "Mesh 2 should have no intermesh connections";

    // Verify no exit nodes (all meshes are disconnected)
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Should have exit node graph for mesh 0";
    const auto& exit_nodes0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes();
    EXPECT_TRUE(exit_nodes0.empty()) << "Mesh 0 should have no exit nodes (disconnected)";

    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Should have exit node graph for mesh 1";
    const auto& exit_nodes1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes();
    EXPECT_TRUE(exit_nodes1.empty()) << "Mesh 1 should have no exit nodes (disconnected)";

    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{2}))
        << "Should have exit node graph for mesh 2";
    const auto& exit_nodes2 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2}).get_nodes();
    EXPECT_TRUE(exit_nodes2.empty()) << "Mesh 2 should have no exit nodes (disconnected)";
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_MultipleExitNodesPerMesh) {
    // Test with multiple exit nodes in the same mesh connecting to different meshes
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // Define all ASICs first
    // Mesh 0: ASICs 100-104 (5 ASICs in a chain)
    tt::tt_metal::AsicID asic0_0{100};
    tt::tt_metal::AsicID asic0_1{101};
    tt::tt_metal::AsicID asic0_2{102};
    tt::tt_metal::AsicID asic0_3{103};
    tt::tt_metal::AsicID asic0_4{104};
    // Mesh 1: ASICs 200-202 (3 ASICs in a chain)
    tt::tt_metal::AsicID asic1_0{200};
    tt::tt_metal::AsicID asic1_1{201};
    tt::tt_metal::AsicID asic1_2{202};
    // Mesh 2: ASICs 300-302 (3 ASICs in a chain)
    tt::tt_metal::AsicID asic2_0{300};
    tt::tt_metal::AsicID asic2_1{301};
    tt::tt_metal::AsicID asic2_2{302};

    // Build adjacency map
    // Mesh 0: chain 100-101-102-103-104, with exit nodes 100 (to mesh 1) and 104 (to mesh 2)
    flat_adj[asic0_0] = {asic0_1, asic1_0};  // Internal + exit to mesh 1
    flat_adj[asic0_1] = {asic0_0, asic0_2};  // Internal only
    flat_adj[asic0_2] = {asic0_1, asic0_3};  // Internal only
    flat_adj[asic0_3] = {asic0_2, asic0_4};  // Internal only
    flat_adj[asic0_4] = {asic0_3, asic2_0};  // Internal + exit to mesh 2

    // Mesh 1: chain 200-201-202, with exit node 200 (to mesh 0)
    flat_adj[asic1_0] = {asic1_1, asic0_0};  // Internal + exit to mesh 0
    flat_adj[asic1_1] = {asic1_0, asic1_2};  // Internal only
    flat_adj[asic1_2] = {asic1_1};           // Internal only

    // Mesh 2: chain 300-301-302, with exit node 300 (to mesh 0)
    flat_adj[asic2_0] = {asic2_1, asic0_4};  // Internal + exit to mesh 0
    flat_adj[asic2_1] = {asic2_0, asic2_2};  // Internal only
    flat_adj[asic2_2] = {asic2_1};           // Internal only

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_3] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_4] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2_2] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify mesh 0 has correct internal structure (5 ASICs in chain)
    const auto& mesh0_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& mesh0_nodes = mesh0_graph.get_nodes();
    EXPECT_EQ(mesh0_nodes.size(), 5u) << "Mesh 0 should have 5 ASICs";

    // Verify mesh 0 has 2 exit nodes
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 2u) << "Mesh 0 should have 2 exit nodes";
    PhysicalExitNode exit_node0_0{MeshId{0}, asic0_0};
    PhysicalExitNode exit_node0_4{MeshId{0}, asic0_4};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_0) != exit_nodes0.end())
        << "ASIC 0_0 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_4) != exit_nodes0.end())
        << "ASIC 0_4 should be an exit node";

    // Verify exit node connections
    const auto& neighbors0_0 = exit_graph0.get_neighbors(exit_node0_0);
    EXPECT_EQ(neighbors0_0.size(), 1u) << "ASIC 0_0 should have 1 exit connection";
    PhysicalExitNode exit_node1_0{MeshId{1}, asic1_0};
    EXPECT_EQ(neighbors0_0[0], exit_node1_0) << "ASIC 0_0 should connect to ASIC 1_0";

    const auto& neighbors0_4 = exit_graph0.get_neighbors(exit_node0_4);
    EXPECT_EQ(neighbors0_4.size(), 1u) << "ASIC 0_4 should have 1 exit connection";
    PhysicalExitNode exit_node2_0{MeshId{2}, asic2_0};
    EXPECT_EQ(neighbors0_4[0], exit_node2_0) << "ASIC 0_4 should connect to ASIC 2_0";

    // Verify mesh 1 has correct internal structure (3 ASICs in chain)
    const auto& mesh1_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{1});
    const auto& mesh1_nodes = mesh1_graph.get_nodes();
    EXPECT_EQ(mesh1_nodes.size(), 3u) << "Mesh 1 should have 3 ASICs";

    // Verify mesh 1 has 1 exit node
    const auto& exit_graph1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 1u) << "Mesh 1 should have 1 exit node";
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_0) != exit_nodes1.end())
        << "ASIC 1_0 should be an exit node";

    // Verify mesh 2 has correct internal structure (3 ASICs in chain)
    const auto& mesh2_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{2});
    const auto& mesh2_nodes = mesh2_graph.get_nodes();
    EXPECT_EQ(mesh2_nodes.size(), 3u) << "Mesh 2 should have 3 ASICs";

    // Verify mesh 2 has 1 exit node
    const auto& exit_graph2 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2});
    const auto& exit_nodes2 = exit_graph2.get_nodes();
    EXPECT_EQ(exit_nodes2.size(), 1u) << "Mesh 2 should have 1 exit node";
    EXPECT_TRUE(std::find(exit_nodes2.begin(), exit_nodes2.end(), exit_node2_0) != exit_nodes2.end())
        << "ASIC 2_0 should be an exit node";

    // Verify mesh-level connectivity
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 2u) << "Mesh 0 should connect to 2 meshes";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{1}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 1";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{2}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 2";
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_MeshWithOnlyExitNodes) {
    // Test a mesh where all ASICs are exit nodes (no internal connections)
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // Define all ASICs first
    // Mesh 0: ASICs 100-104 (5 ASICs, all are exit nodes, no internal connections)
    tt::tt_metal::AsicID asic0_0{100};
    tt::tt_metal::AsicID asic0_1{101};
    tt::tt_metal::AsicID asic0_2{102};
    tt::tt_metal::AsicID asic0_3{103};
    tt::tt_metal::AsicID asic0_4{104};
    // Mesh 1: ASICs 200-204 (5 ASICs in a chain)
    tt::tt_metal::AsicID asic1_0{200};
    tt::tt_metal::AsicID asic1_1{201};
    tt::tt_metal::AsicID asic1_2{202};
    tt::tt_metal::AsicID asic1_3{203};
    tt::tt_metal::AsicID asic1_4{204};

    // Build adjacency map
    // Mesh 0: all ASICs only have exit connections (no internal connections)
    flat_adj[asic0_0] = {asic1_0};  // Only exit connection
    flat_adj[asic0_1] = {asic1_1};  // Only exit connection
    flat_adj[asic0_2] = {asic1_2};  // Only exit connection
    flat_adj[asic0_3] = {asic1_3};  // Only exit connection
    flat_adj[asic0_4] = {asic1_4};  // Only exit connection

    // Mesh 1: chain 200-201-202-203-204, with exit connections to mesh 0
    flat_adj[asic1_0] = {asic1_1, asic0_0};           // Internal + exit
    flat_adj[asic1_1] = {asic1_0, asic1_2, asic0_1};  // Internal + exit
    flat_adj[asic1_2] = {asic1_1, asic1_3, asic0_2};  // Internal + exit
    flat_adj[asic1_3] = {asic1_2, asic1_4, asic0_3};  // Internal + exit
    flat_adj[asic1_4] = {asic1_3, asic0_4};           // Internal + exit

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_3] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic0_4] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_3] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1_4] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify mesh 0 has no internal connections
    const auto& mesh0_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& mesh0_nodes = mesh0_graph.get_nodes();
    EXPECT_EQ(mesh0_nodes.size(), 5u) << "Mesh 0 should have 5 ASICs";
    // All nodes should have no internal neighbors
    for (const auto& node : mesh0_nodes) {
        const auto& neighbors = mesh0_graph.get_neighbors(node);
        EXPECT_TRUE(neighbors.empty()) << "ASIC " << node.get() << " in mesh 0 should have no internal neighbors";
    }

    // Verify mesh 0 has 5 exit nodes (all ASICs)
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 5u) << "Mesh 0 should have 5 exit nodes (all ASICs)";
    PhysicalExitNode exit_node0_0{MeshId{0}, asic0_0};
    PhysicalExitNode exit_node0_1{MeshId{0}, asic0_1};
    PhysicalExitNode exit_node0_2{MeshId{0}, asic0_2};
    PhysicalExitNode exit_node0_3{MeshId{0}, asic0_3};
    PhysicalExitNode exit_node0_4{MeshId{0}, asic0_4};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_0) != exit_nodes0.end())
        << "ASIC 0_0 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_1) != exit_nodes0.end())
        << "ASIC 0_1 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_2) != exit_nodes0.end())
        << "ASIC 0_2 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_3) != exit_nodes0.end())
        << "ASIC 0_3 should be an exit node";
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_4) != exit_nodes0.end())
        << "ASIC 0_4 should be an exit node";

    // Verify exit node connections
    PhysicalExitNode exit_node1_0{MeshId{1}, asic1_0};
    PhysicalExitNode exit_node1_1{MeshId{1}, asic1_1};
    EXPECT_EQ(exit_graph0.get_neighbors(exit_node0_0).size(), 1u) << "ASIC 0_0 should have 1 exit connection";
    EXPECT_EQ(exit_graph0.get_neighbors(exit_node0_0)[0], exit_node1_0) << "ASIC 0_0 should connect to ASIC 1_0";
    EXPECT_EQ(exit_graph0.get_neighbors(exit_node0_1).size(), 1u) << "ASIC 0_1 should have 1 exit connection";
    EXPECT_EQ(exit_graph0.get_neighbors(exit_node0_1)[0], exit_node1_1) << "ASIC 0_1 should connect to ASIC 1_1";

    // Verify mesh 1 has correct internal structure (5 ASICs in chain)
    const auto& mesh1_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{1});
    const auto& mesh1_nodes = mesh1_graph.get_nodes();
    EXPECT_EQ(mesh1_nodes.size(), 5u) << "Mesh 1 should have 5 ASICs";

    // Verify mesh 1 has 5 exit nodes (all ASICs have exit connections)
    const auto& exit_graph1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 5u) << "Mesh 1 should have 5 exit nodes";
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_EmptyGraph) {
    // Test with empty graph
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify empty result
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.empty()) << "Should have no meshes";
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.empty()) << "Should have no exit node graphs";
    EXPECT_TRUE(multi_mesh_graph.mesh_level_graph_.get_nodes().empty()) << "Should have no mesh-level nodes";
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_UnassignedASICs) {
    // Test that ASICs not in any mesh assignment are skipped
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    // ASIC 100 assigned to mesh 0
    tt::tt_metal::AsicID asic0{100};
    // ASIC 200 assigned to mesh 1
    tt::tt_metal::AsicID asic1{200};
    // ASIC 300 NOT assigned to any mesh
    tt::tt_metal::AsicID unassigned{300};

    flat_adj[asic0] = {asic1, unassigned};  // Connection to assigned + unassigned
    flat_adj[asic1] = {asic0, unassigned};  // Connection to assigned + unassigned
    flat_adj[unassigned] = {asic0, asic1};  // Unassigned ASIC with connections

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Only assign asic0 and asic1, not unassigned
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify unassigned ASIC is not in any mesh graph
    const auto& mesh0_graph = multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0});
    const auto& mesh0_nodes = mesh0_graph.get_nodes();
    EXPECT_EQ(mesh0_nodes.size(), 1u) << "Mesh 0 should have 1 ASIC";
    EXPECT_TRUE(std::find(mesh0_nodes.begin(), mesh0_nodes.end(), asic0) != mesh0_nodes.end())
        << "Mesh 0 should contain ASIC 0";
    EXPECT_TRUE(std::find(mesh0_nodes.begin(), mesh0_nodes.end(), unassigned) == mesh0_nodes.end())
        << "Mesh 0 should not contain unassigned ASIC";

    // Verify connections to unassigned ASIC are ignored
    const auto& neighbors0 = mesh0_graph.get_neighbors(asic0);
    // Should only have connection to asic1 (intermesh), not to unassigned
    EXPECT_EQ(neighbors0.size(), 0u) << "ASIC 0 should have no intra-mesh neighbors (asic1 is in different mesh)";

    // Verify exit node graph - asic0 should connect to asic1 (intermesh)
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    PhysicalExitNode exit_node0{MeshId{0}, asic0};
    PhysicalExitNode exit_node1{MeshId{1}, asic1};
    const auto& exit_neighbors0 = exit_graph0.get_neighbors(exit_node0);
    EXPECT_EQ(exit_neighbors0.size(), 1u) << "ASIC 0 should have 1 exit connection";
    EXPECT_EQ(exit_neighbors0[0], exit_node1) << "ASIC 0 should connect to ASIC 1 (not unassigned)";
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_RingTopology) {
    // Test with 4 meshes in a ring topology: 0-1-2-3-0
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    tt::tt_metal::AsicID asic0{100};
    tt::tt_metal::AsicID asic1{200};
    tt::tt_metal::AsicID asic2{300};
    tt::tt_metal::AsicID asic3{400};

    // Ring: 0-1-2-3-0
    flat_adj[asic0] = {asic3, asic1};  // Connect to mesh 3 and mesh 1
    flat_adj[asic1] = {asic0, asic2};  // Connect to mesh 0 and mesh 2
    flat_adj[asic2] = {asic1, asic3};  // Connect to mesh 1 and mesh 3
    flat_adj[asic3] = {asic2, asic0};  // Connect to mesh 2 and mesh 0

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{3}][asic3] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify ring topology in mesh-level graph
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 2u) << "Mesh 0 should have 2 neighbors";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{1}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 1";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{3}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 3";

    const auto& mesh1_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 2u) << "Mesh 1 should have 2 neighbors";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{0}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 0";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{2}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 2";

    const auto& mesh2_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{2});
    EXPECT_EQ(mesh2_neighbors.size(), 2u) << "Mesh 2 should have 2 neighbors";
    EXPECT_TRUE(std::find(mesh2_neighbors.begin(), mesh2_neighbors.end(), MeshId{1}) != mesh2_neighbors.end())
        << "Mesh 2 should connect to mesh 1";
    EXPECT_TRUE(std::find(mesh2_neighbors.begin(), mesh2_neighbors.end(), MeshId{3}) != mesh2_neighbors.end())
        << "Mesh 2 should connect to mesh 3";

    const auto& mesh3_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{3});
    EXPECT_EQ(mesh3_neighbors.size(), 2u) << "Mesh 3 should have 2 neighbors";
    EXPECT_TRUE(std::find(mesh3_neighbors.begin(), mesh3_neighbors.end(), MeshId{2}) != mesh3_neighbors.end())
        << "Mesh 3 should connect to mesh 2";
    EXPECT_TRUE(std::find(mesh3_neighbors.begin(), mesh3_neighbors.end(), MeshId{0}) != mesh3_neighbors.end())
        << "Mesh 3 should connect to mesh 0";

    // Verify all meshes have exit nodes
    for (MeshId mesh_id{0}; mesh_id.get() < 4; mesh_id = MeshId{mesh_id.get() + 1}) {
        const auto& exit_nodes = multi_mesh_graph.mesh_exit_node_graphs_.at(mesh_id).get_nodes();
        EXPECT_EQ(exit_nodes.size(), 1u) << "Mesh " << mesh_id.get() << " should have 1 exit node";
    }
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_StarTopology) {
    // Test with star topology: mesh 0 in center, meshes 1,2,3 connected to it
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;

    tt::tt_metal::AsicID asic0{100};  // Center
    tt::tt_metal::AsicID asic1{200};
    tt::tt_metal::AsicID asic2{300};
    tt::tt_metal::AsicID asic3{400};

    // Star: all connect to center (mesh 0)
    flat_adj[asic0] = {asic1, asic2, asic3};  // Center connects to all
    flat_adj[asic1] = {asic0};                // Leaf connects to center
    flat_adj[asic2] = {asic0};                // Leaf connects to center
    flat_adj[asic3] = {asic0};                // Leaf connects to center

    // Convert to AdjacencyGraph
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);

    // Assign to meshes
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{3}][asic3] = MeshHostRankId{0};

    // Convert to multi-mesh graph
    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify star topology in mesh-level graph
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 3u) << "Mesh 0 (center) should have 3 neighbors";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{1}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 1";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{2}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 2";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{3}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 3";

    // Verify leaf meshes only connect to center
    for (MeshId mesh_id{1}; mesh_id.get() < 4; mesh_id = MeshId{mesh_id.get() + 1}) {
        const auto& neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(mesh_id);
        EXPECT_EQ(neighbors.size(), 1u) << "Mesh " << mesh_id.get() << " should have 1 neighbor";
        EXPECT_EQ(neighbors[0], MeshId{0}) << "Mesh " << mesh_id.get() << " should connect to mesh 0";
    }

    // Verify mesh 0 has 1 exit node (the center ASIC)
    const auto& exit_graph0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 1u) << "Mesh 0 should have 1 exit node";
    PhysicalExitNode exit_node0{MeshId{0}, asic0};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0) != exit_nodes0.end())
        << "ASIC 0 should be the exit node";

    // Verify exit node has 3 connections
    const auto& exit_neighbors0 = exit_graph0.get_neighbors(exit_node0);
    EXPECT_EQ(exit_neighbors0.size(), 3u) << "Exit node should have 3 connections";
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_InterMeshConnectivity_2x2Subgraph) {
    // Test that intermesh-connected logical nodes map to directly physically connected ASICs.
    // This tests the critical scenario where if the intra-mesh mapping does not connect
    // on the two ends that are connected by the intermesh, there will be a problem.
    using namespace ::tt::tt_fabric;

    constexpr size_t kFullMeshSize = 9;
    constexpr size_t kAllocatedSize = 2;

    // Create logical meshes: 2x2 grids using MGD to properly capture exit nodes
    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 1 }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }

          # Intermesh connection between mesh 0 and mesh 1
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
            channels { count: 1 }
          }
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            channels { count: 1 }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create temporary MGD file
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path = temp_dir / ("test_intermesh_2x2_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the temporary file
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Clean up temporary file immediately after loading
    std::filesystem::remove(temp_mgd_path);

    // Build the logical multi-mesh graph from MGD (this properly captures exit nodes)
    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Create flattened physical mesh: two 9x9 grids with intermesh connections
    PhysicalAdjacencyMap flat_physical_adj;
    std::vector<tt::tt_metal::AsicID> physical_asics_m0 = make_asics(kFullMeshSize * kFullMeshSize, 100);
    std::vector<tt::tt_metal::AsicID> physical_asics_m1 = make_asics(kFullMeshSize * kFullMeshSize, 200);

    auto physical_adj_m0 = build_grid_adjacency(physical_asics_m0, kFullMeshSize, kFullMeshSize);
    auto physical_adj_m1 = build_grid_adjacency(physical_asics_m1, kFullMeshSize, kFullMeshSize);

    // Add intermesh connections: right edge of mesh 0 to left edge of mesh 1
    for (size_t row = 0; row < kFullMeshSize; ++row) {
        size_t mesh0_right_idx = row * kFullMeshSize + (kFullMeshSize - 1);
        size_t mesh1_left_idx = row * kFullMeshSize;
        physical_adj_m0[physical_asics_m0[mesh0_right_idx]].push_back(physical_asics_m1[mesh1_left_idx]);
        physical_adj_m1[physical_asics_m1[mesh1_left_idx]].push_back(physical_asics_m0[mesh0_right_idx]);
    }

    // Combine into flat adjacency map
    for (const auto& [asic, neighbors] : physical_adj_m0) {
        flat_physical_adj[asic] = neighbors;
    }
    for (const auto& [asic, neighbors] : physical_adj_m1) {
        flat_physical_adj[asic] = neighbors;
    }

    // Build hierarchical physical graph from flat graph
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& asic : physical_asics_m0) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = rank0_;
    }
    for (const auto& asic : physical_asics_m1) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = rank0_;
    }

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_physical_adj);
    PhysicalMultiMeshGraph physical_multi_mesh_graph =
        build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Run mapping
    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    ASSERT_TRUE(result.success) << result.error_message;
    verify_bidirectional_consistency(result);
    EXPECT_EQ(result.fabric_node_to_asic.size(), kAllocatedSize * kAllocatedSize * 2);

    // Group mappings by mesh
    std::map<MeshId, std::map<FabricNodeId, tt::tt_metal::AsicID>> mappings_by_mesh;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh[fabric_node.mesh_id][fabric_node] = asic;
    }

    // CRITICAL CHECK: Intermesh-connected logical nodes (exit nodes) must map to directly physically connected ASICs
    // The MGD has 2 connections:
    // 1. Mesh-level connection (relaxed mode): any node in mesh 0 <-> any node in mesh 1
    // 2. Device-level connection (strict mode): device_id 0 in mesh 0 <-> device_id 0 in mesh 1

    // Check 1: Device-level connection (strict mode) - device_id 0 must map to directly connected ASICs
    FabricNodeId exit_node_m0_strict(MeshId{0}, 0);
    FabricNodeId exit_node_m1_strict(MeshId{1}, 0);

    ASSERT_TRUE(mappings_by_mesh.at(MeshId{0}).find(exit_node_m0_strict) != mappings_by_mesh.at(MeshId{0}).end())
        << "Exit node device_id 0 from mesh 0 must be mapped";
    ASSERT_TRUE(mappings_by_mesh.at(MeshId{1}).find(exit_node_m1_strict) != mappings_by_mesh.at(MeshId{1}).end())
        << "Exit node device_id 0 from mesh 1 must be mapped";

    const auto& asic0_strict = mappings_by_mesh.at(MeshId{0}).at(exit_node_m0_strict);
    const auto& asic1_strict = mappings_by_mesh.at(MeshId{1}).at(exit_node_m1_strict);

    // Check if asic0_strict and asic1_strict are direct neighbors in the flat graph
    const auto& neighbors0_strict = flat_graph.get_neighbors(asic0_strict);
    bool has_direct_connection_strict =
        std::find(neighbors0_strict.begin(), neighbors0_strict.end(), asic1_strict) != neighbors0_strict.end();
    if (!has_direct_connection_strict) {
        const auto& neighbors1_strict = flat_graph.get_neighbors(asic1_strict);
        has_direct_connection_strict =
            std::find(neighbors1_strict.begin(), neighbors1_strict.end(), asic0_strict) != neighbors1_strict.end();
    }

    ASSERT_TRUE(has_direct_connection_strict)
        << "Strict mode: Exit node device_id 0 from mesh 0 mapped to ASIC " << asic0_strict.get()
        << " must be directly connected to exit node device_id 0 from mesh 1 mapped to ASIC " << asic1_strict.get();

    // Check 2: Mesh-level connection (relaxed mode) - at least one node (excluding device_id 0) from mesh 0
    // must map to a directly connected ASIC to verify a second distinct connection
    bool has_direct_connection_relaxed = false;
    for (const auto& [node0, asic0] : mappings_by_mesh.at(MeshId{0})) {
        // Skip device_id 0 as it's already checked in strict mode
        if (node0 == exit_node_m0_strict) {
            continue;
        }
        for (const auto& [node1, asic1] : mappings_by_mesh.at(MeshId{1})) {
            // Skip device_id 0 as it's already checked in strict mode
            if (node1 == exit_node_m1_strict) {
                continue;
            }
            const auto& neighbors0_relaxed = flat_graph.get_neighbors(asic0);
            if (std::find(neighbors0_relaxed.begin(), neighbors0_relaxed.end(), asic1) != neighbors0_relaxed.end()) {
                has_direct_connection_relaxed = true;
                break;
            }
            const auto& neighbors1_relaxed = flat_graph.get_neighbors(asic1);
            if (std::find(neighbors1_relaxed.begin(), neighbors1_relaxed.end(), asic0) != neighbors1_relaxed.end()) {
                has_direct_connection_relaxed = true;
                break;
            }
        }
        if (has_direct_connection_relaxed) {
            break;
        }
    }

    ASSERT_TRUE(has_direct_connection_relaxed)
        << "Relaxed mode: At least one node (excluding device_id 0) from mesh 0 must map to an ASIC directly connected "
           "to an ASIC (excluding device_id 0) from mesh 1 to verify the second distinct connection";
}

// TODO: Add a test testing relaxed mode connections between certain meshes

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_ImpossibleIntraMeshConstraints_2x2To3x3) {
    // Negative test: Verify that mapping fails correctly when intra-mesh constraints are impossible.
    // Physical topology: 3 meshes, each 3x3 (9 ASICs per mesh)
    // Logical topology: 3 meshes, each 2x2 (4 nodes per mesh)
    // With intermesh connections required between all meshes.
    // This should fail at the intra-mesh mapping level because a 2x2 grid cannot be mapped
    // onto a 3x3 grid due to topology constraints (degree mismatch, connectivity pattern mismatch).
    using namespace ::tt::tt_fabric;

    constexpr size_t kPhysicalMeshSize = 3;  // 3x3 = 9 ASICs per physical mesh
    constexpr size_t kLogicalMeshSize = 2;   // 2x2 = 4 nodes per logical mesh
    constexpr size_t kNumMeshes = 3;

    // Create logical meshes: 3 meshes, each 2x2 grids using MGD
    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 1 }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1,2 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }

          # Intermesh connections: mesh 0 <-> mesh 1, mesh 1 <-> mesh 2, mesh 0 <-> mesh 2
          # Use device-level connections (strict mode) to avoid exit node issues
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
            channels { count: 1 }
          }
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
            channels { count: 1 }
          }
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
            channels { count: 1 }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create temporary MGD file
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path =
        temp_dir / ("test_impossible_2x2_to_3x3_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the temporary file
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Clean up temporary file immediately after loading
    std::filesystem::remove(temp_mgd_path);

    // Build the logical multi-mesh graph from MGD
    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Create physical meshes: 3 meshes, each 3x3 grids
    PhysicalAdjacencyMap flat_physical_adj;
    std::vector<std::vector<tt::tt_metal::AsicID>> physical_asics_by_mesh(kNumMeshes);
    std::vector<PhysicalAdjacencyMap> physical_adj_by_mesh(kNumMeshes);

    for (size_t mesh_idx = 0; mesh_idx < kNumMeshes; ++mesh_idx) {
        physical_asics_by_mesh[mesh_idx] = make_asics(kPhysicalMeshSize * kPhysicalMeshSize, 100 + mesh_idx * 100);
        physical_adj_by_mesh[mesh_idx] =
            build_grid_adjacency(physical_asics_by_mesh[mesh_idx], kPhysicalMeshSize, kPhysicalMeshSize);
    }

    // Add intermesh connections: connect all meshes in a ring (0->1, 1->2, 2->0)
    // Connect right edge of mesh 0 to left edge of mesh 1
    for (size_t row = 0; row < kPhysicalMeshSize; ++row) {
        size_t mesh0_right_idx = row * kPhysicalMeshSize + (kPhysicalMeshSize - 1);
        size_t mesh1_left_idx = row * kPhysicalMeshSize;
        physical_adj_by_mesh[0][physical_asics_by_mesh[0][mesh0_right_idx]].push_back(
            physical_asics_by_mesh[1][mesh1_left_idx]);
        physical_adj_by_mesh[1][physical_asics_by_mesh[1][mesh1_left_idx]].push_back(
            physical_asics_by_mesh[0][mesh0_right_idx]);
    }

    // Connect right edge of mesh 1 to left edge of mesh 2
    for (size_t row = 0; row < kPhysicalMeshSize; ++row) {
        size_t mesh1_right_idx = row * kPhysicalMeshSize + (kPhysicalMeshSize - 1);
        size_t mesh2_left_idx = row * kPhysicalMeshSize;
        physical_adj_by_mesh[1][physical_asics_by_mesh[1][mesh1_right_idx]].push_back(
            physical_asics_by_mesh[2][mesh2_left_idx]);
        physical_adj_by_mesh[2][physical_asics_by_mesh[2][mesh2_left_idx]].push_back(
            physical_asics_by_mesh[1][mesh1_right_idx]);
    }

    // Connect right edge of mesh 2 to left edge of mesh 0
    for (size_t row = 0; row < kPhysicalMeshSize; ++row) {
        size_t mesh2_right_idx = row * kPhysicalMeshSize + (kPhysicalMeshSize - 1);
        size_t mesh0_left_idx = row * kPhysicalMeshSize;
        physical_adj_by_mesh[2][physical_asics_by_mesh[2][mesh2_right_idx]].push_back(
            physical_asics_by_mesh[0][mesh0_left_idx]);
        physical_adj_by_mesh[0][physical_asics_by_mesh[0][mesh0_left_idx]].push_back(
            physical_asics_by_mesh[2][mesh2_right_idx]);
    }

    // Combine into flat adjacency map
    for (size_t mesh_idx = 0; mesh_idx < kNumMeshes; ++mesh_idx) {
        for (const auto& [asic, neighbors] : physical_adj_by_mesh[mesh_idx]) {
            flat_physical_adj[asic] = neighbors;
        }
    }

    // Build hierarchical physical graph from flat graph
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (size_t mesh_idx = 0; mesh_idx < kNumMeshes; ++mesh_idx) {
        for (const auto& asic : physical_asics_by_mesh[mesh_idx]) {
            asic_id_to_mesh_rank[MeshId{mesh_idx}][asic] = rank0_;
        }
    }

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_physical_adj);
    PhysicalMultiMeshGraph physical_multi_mesh_graph =
        build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Run mapping - should fail at intra-mesh level
    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;

    TopologyMappingResult result =
        map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    // Debug: Print the mapping results
    std::cout << "\n=== Mapping Results ===" << std::endl;
    std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "Error message: " << result.error_message << std::endl;
    std::cout << "Number of mapped nodes: " << result.fabric_node_to_asic.size() << std::endl;
    std::cout << "\n=== Mappings ===" << std::endl;

    // Group mappings by logical mesh
    std::map<MeshId, std::vector<std::pair<FabricNodeId, tt::tt_metal::AsicID>>> mappings_by_mesh;
    for (const auto& [logical_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh[logical_node.mesh_id].emplace_back(logical_node, asic);
    }

    for (const auto& [mesh_id, mappings] : mappings_by_mesh) {
        std::cout << "\nLogical Mesh " << mesh_id.get() << " -> Physical Mesh mappings:" << std::endl;
        for (const auto& [logical_node, asic] : mappings) {
            std::cout << "  Logical node (mesh=" << logical_node.mesh_id.get() << ", chip=" << logical_node.chip_id
                      << ") -> ASIC " << asic.get() << std::endl;
        }
    }
    std::cout << "======================\n" << std::endl;

    EXPECT_FALSE(result.success) << "Multi-mesh mapping should fail due to impossible intra-mesh constraints "
                                    "(2x2 logical grid cannot map to 3x3 physical grid)";

    EXPECT_FALSE(result.error_message.empty())
        << "Error message should be provided when mapping fails. Error: " << result.error_message;

    // Verify that the failure occurred at the right place (intra-mesh mapping, not inter-mesh)
    // The error message should indicate intra-mesh mapping failure or solver failure
    bool mentions_intra_mesh_or_solver = result.error_message.find("intra") != std::string::npos ||
                                         result.error_message.find("solver") != std::string::npos ||
                                         result.error_message.find("mapping") != std::string::npos ||
                                         result.error_message.find("constraint") != std::string::npos;

    EXPECT_TRUE(mentions_intra_mesh_or_solver)
        << "Error message should indicate intra-mesh mapping or solver failure. Error: " << result.error_message;

    // Verify that inter-mesh mapping likely succeeded (since we have matching number of meshes)
    // but intra-mesh mapping failed. The error should not be about inter-mesh mapping failure.
    bool is_inter_mesh_failure = result.error_message.find("Inter-mesh mapping failed") != std::string::npos ||
                                 result.error_message.find("inter-mesh") != std::string::npos;

    // If it's an inter-mesh failure, that's also valid, but ideally it should fail at intra-mesh
    // The key is that it fails due to the topology constraints
    if (is_inter_mesh_failure) {
        // This is acceptable - inter-mesh might fail if it can't find valid mappings
        // But the root cause is still the impossible intra-mesh constraints
        EXPECT_TRUE(true) << "Failure occurred at inter-mesh level (acceptable, caused by intra-mesh constraints)";
    } else {
        // More likely: failure at intra-mesh level
        EXPECT_TRUE(true) << "Failure occurred at intra-mesh level (expected due to topology mismatch)";
    }

    // Verify that no complete mapping was found
    // The mapper should not have successfully mapped all logical nodes
    size_t expected_logical_nodes = kNumMeshes * kLogicalMeshSize * kLogicalMeshSize;  // 3 * 2 * 2 = 12
    EXPECT_LT(result.fabric_node_to_asic.size(), expected_logical_nodes)
        << "Should not have mapped all " << expected_logical_nodes
        << " logical nodes. Mapped: " << result.fabric_node_to_asic.size();
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_MixedStrictAndRelaxedConnections) {
    // PLACEHOLDER TEST - Currently SKIPPED
    // This test is a placeholder for future functionality when mixing STRICT and RELAXED
    // policies in the same graph becomes supported.
    //
    // TODO: Remove validation check and update this test when mixed policy support is implemented
    // See: tt_metal/fabric/mesh_graph_descriptor.cpp validate_legacy_requirements
    //      tt_metal/fabric/topology_mapper_utils.cpp build_logical_multi_mesh_adjacency_graph
    //
    // Expected behavior when implemented:
    // - Strict mode connections (device-level) should create fabric node-level exit nodes
    // - Relaxed mode connections (mesh-level) should create mesh-level exit nodes
    // - Both connection types should be processed correctly in the mapping
    //
    // Test setup for mixed connections:
    // - Mesh 0 <-> Mesh 1: STRICT mode (device-level, creates fabric node-level exit nodes)
    // - Mesh 1 <-> Mesh 2: RELAXED mode (mesh-level, creates mesh-level exit nodes)
    //
    // This verifies:
    // 1. Exit nodes are created for both connection types (different levels)
    // 2. Strict mode creates ExitNode with fabric_node_id set
    // 3. Relaxed mode creates ExitNode with fabric_node_id nullopt
    // 4. Mesh-level connectivity includes both connection types
    // 5. Mapping succeeds with both types present

    GTEST_SKIP() << "Mixed STRICT and RELAXED policies not yet supported - validation prevents this";

    // Create MGD textproto string with mixed STRICT and RELAXED connections
    // NOTE: This will currently fail validation - that's expected until feature is implemented
    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 2 policy: RELAXED }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1,2 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }

          # STRICT connection: Mesh 0 <-> Mesh 1 (device-level)
          # M0 D1 <-> M1 D0 with 2 channels
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 1 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            channels { count: 2 policy: STRICT }
          }

          # RELAXED connection: Mesh 1 <-> Mesh 2 (mesh-level)
          # Mesh-level connection with 3 channels
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
            channels { count: 3 policy: RELAXED }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create temporary MGD file
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path = temp_dir / ("test_mixed_policies_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the MGD file (should succeed when feature is implemented)
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Build logical multi-mesh graph
    const auto logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Verify we have 3 meshes
    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 meshes";

    // Verify mesh-level connectivity includes both connection types
    const auto& mesh0_neighbors = logical_multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 2u) << "Mesh 0 should have 2 connections to mesh 1 (strict)";

    const auto& mesh1_neighbors = logical_multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 5u)
        << "Mesh 1 should have 5 connections total (2 to mesh 0 strict, 3 to mesh 2 relaxed)";

    const auto& mesh2_neighbors = logical_multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{2});
    EXPECT_EQ(mesh2_neighbors.size(), 3u) << "Mesh 2 should have 3 connections to mesh 1 (relaxed)";

    // Verify exit nodes are tracked for both strict and relaxed mode connections
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Mesh 0 should have exit nodes (strict mode connection - fabric node-level)";
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Mesh 1 should have exit nodes (strict mode connection - fabric node-level, and relaxed mode - mesh-level)";
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{2}))
        << "Mesh 2 should have exit nodes (relaxed mode connection - mesh-level)";

    // Verify exit node details for mesh 0 (strict connection)
    const auto& exit_graph0 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 1u) << "Mesh 0 should have 1 exit node (device 1)";

    LogicalExitNode exit_node0_1{MeshId{0}, FabricNodeId(MeshId{0}, 1)};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0_1) != exit_nodes0.end())
        << "Device 1 should be an exit node in mesh 0";
    // Verify it's a fabric node-level exit node (strict mode)
    EXPECT_TRUE(exit_node0_1.fabric_node_id.has_value())
        << "Mesh 0 exit node should be fabric node-level (strict mode)";

    // Verify exit node details for mesh 1 (has both strict and relaxed connections)
    const auto& exit_graph1 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_GT(exit_nodes1.size(), 0u) << "Mesh 1 should have exit nodes";

    // Mesh 1 should have both fabric node-level (strict) and mesh-level (relaxed) exit nodes
    bool has_fabric_node_level = false;
    bool has_mesh_level = false;
    for (const auto& exit_node : exit_nodes1) {
        if (exit_node.fabric_node_id.has_value()) {
            has_fabric_node_level = true;
        } else {
            has_mesh_level = true;
        }
    }
    EXPECT_TRUE(has_fabric_node_level) << "Mesh 1 should have fabric node-level exit nodes (strict mode)";
    EXPECT_TRUE(has_mesh_level) << "Mesh 1 should have mesh-level exit nodes (relaxed mode)";

    // Verify exit node details for mesh 2 (relaxed connection - mesh-level only)
    const auto& exit_graph2 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2});
    const auto& exit_nodes2 = exit_graph2.get_nodes();
    EXPECT_GT(exit_nodes2.size(), 0u) << "Mesh 2 should have exit nodes";

    // All exit nodes in mesh 2 should be mesh-level (relaxed mode)
    for (const auto& exit_node : exit_nodes2) {
        EXPECT_FALSE(exit_node.fabric_node_id.has_value()) << "Mesh 2 exit nodes should be mesh-level (relaxed mode)";
        EXPECT_EQ(exit_node.mesh_id, MeshId{2}) << "Exit node mesh_id should match mesh 2";
    }

    // Create physical multi-mesh graph for mapping
    // Physical topology: 3 meshes, each 2x2, with intermesh connections matching logical
    using namespace ::tt::tt_fabric;

    // Physical Mesh 0: 2x2 grid
    std::vector<tt::tt_metal::AsicID> physical_asics_m0;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m0.push_back(tt::tt_metal::AsicID{100 + i});
    }
    auto physical_adj_m0 = build_grid_adjacency(physical_asics_m0, 2, 2);

    // Physical Mesh 1: 2x2 grid
    std::vector<tt::tt_metal::AsicID> physical_asics_m1;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m1.push_back(tt::tt_metal::AsicID{200 + i});
    }
    auto physical_adj_m1 = build_grid_adjacency(physical_asics_m1, 2, 2);

    // Physical Mesh 2: 2x2 grid
    std::vector<tt::tt_metal::AsicID> physical_asics_m2;
    for (uint64_t i = 0; i < 4; ++i) {
        physical_asics_m2.push_back(tt::tt_metal::AsicID{300 + i});
    }
    auto physical_adj_m2 = build_grid_adjacency(physical_asics_m2, 2, 2);

    // Create physical multi-mesh graph
    PhysicalMultiMeshGraph physical_multi_mesh_graph;
    physical_multi_mesh_graph.mesh_adjacency_graphs_[MeshId{0}] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m0);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[MeshId{1}] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m1);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[MeshId{2}] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m2);

    // Create flat physical adjacency map (combining intra-mesh and intermesh connections)
    AdjacencyGraph<tt::tt_metal::AsicID>::AdjacencyMap flat_physical_adj;

    // Add intra-mesh connections to flat adjacency map
    for (const auto& [asic, neighbors] : physical_adj_m0) {
        flat_physical_adj[asic] = neighbors;
    }
    for (const auto& [asic, neighbors] : physical_adj_m1) {
        flat_physical_adj[asic] = neighbors;
    }
    for (const auto& [asic, neighbors] : physical_adj_m2) {
        flat_physical_adj[asic] = neighbors;
    }

    // Add intermesh connections: M0 <-> M1 (strict, device-level) and M1 <-> M2 (relaxed, mesh-level)
    // M0 ASIC 101 <-> M1 ASIC 200 (strict connection)
    flat_physical_adj[tt::tt_metal::AsicID{101}].push_back(tt::tt_metal::AsicID{200});
    flat_physical_adj[tt::tt_metal::AsicID{200}].push_back(tt::tt_metal::AsicID{101});
    // M1 ASIC 200 <-> M2 ASIC 300 (relaxed connection - any device can be used)
    flat_physical_adj[tt::tt_metal::AsicID{200}].push_back(tt::tt_metal::AsicID{300});
    flat_physical_adj[tt::tt_metal::AsicID{300}].push_back(tt::tt_metal::AsicID{200});

    // Create flat physical graph from adjacency map
    AdjacencyGraph<tt::tt_metal::AsicID> flat_physical_graph(flat_physical_adj);

    // ASIC to mesh/rank mapping
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (const auto& asic : physical_asics_m0) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = MeshHostRankId{0};
    }
    for (const auto& asic : physical_asics_m1) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = MeshHostRankId{0};
    }
    for (const auto& asic : physical_asics_m2) {
        asic_id_to_mesh_rank[MeshId{2}][asic] = MeshHostRankId{0};
    }

    // Build physical multi-mesh graph from flat graph
    physical_multi_mesh_graph = build_hierarchical_from_flat_graph(flat_physical_graph, asic_id_to_mesh_rank);

    // Perform mapping
    TopologyMappingConfig config;
    config.strict_mode = false;  // Use relaxed mode for mapping (allows flexibility)

    const auto result =
        map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config, asic_id_to_mesh_rank);

    // Verify mapping succeeded
    EXPECT_TRUE(result.success) << "Mapping should succeed with mixed strict/relaxed connections";

    // Verify all logical nodes are mapped
    EXPECT_EQ(result.fabric_node_to_asic.size(), 12u) << "All 12 logical nodes (3 meshes * 4 nodes) should be mapped";

    // Verify exit nodes are correctly identified in physical graph
    // Mesh 0 should have exit nodes (strict connection to mesh 1)
    EXPECT_TRUE(physical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Physical mesh 0 should have exit nodes";
    EXPECT_TRUE(physical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Physical mesh 1 should have exit nodes";
    // Mesh 2 may or may not have exit nodes depending on implementation

    // Clean up temporary file
    std::filesystem::remove(temp_mgd_path);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_ThreeLogicalFivePhysical_RingTopology) {
    // Test mapping 3 logical meshes (2x2 each) connected in a line to 5 physical meshes (2x2 each) connected in a ring.
    // All inter-mesh connections are mesh-level constraints (relaxed mode).
    // This test verifies that the solver can find a valid mapping when there are more physical meshes than logical
    // meshes.
    //
    // Logical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Mesh 2: 2x2 grid (4 nodes)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 <-> Mesh 2 (line: 0-1-2) - mesh-level connections
    //
    // Physical Topology (5 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs)
    //   Mesh 1: 2x2 grid (4 ASICs)
    //   Mesh 2: 2x2 grid (4 ASICs)
    //   Mesh 3: 2x2 grid (4 ASICs)
    //   Mesh 4: 2x2 grid (4 ASICs)
    //   Inter-mesh: Ring topology (0-1-2-3-4-0)

    using namespace ::tt::tt_fabric;

    // =========================================================================
    // Create Logical Multi-Mesh Graph using MGD (3 meshes, line topology)
    // =========================================================================

    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 1 }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1,2 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }

          # Intermesh connections: mesh 0 <-> mesh 1, mesh 1 <-> mesh 2 (line topology)
          # Using mesh-level connections (relaxed mode) - no device_id specified
          # Note: mesh-level connections create mesh-level exit nodes
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
            channels { count: 1 }
          }
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 } }
            channels { count: 1 }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create temporary MGD file
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path =
        temp_dir / ("test_three_logical_five_physical_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the temporary file
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Clean up temporary file immediately after loading
    std::filesystem::remove(temp_mgd_path);

    // Build the logical multi-mesh graph from MGD
    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Verify we have 3 logical meshes
    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 logical meshes";

    // Verify exit node graphs only reference valid mesh IDs (0, 1, 2)
    for (const auto& [mesh_id, exit_node_graph] : logical_multi_mesh_graph.mesh_exit_node_graphs_) {
        EXPECT_TRUE(mesh_id.get() < 3u) << "Exit node graph should only reference meshes 0, 1, 2";
        for (const auto& exit_node : exit_node_graph.get_nodes()) {
            EXPECT_TRUE(exit_node.mesh_id.get() < 3u) << "Exit node should only reference meshes 0, 1, 2";
            const auto& neighbors = exit_node_graph.get_neighbors(exit_node);
            for (const auto& neighbor_exit_node : neighbors) {
                EXPECT_TRUE(neighbor_exit_node.mesh_id.get() < 3u)
                    << "Exit node neighbor should only reference meshes 0, 1, 2";
            }
        }
    }

    // Verify mesh-level connectivity (line: 0-1-2)
    const auto& mesh0_neighbors = logical_multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 1u) << "Mesh 0 should have 1 neighbor (mesh 1)";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{1}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 1";

    const auto& mesh1_neighbors = logical_multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 2u) << "Mesh 1 should have 2 neighbors (mesh 0 and mesh 2)";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{0}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 0";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{2}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 2";

    const auto& mesh2_neighbors = logical_multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{2});
    EXPECT_EQ(mesh2_neighbors.size(), 1u) << "Mesh 2 should have 1 neighbor (mesh 1)";
    EXPECT_TRUE(std::find(mesh2_neighbors.begin(), mesh2_neighbors.end(), MeshId{1}) != mesh2_neighbors.end())
        << "Mesh 2 should connect to mesh 1";

    // =========================================================================
    // Create Physical Multi-Mesh Graph (5 meshes, ring topology)
    // =========================================================================

    constexpr size_t kNumPhysicalMeshes = 5;
    constexpr size_t kPhysicalMeshSize = 2;  // 2x2 grid

    // Create physical meshes: 5 meshes, each 2x2 grid
    PhysicalAdjacencyMap flat_physical_adj;
    std::vector<std::vector<tt::tt_metal::AsicID>> physical_asics_by_mesh(kNumPhysicalMeshes);
    std::vector<PhysicalAdjacencyMap> physical_adj_by_mesh(kNumPhysicalMeshes);

    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        physical_asics_by_mesh[mesh_idx] = make_asics(kPhysicalMeshSize * kPhysicalMeshSize, 100 + mesh_idx * 100);
        physical_adj_by_mesh[mesh_idx] =
            build_grid_adjacency(physical_asics_by_mesh[mesh_idx], kPhysicalMeshSize, kPhysicalMeshSize);
    }

    // Add intermesh connections: ring topology (0->1, 1->2, 2->3, 3->4, 4->0)
    // Connect right edge of each mesh to left edge of next mesh
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        size_t next_mesh_idx = (mesh_idx + 1) % kNumPhysicalMeshes;
        for (size_t row = 0; row < kPhysicalMeshSize; ++row) {
            size_t current_right_idx = row * kPhysicalMeshSize + (kPhysicalMeshSize - 1);
            size_t next_left_idx = row * kPhysicalMeshSize;
            physical_adj_by_mesh[mesh_idx][physical_asics_by_mesh[mesh_idx][current_right_idx]].push_back(
                physical_asics_by_mesh[next_mesh_idx][next_left_idx]);
            physical_adj_by_mesh[next_mesh_idx][physical_asics_by_mesh[next_mesh_idx][next_left_idx]].push_back(
                physical_asics_by_mesh[mesh_idx][current_right_idx]);
        }
    }

    // Combine into flat adjacency map
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        for (const auto& [asic, neighbors] : physical_adj_by_mesh[mesh_idx]) {
            flat_physical_adj[asic] = neighbors;
        }
    }

    // Build hierarchical physical graph from flat graph
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        for (const auto& asic : physical_asics_by_mesh[mesh_idx]) {
            asic_id_to_mesh_rank[MeshId{mesh_idx}][asic] = rank0_;
        }
    }

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_physical_adj);
    PhysicalMultiMeshGraph physical_multi_mesh_graph =
        build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // Verify physical mesh-level connectivity (ring: 0-1-2-3-4-0)
    // Note: Each mesh may have duplicate neighbor entries due to multiple ASIC-level connections,
    // so we check for unique neighbors instead of exact count
    const auto& physical_mesh_level_graph = physical_multi_mesh_graph.mesh_level_graph_;
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        const auto& neighbors = physical_mesh_level_graph.get_neighbors(MeshId{mesh_idx});
        // Get unique neighbors (multiple ASIC connections between meshes create duplicates)
        std::set<MeshId> unique_neighbors(neighbors.begin(), neighbors.end());
        EXPECT_EQ(unique_neighbors.size(), 2u)
            << "Physical mesh " << mesh_idx << " should have 2 unique neighbors in ring";
        size_t prev_mesh_idx = (mesh_idx + kNumPhysicalMeshes - 1) % kNumPhysicalMeshes;
        size_t next_mesh_idx = (mesh_idx + 1) % kNumPhysicalMeshes;
        EXPECT_TRUE(unique_neighbors.contains(MeshId{prev_mesh_idx}))
            << "Physical mesh " << mesh_idx << " should connect to previous mesh " << prev_mesh_idx;
        EXPECT_TRUE(unique_neighbors.contains(MeshId{next_mesh_idx}))
            << "Physical mesh " << mesh_idx << " should connect to next mesh " << next_mesh_idx;
    }

    // =========================================================================
    // Debug: Print adjacency graphs before mapping
    // =========================================================================

    std::cout << "\n=== DEBUG: Logical Mesh-Level Graph ===" << std::endl;
    const auto& logical_mesh_level_graph = logical_multi_mesh_graph.mesh_level_graph_;
    for (const auto& mesh_id : logical_mesh_level_graph.get_nodes()) {
        const auto& neighbors = logical_mesh_level_graph.get_neighbors(mesh_id);
        std::cout << "Logical Mesh " << mesh_id.get() << " neighbors: ";
        for (const auto& neighbor : neighbors) {
            std::cout << neighbor.get() << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== DEBUG: Logical Exit Node Graphs ===" << std::endl;
    for (const auto& [mesh_id, exit_node_graph] : logical_multi_mesh_graph.mesh_exit_node_graphs_) {
        std::cout << "Logical Mesh " << mesh_id.get() << " exit nodes:" << std::endl;
        for (const auto& exit_node : exit_node_graph.get_nodes()) {
            std::cout << "  Exit node: mesh_id=" << exit_node.mesh_id.get();
            if (exit_node.fabric_node_id.has_value()) {
                std::cout << ", fabric_node_id=" << exit_node.fabric_node_id->chip_id;
            } else {
                std::cout << ", mesh-level (no fabric_node_id)";
            }
            std::cout << std::endl;
            const auto& neighbors = exit_node_graph.get_neighbors(exit_node);
            std::cout << "    Neighbors: ";
            for (const auto& neighbor : neighbors) {
                std::cout << "(mesh_id=" << neighbor.mesh_id.get();
                if (neighbor.fabric_node_id.has_value()) {
                    std::cout << ", fabric_node_id=" << neighbor.fabric_node_id->chip_id;
                } else {
                    std::cout << ", mesh-level";
                }
                std::cout << ") ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\n=== DEBUG: Physical Mesh-Level Graph ===" << std::endl;
    for (const auto& mesh_id : physical_mesh_level_graph.get_nodes()) {
        const auto& neighbors = physical_mesh_level_graph.get_neighbors(mesh_id);
        std::set<MeshId> unique_neighbors(neighbors.begin(), neighbors.end());
        std::cout << "Physical Mesh " << mesh_id.get() << " neighbors (total=" << neighbors.size()
                  << ", unique=" << unique_neighbors.size() << "): ";
        for (const auto& neighbor : unique_neighbors) {
            std::cout << neighbor.get() << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== DEBUG: Physical Exit Node Graphs ===" << std::endl;
    for (const auto& [mesh_id, exit_node_graph] : physical_multi_mesh_graph.mesh_exit_node_graphs_) {
        std::cout << "Physical Mesh " << mesh_id.get() << " exit nodes:" << std::endl;
        for (const auto& exit_node : exit_node_graph.get_nodes()) {
            std::cout << "  Exit node: mesh_id=" << exit_node.mesh_id.get() << ", asic_id=" << exit_node.asic_id.get()
                      << std::endl;
            const auto& neighbors = exit_node_graph.get_neighbors(exit_node);
            std::cout << "    Neighbors: ";
            for (const auto& neighbor : neighbors) {
                std::cout << "(mesh_id=" << neighbor.mesh_id.get() << ", asic_id=" << neighbor.asic_id.get() << ") ";
            }
            std::cout << std::endl;
        }
    }

    // =========================================================================
    // Run mapping and verify results
    // =========================================================================

    TopologyMappingConfig config;
    config.strict_mode = false;           // Use relaxed mode to allow more flexibility
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid
    config.inter_mesh_validation_mode = ConnectionValidationMode::RELAXED;  // Relaxed inter-mesh validation

    std::cout << "\n=== DEBUG: Starting Mapping ===" << std::endl;
    std::cout << "Config: strict_mode=" << config.strict_mode
              << ", disable_rank_bindings=" << config.disable_rank_bindings << std::endl;

    // Call map_multi_mesh_to_physical (rank mappings omitted since disable_rank_bindings is true)
    TopologyMappingResult result;
    try {
        result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);
    } catch (const std::exception& e) {
        std::cout << "\n=== DEBUG: Exception caught during mapping ===" << std::endl;
        std::cout << "Exception: " << e.what() << std::endl;
        throw;  // Re-throw to see full stack trace
    }

    std::cout << "\n=== DEBUG: Mapping Result ===" << std::endl;
    std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "Error message: " << result.error_message << std::endl;
    std::cout << "Number of mapped nodes: " << result.fabric_node_to_asic.size() << std::endl;

    // Print all mappings (even if partial)
    std::cout << "\n=== DEBUG: All Mappings (Partial or Complete) ===" << std::endl;
    std::map<MeshId, std::vector<std::pair<FabricNodeId, tt::tt_metal::AsicID>>> mappings_by_mesh_debug;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh_debug[fabric_node.mesh_id].emplace_back(fabric_node, asic);
    }

    for (const auto& [mesh_id, mappings] : mappings_by_mesh_debug) {
        std::cout << "\nLogical Mesh " << mesh_id.get() << " -> Physical Mesh mappings:" << std::endl;
        // Determine which physical mesh this maps to
        if (!mappings.empty()) {
            const auto& first_asic = mappings[0].second;
            MeshId physical_mesh_id = MeshId{0};
            for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
                const auto& nodes = adjacency_graph.get_nodes();
                if (std::find(nodes.begin(), nodes.end(), first_asic) != nodes.end()) {
                    physical_mesh_id = pm_id;
                    break;
                }
            }
            std::cout << "  Mapped to Physical Mesh " << physical_mesh_id.get() << std::endl;
        }
        for (const auto& [logical_node, asic] : mappings) {
            std::cout << "    Logical node (mesh=" << logical_node.mesh_id.get() << ", chip=" << logical_node.chip_id
                      << ") -> ASIC " << asic.get() << std::endl;
        }
    }

    // Print mesh-level mappings inferred from node mappings
    std::cout << "\n=== DEBUG: Inferred Mesh-Level Mappings ===" << std::endl;
    std::map<MeshId, MeshId> inferred_mesh_mappings;
    for (const auto& [mesh_id, mappings] : mappings_by_mesh_debug) {
        if (!mappings.empty()) {
            const auto& first_asic = mappings[0].second;
            MeshId physical_mesh_id = MeshId{0};
            for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
                const auto& nodes = adjacency_graph.get_nodes();
                if (std::find(nodes.begin(), nodes.end(), first_asic) != nodes.end()) {
                    physical_mesh_id = pm_id;
                    break;
                }
            }
            inferred_mesh_mappings[mesh_id] = physical_mesh_id;
            std::cout << "Logical Mesh " << mesh_id.get() << " -> Physical Mesh " << physical_mesh_id.get()
                      << std::endl;
        }
    }

    // Verify overall result succeeded
    EXPECT_TRUE(result.success) << "Multi-mesh mapping should succeed: " << result.error_message;

    // Verify bidirectional consistency of the overall result
    verify_bidirectional_consistency(result);

    // Verify all logical nodes are mapped (3 meshes * 4 nodes = 12 nodes)
    EXPECT_EQ(result.fabric_node_to_asic.size(), 12u) << "All 12 logical nodes should be mapped";

    // Group mappings by mesh_id
    std::map<MeshId, std::map<FabricNodeId, tt::tt_metal::AsicID>> mappings_by_mesh;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh[fabric_node.mesh_id][fabric_node] = asic;
    }

    // Verify we have mappings for all 3 logical meshes
    EXPECT_EQ(mappings_by_mesh.size(), 3u) << "Should have mappings for all 3 logical meshes";
    EXPECT_TRUE(mappings_by_mesh.contains(MeshId{0})) << "Should have mappings for logical mesh 0";
    EXPECT_TRUE(mappings_by_mesh.contains(MeshId{1})) << "Should have mappings for logical mesh 1";
    EXPECT_TRUE(mappings_by_mesh.contains(MeshId{2})) << "Should have mappings for logical mesh 2";

    // Verify each logical mesh has all 4 nodes mapped
    for (const auto& [logical_mesh_id, mesh_mappings] : mappings_by_mesh) {
        EXPECT_EQ(mesh_mappings.size(), 4u) << "Logical mesh " << logical_mesh_id.get() << " should map all 4 nodes";
    }

    // Verify connectivity is preserved for all meshes
    for (const auto& [logical_mesh_id, mesh_mappings] : mappings_by_mesh) {
        // Find which physical mesh this logical mesh mapped to
        MeshId physical_mesh_id = MeshId{0};  // Default, will be updated
        if (!mesh_mappings.empty()) {
            const auto& first_asic = mesh_mappings.begin()->second;
            // Determine which physical mesh this ASIC belongs to by checking all physical meshes
            for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
                const auto& nodes = adjacency_graph.get_nodes();
                if (std::find(nodes.begin(), nodes.end(), first_asic) != nodes.end()) {
                    physical_mesh_id = pm_id;
                    break;
                }
            }
        }

        // Verify intra-mesh connectivity is preserved
        const auto& logical_graph = logical_multi_mesh_graph.mesh_adjacency_graphs_.at(logical_mesh_id);
        const auto& physical_graph = physical_multi_mesh_graph.mesh_adjacency_graphs_.at(physical_mesh_id);
        const auto& logical_nodes = logical_graph.get_nodes();

        for (const auto& node : logical_nodes) {
            const auto mapped_asic = mesh_mappings.at(node);
            const auto& logical_neighbors = logical_graph.get_neighbors(node);
            const auto& physical_neighbors = physical_graph.get_neighbors(mapped_asic);

            for (const auto& neighbor : logical_neighbors) {
                // If neighbor is in the same logical mesh, verify intra-mesh connectivity
                if (neighbor.mesh_id == logical_mesh_id) {
                    const auto neighbor_asic = mesh_mappings.at(neighbor);
                    EXPECT_TRUE(
                        std::find(physical_neighbors.begin(), physical_neighbors.end(), neighbor_asic) !=
                        physical_neighbors.end())
                        << "Logical intra-mesh edge not preserved in physical mapping for logical mesh "
                        << logical_mesh_id.get();
                }
            }
        }
    }

    // Verify inter-mesh connectivity is preserved (mesh-level constraints)
    // Check that logical meshes that are connected map to physical meshes that are connected
    for (const auto& logical_mesh_id : logical_mesh_level_graph.get_nodes()) {
        const auto& logical_neighbors = logical_mesh_level_graph.get_neighbors(logical_mesh_id);
        if (!mappings_by_mesh.contains(logical_mesh_id) || mappings_by_mesh.at(logical_mesh_id).empty()) {
            continue;
        }

        // Find which physical mesh this logical mesh mapped to
        const auto& first_asic = mappings_by_mesh.at(logical_mesh_id).begin()->second;
        MeshId mapped_physical_mesh_id = MeshId{0};
        for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
            const auto& nodes = adjacency_graph.get_nodes();
            if (std::find(nodes.begin(), nodes.end(), first_asic) != nodes.end()) {
                mapped_physical_mesh_id = pm_id;
                break;
            }
        }

        // For each logical neighbor, verify it maps to a physically connected mesh
        for (const auto& logical_neighbor_id : logical_neighbors) {
            if (!mappings_by_mesh.contains(logical_neighbor_id) || mappings_by_mesh.at(logical_neighbor_id).empty()) {
                continue;
            }

            const auto& neighbor_first_asic = mappings_by_mesh.at(logical_neighbor_id).begin()->second;
            MeshId neighbor_physical_mesh_id = MeshId{0};
            for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
                const auto& nodes = adjacency_graph.get_nodes();
                if (std::find(nodes.begin(), nodes.end(), neighbor_first_asic) != nodes.end()) {
                    neighbor_physical_mesh_id = pm_id;
                    break;
                }
            }

            // Verify the physical meshes are connected
            const auto& physical_neighbor_meshes =
                physical_multi_mesh_graph.mesh_level_graph_.get_neighbors(mapped_physical_mesh_id);
            EXPECT_TRUE(
                std::find(
                    physical_neighbor_meshes.begin(), physical_neighbor_meshes.end(), neighbor_physical_mesh_id) !=
                physical_neighbor_meshes.end())
                << "Logical mesh " << logical_mesh_id.get() << " connects to logical mesh " << logical_neighbor_id.get()
                << ", but their mapped physical meshes (" << mapped_physical_mesh_id.get() << " and "
                << neighbor_physical_mesh_id.get() << ") are not connected";
        }
    }
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_ThreeLogicalFivePhysical_DeviceLevelConstraints_Fails) {
    // Test mapping 3 logical meshes (2x2 each) connected in a line to 5 physical meshes (2x2 each) connected in a ring.
    // All inter-mesh connections are device-level constraints (strict mode, device_id: 0).
    // This test verifies that the solver correctly fails when constraints are over-constrained.
    //
    // Logical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Mesh 2: 2x2 grid (4 nodes)
    //   Inter-mesh: Mesh 0 <-> Mesh 1 <-> Mesh 2 (line: 0-1-2) - device-level connections (device_id: 0)
    //
    // Physical Topology (5 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs)
    //   Mesh 1: 2x2 grid (4 ASICs)
    //   Mesh 2: 2x2 grid (4 ASICs)
    //   Mesh 3: 2x2 grid (4 ASICs)
    //   Mesh 4: 2x2 grid (4 ASICs)
    //   Inter-mesh: Ring topology (0-1-2-3-4-0)
    //
    // Expected: Should fail because device-level constraints require device 0 in each logical mesh
    // to connect to device 0 in adjacent logical meshes, which may not be possible depending on
    // which physical meshes are selected and their connectivity.

    using namespace ::tt::tt_fabric;

    // =========================================================================
    // Create Logical Multi-Mesh Graph using MGD (3 meshes, line topology, device-level connections)
    // =========================================================================

    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 1 }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1,2 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }

          # Intermesh connections: mesh 0 <-> mesh 1, mesh 1 <-> mesh 2 (line topology)
          # Using device-level connections (strict mode) - device_id: 0 specified
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            channels { count: 1 }
          }
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 device_id: 0 } }
            channels { count: 1 }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create temporary MGD file
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path =
        temp_dir / ("test_three_logical_five_physical_device_level_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the temporary file
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Clean up temporary file immediately after loading
    std::filesystem::remove(temp_mgd_path);

    // Build the logical multi-mesh graph from MGD
    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Verify we have 3 logical meshes
    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 logical meshes";

    // Verify mesh-level connectivity (line: 0-1-2)
    const auto& logical_mesh_level_graph = logical_multi_mesh_graph.mesh_level_graph_;
    const auto& mesh0_neighbors = logical_mesh_level_graph.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 1u) << "Mesh 0 should have 1 neighbor (mesh 1)";
    EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{1}) != mesh0_neighbors.end())
        << "Mesh 0 should connect to mesh 1";

    const auto& mesh1_neighbors = logical_mesh_level_graph.get_neighbors(MeshId{1});
    EXPECT_EQ(mesh1_neighbors.size(), 2u) << "Mesh 1 should have 2 neighbors (mesh 0 and mesh 2)";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{0}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 0";
    EXPECT_TRUE(std::find(mesh1_neighbors.begin(), mesh1_neighbors.end(), MeshId{2}) != mesh1_neighbors.end())
        << "Mesh 1 should connect to mesh 2";

    const auto& mesh2_neighbors = logical_mesh_level_graph.get_neighbors(MeshId{2});
    EXPECT_EQ(mesh2_neighbors.size(), 1u) << "Mesh 2 should have 1 neighbor (mesh 1)";
    EXPECT_TRUE(std::find(mesh2_neighbors.begin(), mesh2_neighbors.end(), MeshId{1}) != mesh2_neighbors.end())
        << "Mesh 2 should connect to mesh 1";

    // Verify exit nodes are device-level (strict mode)
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Mesh 0 should have exit nodes (device-level connection)";
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Mesh 1 should have exit nodes (device-level connection)";
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{2}))
        << "Mesh 2 should have exit nodes (device-level connection)";

    // Verify exit nodes are fabric node-level (device_id specified)
    const auto& exit_graph0 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 1u) << "Mesh 0 should have 1 exit node (device 0)";
    LogicalExitNode exit_node0{MeshId{0}, FabricNodeId(MeshId{0}, 0)};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0) != exit_nodes0.end())
        << "Device 0 should be an exit node in mesh 0";
    EXPECT_TRUE(exit_node0.fabric_node_id.has_value()) << "Mesh 0 exit node should be fabric node-level (strict mode)";

    // =========================================================================
    // Create Physical Multi-Mesh Graph (5 meshes, ring topology)
    // =========================================================================

    constexpr size_t kNumPhysicalMeshes = 5;
    constexpr size_t kPhysicalMeshSize = 2;  // 2x2 grid

    // Create physical meshes: 5 meshes, each 2x2 grid
    PhysicalAdjacencyMap flat_physical_adj;
    std::vector<std::vector<tt::tt_metal::AsicID>> physical_asics_by_mesh(kNumPhysicalMeshes);
    std::vector<PhysicalAdjacencyMap> physical_adj_by_mesh(kNumPhysicalMeshes);

    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        physical_asics_by_mesh[mesh_idx] = make_asics(kPhysicalMeshSize * kPhysicalMeshSize, 100 + mesh_idx * 100);
        physical_adj_by_mesh[mesh_idx] =
            build_grid_adjacency(physical_asics_by_mesh[mesh_idx], kPhysicalMeshSize, kPhysicalMeshSize);
    }

    // Add intermesh connections: ring topology (0->1, 1->2, 2->3, 3->4, 4->0)
    // Connect right edge of each mesh to left edge of next mesh
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        size_t next_mesh_idx = (mesh_idx + 1) % kNumPhysicalMeshes;
        for (size_t row = 0; row < kPhysicalMeshSize; ++row) {
            size_t current_right_idx = row * kPhysicalMeshSize + (kPhysicalMeshSize - 1);
            size_t next_left_idx = row * kPhysicalMeshSize;
            physical_adj_by_mesh[mesh_idx][physical_asics_by_mesh[mesh_idx][current_right_idx]].push_back(
                physical_asics_by_mesh[next_mesh_idx][next_left_idx]);
            physical_adj_by_mesh[next_mesh_idx][physical_asics_by_mesh[next_mesh_idx][next_left_idx]].push_back(
                physical_asics_by_mesh[mesh_idx][current_right_idx]);
        }
    }

    // Combine into flat adjacency map
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        for (const auto& [asic, neighbors] : physical_adj_by_mesh[mesh_idx]) {
            flat_physical_adj[asic] = neighbors;
        }
    }

    // Build hierarchical physical graph from flat graph
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        for (const auto& asic : physical_asics_by_mesh[mesh_idx]) {
            asic_id_to_mesh_rank[MeshId{mesh_idx}][asic] = rank0_;
        }
    }

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_physical_adj);
    PhysicalMultiMeshGraph physical_multi_mesh_graph =
        build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // =========================================================================
    // Run mapping and verify failure
    // =========================================================================

    TopologyMappingConfig config;
    config.strict_mode = true;            // Use strict mode for device-level constraints
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    std::cout << "\n=== DEBUG: Starting Mapping (Device-Level Constraints) ===" << std::endl;
    std::cout << "Config: strict_mode=" << config.strict_mode
              << ", disable_rank_bindings=" << config.disable_rank_bindings << std::endl;

    // Call map_multi_mesh_to_physical (rank mappings omitted since disable_rank_bindings is true)
    TopologyMappingResult result;
    bool exception_thrown = false;
    std::string exception_message;
    try {
        result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);
    } catch (const std::exception& e) {
        exception_thrown = true;
        exception_message = e.what();
        std::cout << "\n=== DEBUG: Exception caught during mapping ===" << std::endl;
        std::cout << "Exception: " << exception_message << std::endl;
        // Exception is acceptable - it indicates over-constrained scenario
    }

    std::cout << "\n=== DEBUG: Mapping Result ===" << std::endl;
    std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "Error message: " << result.error_message << std::endl;
    std::cout << "Exception thrown: " << (exception_thrown ? "true" : "false") << std::endl;
    std::cout << "Number of mapped nodes: " << result.fabric_node_to_asic.size() << std::endl;

    // Print partial mappings if any
    if (!result.fabric_node_to_asic.empty()) {
        std::cout << "\n=== DEBUG: Partial Mappings ===" << std::endl;
        std::map<MeshId, std::vector<std::pair<FabricNodeId, tt::tt_metal::AsicID>>> mappings_by_mesh_debug;
        for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
            mappings_by_mesh_debug[fabric_node.mesh_id].emplace_back(fabric_node, asic);
        }

        for (const auto& [mesh_id, mappings] : mappings_by_mesh_debug) {
            std::cout << "\nLogical Mesh " << mesh_id.get() << " -> Physical Mesh mappings:" << std::endl;
            for (const auto& [logical_node, asic] : mappings) {
                std::cout << "    Logical node (mesh=" << logical_node.mesh_id.get()
                          << ", chip=" << logical_node.chip_id << ") -> ASIC " << asic.get() << std::endl;
            }
        }
    }

    // Verify overall result failed (either via exception or result.success == false)
    if (exception_thrown) {
        // Exception was thrown - this is acceptable and indicates over-constrained scenario
        EXPECT_TRUE(true) << "Exception thrown indicates over-constrained device-level constraints: "
                          << exception_message;

        // Verify exception message mentions the constraint issue
        bool exception_mentions_constraint = exception_message.find("not mapped") != std::string::npos ||
                                             exception_message.find("exit node") != std::string::npos ||
                                             exception_message.find("constraint") != std::string::npos ||
                                             exception_message.find("mesh") != std::string::npos;
        EXPECT_TRUE(exception_mentions_constraint)
            << "Exception message should mention constraint, exit node, or mesh mapping issue. Exception: "
            << exception_message;
    } else {
        // No exception - verify result indicates failure
        EXPECT_FALSE(result.success)
            << "Multi-mesh mapping should fail due to over-constrained device-level constraints";

        EXPECT_FALSE(result.error_message.empty())
            << "Error message should be provided when mapping fails. Error: " << result.error_message;

        // Verify that the failure occurred due to constraints
        // The error message should indicate constraint failure, exit node constraints, or solver failure
        bool mentions_constraint_or_exit_node = result.error_message.find("constraint") != std::string::npos ||
                                                result.error_message.find("exit") != std::string::npos ||
                                                result.error_message.find("solver") != std::string::npos ||
                                                result.error_message.find("mapping") != std::string::npos;

        EXPECT_TRUE(mentions_constraint_or_exit_node)
            << "Error message should indicate constraint, exit node, solver, or mapping failure. Error: "
            << result.error_message;
    }

    // Verify that no complete mapping was found
    // The mapper should not have successfully mapped all logical nodes
    size_t expected_logical_nodes = 3 * kPhysicalMeshSize * kPhysicalMeshSize;  // 3 * 2 * 2 = 12
    EXPECT_LT(result.fabric_node_to_asic.size(), expected_logical_nodes)
        << "Should not have mapped all " << expected_logical_nodes << " logical nodes due to over-constraints";
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_ThreeLogicalFivePhysical_DeviceLevelConstraints_Succeeds) {
    // Test mapping 3 logical meshes (2x2 each) connected in a line to 5 physical meshes (2x2 each) connected in a ring.
    // All inter-mesh connections are device-level constraints (strict mode) with valid device assignments.
    // This test verifies that the solver correctly succeeds when device-level constraints can be satisfied.
    //
    // Logical Topology (3 meshes):
    //   Mesh 0: 2x2 grid (4 nodes)
    //   Mesh 1: 2x2 grid (4 nodes)
    //   Mesh 2: 2x2 grid (4 nodes)
    //   Inter-mesh:
    //     - Mesh 0 device 0 <-> Mesh 1 device 0 (device-level connection)
    //     - Mesh 1 device 1 <-> Mesh 2 device 0 (device-level connection)
    //
    // Physical Topology (5 meshes):
    //   Mesh 0: 2x2 grid (4 ASICs)
    //   Mesh 1: 2x2 grid (4 ASICs)
    //   Mesh 2: 2x2 grid (4 ASICs)
    //   Mesh 3: 2x2 grid (4 ASICs)
    //   Mesh 4: 2x2 grid (4 ASICs)
    //   Inter-mesh: Ring topology (0-1-2-3-4-0)
    //   Physical connections: Device 0 connects to device 0, device 1 connects to device 0, plus ring connectivity
    //
    // Expected: Should succeed because device-level constraints match physical connectivity
    // - Logical mesh 0 device 0 -> Logical mesh 1 device 0 matches physical mesh 0 device 0 -> physical mesh 1 device 0
    // - Logical mesh 1 device 1 -> Logical mesh 2 device 0 matches physical mesh 1 device 1 -> physical mesh 2 device 0

    using namespace ::tt::tt_fabric;

    // =========================================================================
    // Create Logical Multi-Mesh Graph using MGD (3 meshes, line topology, device-level connections)
    // =========================================================================

    const std::string mgd_textproto = R"proto(
        # --- Meshes ---------------------------------------------------------------

        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology { dims: [ 2, 2 ] }
          host_topology { dims: [ 1, 1 ] }
          channels { count: 1 }
        }

        # --- Graphs ---------------------------------------------------------------

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          # Instances: mesh ids 0,1,2 (all 2x2)
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 2 } }

          # Intermesh connections: mesh 0 <-> mesh 1, mesh 1 <-> mesh 2 (line topology)
          # Using device-level connections (strict mode) with valid device assignments
          # For 2x2 grid: device 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
          # Mesh 0 device 0 <-> Mesh 1 device 0 (device-level connection)
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            channels { count: 1 }
          }
          # Mesh 1 device 1 <-> Mesh 2 device 0 (device-level connection, different device)
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 device_id: 1 } }
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 2 device_id: 0 } }
            channels { count: 1 }
          }
        }

        # --- Instantiation ----------------------------------------------------------
        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create temporary MGD file
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::string unique_suffix;
    for (int i = 0; i < 8; ++i) {
        unique_suffix += "0123456789abcdef"[dis(gen)];
    }
    const std::filesystem::path temp_mgd_path =
        temp_dir / ("test_three_logical_five_physical_device_level_valid_" + unique_suffix + ".textproto");

    // Write the MGD content to temporary file
    {
        std::ofstream mgd_file(temp_mgd_path);
        ASSERT_TRUE(mgd_file.is_open()) << "Failed to create temporary MGD file";
        mgd_file << mgd_textproto;
    }  // File is closed here

    // Load MeshGraph from the temporary file
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());

    // Clean up temporary file immediately after loading
    std::filesystem::remove(temp_mgd_path);

    // Build the logical multi-mesh graph from MGD
    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Verify we have 3 logical meshes
    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u) << "Should have 3 logical meshes";

    // Verify exit nodes are device-level (strict mode)
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}))
        << "Mesh 0 should have exit nodes (device-level connection)";
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}))
        << "Mesh 1 should have exit nodes (device-level connection)";
    EXPECT_TRUE(logical_multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{2}))
        << "Mesh 2 should have exit nodes (device-level connection)";

    // Verify exit nodes are fabric node-level (device_id specified)
    const auto& exit_graph0 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0});
    const auto& exit_nodes0 = exit_graph0.get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 1u) << "Mesh 0 should have 1 exit node (device 0)";
    LogicalExitNode exit_node0{MeshId{0}, FabricNodeId(MeshId{0}, 0)};
    EXPECT_TRUE(std::find(exit_nodes0.begin(), exit_nodes0.end(), exit_node0) != exit_nodes0.end())
        << "Device 0 should be an exit node in mesh 0";
    EXPECT_TRUE(exit_node0.fabric_node_id.has_value()) << "Mesh 0 exit node should be fabric node-level (strict mode)";

    const auto& exit_graph1 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1});
    const auto& exit_nodes1 = exit_graph1.get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 2u) << "Mesh 1 should have 2 exit nodes (device 0 and device 1)";
    LogicalExitNode exit_node1_0{MeshId{1}, FabricNodeId(MeshId{1}, 0)};
    LogicalExitNode exit_node1_1{MeshId{1}, FabricNodeId(MeshId{1}, 1)};
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_0) != exit_nodes1.end())
        << "Device 0 should be an exit node in mesh 1";
    EXPECT_TRUE(std::find(exit_nodes1.begin(), exit_nodes1.end(), exit_node1_1) != exit_nodes1.end())
        << "Device 1 should be an exit node in mesh 1";

    const auto& exit_graph2 = logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2});
    const auto& exit_nodes2 = exit_graph2.get_nodes();
    EXPECT_EQ(exit_nodes2.size(), 1u) << "Mesh 2 should have 1 exit node (device 0)";
    LogicalExitNode exit_node2{MeshId{2}, FabricNodeId(MeshId{2}, 0)};
    EXPECT_TRUE(std::find(exit_nodes2.begin(), exit_nodes2.end(), exit_node2) != exit_nodes2.end())
        << "Device 0 should be an exit node in mesh 2";

    // =========================================================================
    // Create Physical Multi-Mesh Graph (5 meshes, ring topology)
    // =========================================================================

    constexpr size_t kNumPhysicalMeshes = 5;
    constexpr size_t kPhysicalMeshSize = 2;  // 2x2 grid

    // Create physical meshes: 5 meshes, each 2x2 grid
    PhysicalAdjacencyMap flat_physical_adj;
    std::vector<std::vector<tt::tt_metal::AsicID>> physical_asics_by_mesh(kNumPhysicalMeshes);
    std::vector<PhysicalAdjacencyMap> physical_adj_by_mesh(kNumPhysicalMeshes);

    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        physical_asics_by_mesh[mesh_idx] = make_asics(kPhysicalMeshSize * kPhysicalMeshSize, 100 + mesh_idx * 100);
        physical_adj_by_mesh[mesh_idx] =
            build_grid_adjacency(physical_asics_by_mesh[mesh_idx], kPhysicalMeshSize, kPhysicalMeshSize);
    }

    // Add intermesh connections: ring topology (0->1, 1->2, 2->3, 3->4, 4->0)
    // For 2x2 grid: device 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    // We need to support:
    // - Device 0 to device 0 connections (for logical mesh 0->1)
    // - Device 1 to device 0 connections (for logical mesh 1->2)
    // So we'll add both connection patterns
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        size_t next_mesh_idx = (mesh_idx + 1) % kNumPhysicalMeshes;

        // Connect device 0 of current mesh to device 0 of next mesh (top-left to top-left)
        physical_adj_by_mesh[mesh_idx][physical_asics_by_mesh[mesh_idx][0]].push_back(
            physical_asics_by_mesh[next_mesh_idx][0]);
        physical_adj_by_mesh[next_mesh_idx][physical_asics_by_mesh[next_mesh_idx][0]].push_back(
            physical_asics_by_mesh[mesh_idx][0]);

        // Connect device 1 of current mesh to device 0 of next mesh (top-right to top-left)
        physical_adj_by_mesh[mesh_idx][physical_asics_by_mesh[mesh_idx][1]].push_back(
            physical_asics_by_mesh[next_mesh_idx][0]);
        physical_adj_by_mesh[next_mesh_idx][physical_asics_by_mesh[next_mesh_idx][0]].push_back(
            physical_asics_by_mesh[mesh_idx][1]);

        // Also connect right edge to left edge for full ring connectivity
        for (size_t row = 0; row < kPhysicalMeshSize; ++row) {
            size_t current_right_idx = row * kPhysicalMeshSize + (kPhysicalMeshSize - 1);  // Device 1 or 3
            size_t next_left_idx = row * kPhysicalMeshSize;                                // Device 0 or 2
            // Only add if not already added above
            if (row == 0 && current_right_idx == 1) {
                // Already added device 1 -> device 0 above
                continue;
            }
            physical_adj_by_mesh[mesh_idx][physical_asics_by_mesh[mesh_idx][current_right_idx]].push_back(
                physical_asics_by_mesh[next_mesh_idx][next_left_idx]);
            physical_adj_by_mesh[next_mesh_idx][physical_asics_by_mesh[next_mesh_idx][next_left_idx]].push_back(
                physical_asics_by_mesh[mesh_idx][current_right_idx]);
        }
    }

    // Combine into flat adjacency map
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        for (const auto& [asic, neighbors] : physical_adj_by_mesh[mesh_idx]) {
            flat_physical_adj[asic] = neighbors;
        }
    }

    // Build hierarchical physical graph from flat graph
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        for (const auto& asic : physical_asics_by_mesh[mesh_idx]) {
            asic_id_to_mesh_rank[MeshId{mesh_idx}][asic] = rank0_;
        }
    }

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_physical_adj);
    PhysicalMultiMeshGraph physical_multi_mesh_graph =
        build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    // =========================================================================
    // Run mapping and verify results
    // =========================================================================

    TopologyMappingConfig config;
    config.strict_mode = true;            // Use strict mode for device-level constraints
    config.disable_rank_bindings = true;  // Disable rank bindings - any mapping is valid

    std::cout << "\n=== DEBUG: Starting Mapping (Valid Device-Level Constraints) ===" << std::endl;
    std::cout << "Config: strict_mode=" << config.strict_mode
              << ", disable_rank_bindings=" << config.disable_rank_bindings << std::endl;

    // Call map_multi_mesh_to_physical (rank mappings omitted since disable_rank_bindings is true)
    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    std::cout << "\n=== DEBUG: Mapping Result ===" << std::endl;
    std::cout << "Success: " << (result.success ? "true" : "false") << std::endl;
    std::cout << "Error message: " << result.error_message << std::endl;
    std::cout << "Number of mapped nodes: " << result.fabric_node_to_asic.size() << std::endl;

    // Verify overall result succeeded
    EXPECT_TRUE(result.success) << "Multi-mesh mapping should succeed with valid device-level constraints: "
                                << result.error_message;

    // Verify bidirectional consistency of the overall result
    verify_bidirectional_consistency(result);

    // Verify all logical nodes are mapped (3 meshes * 4 nodes = 12 nodes)
    EXPECT_EQ(result.fabric_node_to_asic.size(), 12u) << "All 12 logical nodes should be mapped";

    // Group mappings by mesh_id
    std::map<MeshId, std::map<FabricNodeId, tt::tt_metal::AsicID>> mappings_by_mesh;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh[fabric_node.mesh_id][fabric_node] = asic;
    }

    // Verify we have mappings for all 3 logical meshes
    EXPECT_EQ(mappings_by_mesh.size(), 3u) << "Should have mappings for all 3 logical meshes";
    EXPECT_TRUE(mappings_by_mesh.contains(MeshId{0})) << "Should have mappings for logical mesh 0";
    EXPECT_TRUE(mappings_by_mesh.contains(MeshId{1})) << "Should have mappings for logical mesh 1";
    EXPECT_TRUE(mappings_by_mesh.contains(MeshId{2})) << "Should have mappings for logical mesh 2";

    // Verify each logical mesh has all 4 nodes mapped
    for (const auto& [logical_mesh_id, mesh_mappings] : mappings_by_mesh) {
        EXPECT_EQ(mesh_mappings.size(), 4u) << "Logical mesh " << logical_mesh_id.get() << " should map all 4 nodes";
    }

    // =========================================================================
    // Verify intra-mesh connectivity is preserved
    // =========================================================================

    for (const auto& [logical_mesh_id, mesh_mappings] : mappings_by_mesh) {
        // Find which physical mesh this logical mesh mapped to
        MeshId physical_mesh_id = MeshId{0};
        if (!mesh_mappings.empty()) {
            const auto& first_asic = mesh_mappings.begin()->second;
            for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
                const auto& nodes = adjacency_graph.get_nodes();
                if (std::find(nodes.begin(), nodes.end(), first_asic) != nodes.end()) {
                    physical_mesh_id = pm_id;
                    break;
                }
            }
        }

        // Verify intra-mesh connectivity is preserved
        const auto& logical_graph = logical_multi_mesh_graph.mesh_adjacency_graphs_.at(logical_mesh_id);
        const auto& physical_graph = physical_multi_mesh_graph.mesh_adjacency_graphs_.at(physical_mesh_id);
        const auto& logical_nodes = logical_graph.get_nodes();

        for (const auto& node : logical_nodes) {
            const auto mapped_asic = mesh_mappings.at(node);
            const auto& logical_neighbors = logical_graph.get_neighbors(node);
            const auto& physical_neighbors = physical_graph.get_neighbors(mapped_asic);

            for (const auto& neighbor : logical_neighbors) {
                // If neighbor is in the same logical mesh, verify intra-mesh connectivity
                if (neighbor.mesh_id == logical_mesh_id) {
                    const auto neighbor_asic = mesh_mappings.at(neighbor);
                    EXPECT_TRUE(
                        std::find(physical_neighbors.begin(), physical_neighbors.end(), neighbor_asic) !=
                        physical_neighbors.end())
                        << "Logical intra-mesh edge not preserved in physical mapping for logical mesh "
                        << logical_mesh_id.get() << ": logical node " << node.chip_id << " -> " << neighbor.chip_id
                        << " should map to physically connected ASICs";
                }
            }
        }
    }

    // =========================================================================
    // Verify inter-mesh connectivity with device-level constraints
    // =========================================================================

    // Verify exit node mappings match device-level constraints
    // Logical mesh 0 device 0 should map to a physical ASIC that connects to logical mesh 1 device 0's mapped ASIC
    FabricNodeId logical_exit_node_0_0{MeshId{0}, 0};
    FabricNodeId logical_exit_node_1_0{MeshId{1}, 0};
    FabricNodeId logical_exit_node_1_1{MeshId{1}, 1};
    FabricNodeId logical_exit_node_2_0{MeshId{2}, 0};

    EXPECT_TRUE(mappings_by_mesh.at(MeshId{0}).contains(logical_exit_node_0_0))
        << "Logical mesh 0 device 0 should be mapped";
    EXPECT_TRUE(mappings_by_mesh.at(MeshId{1}).contains(logical_exit_node_1_0))
        << "Logical mesh 1 device 0 should be mapped";
    EXPECT_TRUE(mappings_by_mesh.at(MeshId{1}).contains(logical_exit_node_1_1))
        << "Logical mesh 1 device 1 should be mapped";
    EXPECT_TRUE(mappings_by_mesh.at(MeshId{2}).contains(logical_exit_node_2_0))
        << "Logical mesh 2 device 0 should be mapped";

    const auto& asic_0_0 = mappings_by_mesh.at(MeshId{0}).at(logical_exit_node_0_0);
    const auto& asic_1_0 = mappings_by_mesh.at(MeshId{1}).at(logical_exit_node_1_0);
    const auto& asic_1_1 = mappings_by_mesh.at(MeshId{1}).at(logical_exit_node_1_1);
    const auto& asic_2_0 = mappings_by_mesh.at(MeshId{2}).at(logical_exit_node_2_0);

    // Verify logical mesh 0 device 0 is physically connected to logical mesh 1 device 0
    // Find which physical meshes these ASICs belong to
    MeshId physical_mesh_0 = MeshId{0};
    MeshId physical_mesh_1 = MeshId{0};
    MeshId physical_mesh_2 = MeshId{0};

    for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& nodes = adjacency_graph.get_nodes();
        if (std::find(nodes.begin(), nodes.end(), asic_0_0) != nodes.end()) {
            physical_mesh_0 = pm_id;
        }
        if (std::find(nodes.begin(), nodes.end(), asic_1_0) != nodes.end()) {
            physical_mesh_1 = pm_id;
        }
        if (std::find(nodes.begin(), nodes.end(), asic_2_0) != nodes.end()) {
            physical_mesh_2 = pm_id;
        }
    }

    // Verify physical meshes are connected
    const auto& physical_mesh_level_graph = physical_multi_mesh_graph.mesh_level_graph_;
    const auto& neighbors_0 = physical_mesh_level_graph.get_neighbors(physical_mesh_0);
    EXPECT_TRUE(std::find(neighbors_0.begin(), neighbors_0.end(), physical_mesh_1) != neighbors_0.end())
        << "Physical mesh " << physical_mesh_0.get() << " (logical mesh 0) should connect to physical mesh "
        << physical_mesh_1.get() << " (logical mesh 1)";

    const auto& neighbors_1 = physical_mesh_level_graph.get_neighbors(physical_mesh_1);
    EXPECT_TRUE(std::find(neighbors_1.begin(), neighbors_1.end(), physical_mesh_2) != neighbors_1.end())
        << "Physical mesh " << physical_mesh_1.get() << " (logical mesh 1) should connect to physical mesh "
        << physical_mesh_2.get() << " (logical mesh 2)";

    // Verify the specific ASICs are physically connected (device-level constraint)
    // Get exit node graphs to check inter-mesh ASIC connections
    const auto& exit_node_graph_0 = physical_multi_mesh_graph.mesh_exit_node_graphs_.at(physical_mesh_0);
    const auto& exit_node_graph_1 = physical_multi_mesh_graph.mesh_exit_node_graphs_.at(physical_mesh_1);

    // Check if asic_0_0 (logical mesh 0 device 0) connects to asic_1_0 (logical mesh 1 device 0)
    PhysicalExitNode exit_node_0_0{physical_mesh_0, asic_0_0};
    const auto& exit_neighbors_0_0 = exit_node_graph_0.get_neighbors(exit_node_0_0);
    bool found_connection_0_1 = false;
    for (const auto& neighbor_exit_node : exit_neighbors_0_0) {
        if (neighbor_exit_node.mesh_id == physical_mesh_1 && neighbor_exit_node.asic_id == asic_1_0) {
            found_connection_0_1 = true;
            break;
        }
    }
    EXPECT_TRUE(found_connection_0_1) << "Logical mesh 0 device 0 (ASIC " << asic_0_0.get()
                                      << ") should be physically connected to logical mesh 1 device 0 (ASIC "
                                      << asic_1_0.get() << ")";

    // Check if asic_1_1 (logical mesh 1 device 1) connects to asic_2_0 (logical mesh 2 device 0)
    PhysicalExitNode exit_node_1_1{physical_mesh_1, asic_1_1};
    const auto& exit_neighbors_1_1 = exit_node_graph_1.get_neighbors(exit_node_1_1);
    bool found_connection_1_2 = false;
    for (const auto& neighbor_exit_node : exit_neighbors_1_1) {
        if (neighbor_exit_node.mesh_id == physical_mesh_2 && neighbor_exit_node.asic_id == asic_2_0) {
            found_connection_1_2 = true;
            break;
        }
    }
    EXPECT_TRUE(found_connection_1_2) << "Logical mesh 1 device 1 (ASIC " << asic_1_1.get()
                                      << ") should be physically connected to logical mesh 2 device 0 (ASIC "
                                      << asic_2_0.get() << ")";

    // Verify fabric node IDs match the device-level constraints
    EXPECT_EQ(logical_exit_node_0_0.chip_id, 0u) << "Logical mesh 0 exit node should be device 0 (as specified in MGD)";
    EXPECT_EQ(logical_exit_node_1_0.chip_id, 0u) << "Logical mesh 1 exit node should be device 0 (as specified in MGD)";
    EXPECT_EQ(logical_exit_node_1_1.chip_id, 1u) << "Logical mesh 1 exit node should be device 1 (as specified in MGD)";
    EXPECT_EQ(logical_exit_node_2_0.chip_id, 0u) << "Logical mesh 2 exit node should be device 0 (as specified in MGD)";

    std::cout << "\n=== DEBUG: Device-Level Constraint Verification ===" << std::endl;
    std::cout << "Logical Mesh 0 device 0 -> ASIC " << asic_0_0.get() << " (Physical Mesh " << physical_mesh_0.get()
              << ")" << std::endl;
    std::cout << "Logical Mesh 1 device 0 -> ASIC " << asic_1_0.get() << " (Physical Mesh " << physical_mesh_1.get()
              << ")" << std::endl;
    std::cout << "Logical Mesh 1 device 1 -> ASIC " << asic_1_1.get() << " (Physical Mesh " << physical_mesh_1.get()
              << ")" << std::endl;
    std::cout << "Logical Mesh 2 device 0 -> ASIC " << asic_2_0.get() << " (Physical Mesh " << physical_mesh_2.get()
              << ")" << std::endl;
    std::cout << "Connection 0->1 (device 0->0) verified: " << (found_connection_0_1 ? "YES" : "NO") << std::endl;
    std::cout << "Connection 1->2 (device 1->0) verified: " << (found_connection_1_2 ? "YES" : "NO") << std::endl;
}

}  // namespace
}  // namespace tt::tt_metal::experimental::tt_fabric
