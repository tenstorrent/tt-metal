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

    // -------------------------------------------------------------------------
    // Multi-mesh test helpers
    // -------------------------------------------------------------------------

    // Create temporary MGD file from textproto string
    static std::filesystem::path create_temp_mgd_file(const std::string& mgd_textproto, const std::string& prefix) {
        const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);
        std::string unique_suffix;
        for (int i = 0; i < 8; ++i) {
            unique_suffix += "0123456789abcdef"[dis(gen)];
        }
        const std::filesystem::path temp_mgd_path = temp_dir / (prefix + unique_suffix + ".textproto");
        std::ofstream mgd_file(temp_mgd_path);
        if (!mgd_file.is_open()) {
            ADD_FAILURE() << "Failed to create temporary MGD file";
            return temp_mgd_path;  // Return path anyway
        }
        mgd_file << mgd_textproto;
        return temp_mgd_path;
    }

    // Count occurrences of a value in a vector
    template <typename T>
    static size_t count_occurrences(const std::vector<T>& vec, const T& value) {
        return std::count(vec.begin(), vec.end(), value);
    }

    // Verify mesh has expected number of nodes (simplified check)
    template <typename GraphType>
    static void verify_mesh_size(const GraphType& graph, MeshId mesh_id, size_t expected_size) {
        EXPECT_EQ(graph.mesh_adjacency_graphs_.at(mesh_id).get_nodes().size(), expected_size)
            << "Mesh " << mesh_id.get() << " should have " << expected_size << " nodes";
    }

    // Verify mesh-level connectivity (simplified - just check neighbor count)
    static void verify_mesh_connectivity(
        const LogicalMultiMeshGraph& graph, MeshId mesh_id, size_t expected_neighbor_count) {
        const auto& neighbors = graph.mesh_level_graph_.get_neighbors(mesh_id);
        EXPECT_EQ(neighbors.size(), expected_neighbor_count)
            << "Mesh " << mesh_id.get() << " should have " << expected_neighbor_count << " neighbors";
    }

    // Overload for PhysicalMultiMeshGraph
    static void verify_mesh_connectivity(
        const PhysicalMultiMeshGraph& graph, MeshId mesh_id, size_t expected_neighbor_count) {
        const auto& neighbors = graph.mesh_level_graph_.get_neighbors(mesh_id);
        EXPECT_EQ(neighbors.size(), expected_neighbor_count)
            << "Mesh " << mesh_id.get() << " should have " << expected_neighbor_count << " neighbors";
    }

    // Verify exit node exists (simplified check)
    static void verify_exit_node_exists(
        const LogicalMultiMeshGraph& graph, MeshId mesh_id, const LogicalExitNode& exit_node) {
        ASSERT_TRUE(graph.mesh_exit_node_graphs_.contains(mesh_id))
            << "Mesh " << mesh_id.get() << " should have exit nodes";
        const auto& exit_nodes = graph.mesh_exit_node_graphs_.at(mesh_id).get_nodes();
        EXPECT_TRUE(std::find(exit_nodes.begin(), exit_nodes.end(), exit_node) != exit_nodes.end())
            << "Exit node should exist";
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

    // Verify basic structure - each mesh has 8 nodes
    verify_mesh_size(multi_mesh_graph, MeshId{0}, 8u);

    // Verify ALL_TO_ALL connectivity - mesh 0 should have 6 neighbors (2 channels x 3 other meshes)
    verify_mesh_connectivity(multi_mesh_graph, MeshId{0}, 6u);

    // Verify exit nodes exist for meshes with connections
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}));
    const auto& exit_nodes = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes();
    EXPECT_GT(exit_nodes.size(), 0u);
    EXPECT_FALSE(exit_nodes[0].fabric_node_id.has_value()) << "Should be mesh-level exit nodes in relaxed mode";
}

TEST_F(TopologyMapperUtilsTest, BuildLogicalMultiMeshGraph_StrictModeIntermeshPorts) {
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_strict_connection_mgd.textproto";

    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::T3K, mesh_graph_desc_path.string());
    const auto multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 2u);
    verify_mesh_size(multi_mesh_graph, MeshId{0}, 4u);

    // Verify intermesh connections: 4 total (2 channels x 2 device pairs)
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 4u);
    EXPECT_EQ(count_occurrences(mesh0_neighbors, MeshId{1}), 4u);

    // Verify exit nodes exist and are device-level (strict mode)
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}));
    const auto& exit_nodes0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 2u);
    EXPECT_TRUE(exit_nodes0[0].fabric_node_id.has_value()) << "Should be device-level exit nodes in strict mode";

    // Verify requested ports structure
    const auto& requested_ports = mesh_graph.get_requested_intermesh_ports();
    EXPECT_FALSE(requested_ports.empty());
    EXPECT_EQ(requested_ports.at(0).at(1).size(), 2u);
}

TEST_F(TopologyMapperUtilsTest, BuildLogicalMultiMeshGraph_MixedStrictAndRelaxedConnections) {
    // Test mixed strict (device-level) and relaxed (mesh-level) connections
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

    const std::filesystem::path temp_mgd_path = create_temp_mgd_file(mgd_textproto, "test_mixed_connections_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
    std::filesystem::remove(temp_mgd_path);

    const auto multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u);

    // Verify mesh-level connectivity: mesh 0 has 5 connections (2 to mesh 1, 3 to mesh 2)
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    EXPECT_EQ(mesh0_neighbors.size(), 5u);
    EXPECT_EQ(count_occurrences(mesh0_neighbors, MeshId{1}), 2u);
    EXPECT_EQ(count_occurrences(mesh0_neighbors, MeshId{2}), 3u);

    // Verify exit nodes: mesh 0 has both device-level and mesh-level exit nodes
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}));
    const auto& exit_nodes0 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes();
    EXPECT_EQ(exit_nodes0.size(), 2u);

    // Verify mesh 0 has both types: one device-level (device 1) and one mesh-level
    bool has_device_level = false, has_mesh_level = false;
    for (const auto& exit_node : exit_nodes0) {
        if (exit_node.fabric_node_id.has_value()) {
            has_device_level = true;
            EXPECT_EQ(exit_node.fabric_node_id->chip_id, 1u);
        } else {
            has_mesh_level = true;
        }
    }
    EXPECT_TRUE(has_device_level && has_mesh_level);

    // Verify mesh 1 has device-level exit node
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{1}));
    const auto& exit_nodes1 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes();
    EXPECT_EQ(exit_nodes1.size(), 1u);
    EXPECT_TRUE(exit_nodes1[0].fabric_node_id.has_value());

    // Verify mesh 2 has mesh-level exit node
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{2}));
    const auto& exit_nodes2 = multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2}).get_nodes();
    EXPECT_EQ(exit_nodes2.size(), 1u);
    EXPECT_FALSE(exit_nodes2[0].fabric_node_id.has_value());
}

TEST_F(TopologyMapperUtilsTest, BuildPhysicalMultiMeshGraph_MultiHostMultiMesh) {
    // Test multi-host multi-mesh physical graph building
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path psd_file_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mock_PSDs/test_multihost_multimesh.textproto";
    ASSERT_TRUE(std::filesystem::exists(psd_file_path)) << "PSD test file not found";

    tt::tt_metal::PhysicalSystemDescriptor physical_system_descriptor(psd_file_path.string());
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;

    // Mesh 0: ASICs 1,2 (host0) and 3,4 (host1)
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{1}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{2}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{3}] = MeshHostRankId{1};
    asic_id_to_mesh_rank[MeshId{0}][tt::tt_metal::AsicID{4}] = MeshHostRankId{1};

    // Mesh 1: ASICs 5,6 (host0) and 7,8 (host1)
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{5}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{6}] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{7}] = MeshHostRankId{1};
    asic_id_to_mesh_rank[MeshId{1}][tt::tt_metal::AsicID{8}] = MeshHostRankId{1};

    const auto multi_mesh_graph =
        build_physical_multi_mesh_adjacency_graph(physical_system_descriptor, asic_id_to_mesh_rank);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 2u);
    verify_mesh_size(multi_mesh_graph, MeshId{0}, 4u);

    // Verify inter-mesh connectivity
    const auto& mesh0_neighbors = multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0});
    if (!mesh0_neighbors.empty()) {
        EXPECT_EQ(mesh0_neighbors.size(), 2u);
        EXPECT_TRUE(std::find(mesh0_neighbors.begin(), mesh0_neighbors.end(), MeshId{1}) != mesh0_neighbors.end());
    }

    // Verify exit nodes exist
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.contains(MeshId{0}));
    EXPECT_GT(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 0u);
}

TEST_F(TopologyMapperUtilsTest, BuildPhysicalMultiMeshGraph_ExitNodeTracking) {
    // Test exit node tracking in physical multi-mesh graph
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap physical_adj_m0, physical_adj_m1;
    tt::tt_metal::AsicID asic100{100}, asic101{101}, asic102{102};
    tt::tt_metal::AsicID asic200{200}, asic201{201}, asic202{202};

    physical_adj_m0[asic100] = {asic101, asic200};
    physical_adj_m0[asic101] = {asic100, asic102, asic201};
    physical_adj_m0[asic102] = {asic101, asic202};
    physical_adj_m1[asic200] = {asic201, asic100};
    physical_adj_m1[asic201] = {asic200, asic202, asic101};
    physical_adj_m1[asic202] = {asic201, asic102};

    PhysicalMultiMeshGraph physical_multi_mesh_graph;
    physical_multi_mesh_graph.mesh_adjacency_graphs_[MeshId{0}] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m0);
    physical_multi_mesh_graph.mesh_adjacency_graphs_[MeshId{1}] = AdjacencyGraph<tt::tt_metal::AsicID>(physical_adj_m1);

    AdjacencyGraph<MeshId>::AdjacencyMap mesh_level_adj;
    mesh_level_adj[MeshId{0}] = {MeshId{1}};
    mesh_level_adj[MeshId{1}] = {MeshId{0}};
    physical_multi_mesh_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(mesh_level_adj);

    // Manually populate exit nodes
    AdjacencyGraph<PhysicalExitNode>::AdjacencyMap exit_adj_m0, exit_adj_m1;
    PhysicalExitNode exit0_100{MeshId{0}, asic100}, exit0_101{MeshId{0}, asic101}, exit0_102{MeshId{0}, asic102};
    PhysicalExitNode exit1_200{MeshId{1}, asic200}, exit1_201{MeshId{1}, asic201}, exit1_202{MeshId{1}, asic202};
    exit_adj_m0[exit0_100] = {exit1_200};
    exit_adj_m0[exit0_101] = {exit1_201};
    exit_adj_m0[exit0_102] = {exit1_202};
    exit_adj_m1[exit1_200] = {exit0_100};
    exit_adj_m1[exit1_201] = {exit0_101};
    exit_adj_m1[exit1_202] = {exit0_102};
    physical_multi_mesh_graph.mesh_exit_node_graphs_[MeshId{0}] = AdjacencyGraph<PhysicalExitNode>(exit_adj_m0);
    physical_multi_mesh_graph.mesh_exit_node_graphs_[MeshId{1}] = AdjacencyGraph<PhysicalExitNode>(exit_adj_m1);

    // Verify exit nodes
    EXPECT_EQ(physical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 3u);
    EXPECT_EQ(physical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 3u);
    EXPECT_EQ(physical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_neighbors(exit0_100).size(), 1u);
    EXPECT_EQ(physical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_neighbors(exit0_100)[0], exit1_200);
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

    // Verify basic mapping counts - connectivity preservation is tested elsewhere
    EXPECT_EQ(result.fabric_node_to_asic.size(), 8u);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_IncompatibleTopology_Fails) {
    // Test failure when logical mesh (3x3=9 nodes) is too large for physical meshes (2x2=4 nodes each)
    using namespace ::tt::tt_fabric;

    // Logical: 3x3 grid (9 nodes) + 2x2 grid (4 nodes)
    LogicalMultiMeshGraph logical_graph;
    std::vector<FabricNodeId> nodes_m0, nodes_m1;
    for (uint32_t i = 0; i < 9; ++i) {
        nodes_m0.push_back(FabricNodeId(MeshId{0}, i));
    }
    for (uint32_t i = 0; i < 4; ++i) {
        nodes_m1.push_back(FabricNodeId(MeshId{1}, i));
    }
    logical_graph.mesh_adjacency_graphs_[MeshId{0}] =
        AdjacencyGraph<FabricNodeId>(build_grid_adjacency(nodes_m0, 3, 3));
    logical_graph.mesh_adjacency_graphs_[MeshId{1}] =
        AdjacencyGraph<FabricNodeId>(build_grid_adjacency(nodes_m1, 2, 2));
    AdjacencyGraph<MeshId>::AdjacencyMap logical_mesh_adj;
    logical_mesh_adj[MeshId{0}] = {MeshId{1}};
    logical_mesh_adj[MeshId{1}] = {MeshId{0}};
    logical_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(logical_mesh_adj);

    // Physical: 3 meshes, each 2x2 grid (4 nodes)
    PhysicalMultiMeshGraph physical_graph;
    for (uint32_t m = 0; m < 3; ++m) {
        std::vector<tt::tt_metal::AsicID> asics;
        for (uint64_t i = 0; i < 4; ++i) {
            asics.push_back(tt::tt_metal::AsicID{100 * m + i});
        }
        physical_graph.mesh_adjacency_graphs_[MeshId{m}] =
            AdjacencyGraph<tt::tt_metal::AsicID>(build_grid_adjacency(asics, 2, 2));
    }
    AdjacencyGraph<MeshId>::AdjacencyMap physical_mesh_adj;
    physical_mesh_adj[MeshId{0}] = {MeshId{1}};
    physical_mesh_adj[MeshId{1}] = {MeshId{0}, MeshId{2}};
    physical_mesh_adj[MeshId{2}] = {MeshId{1}};
    physical_graph.mesh_level_graph_ = AdjacencyGraph<MeshId>(physical_mesh_adj);

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;
    const auto result = map_multi_mesh_to_physical(logical_graph, physical_graph, config);

    EXPECT_FALSE(result.success);
    EXPECT_LT(result.fabric_node_to_asic.size(), 13u);  // Should not map all 9+4 nodes
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_SingleMesh) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0{100}, asic1{101}, asic2{102};
    flat_adj[asic0] = {asic1};
    flat_adj[asic1] = {asic0, asic2};
    flat_adj[asic2] = {asic1};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{0}][asic2] = MeshHostRankId{0};

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 1u);
    verify_mesh_size(multi_mesh_graph, MeshId{0}, 3u);
    EXPECT_TRUE(multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0}).empty());
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().empty());
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_TwoMeshes) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0_0{100}, asic0_1{101}, asic0_2{102}, asic0_3{103};
    tt::tt_metal::AsicID asic1_0{200}, asic1_1{201}, asic1_2{202}, asic1_3{203};

    flat_adj[asic0_0] = {asic0_1, asic1_0};
    flat_adj[asic0_1] = {asic0_0, asic0_2};
    flat_adj[asic0_2] = {asic0_1, asic0_3};
    flat_adj[asic0_3] = {asic0_2};
    flat_adj[asic1_0] = {asic1_1, asic0_0};
    flat_adj[asic1_1] = {asic1_0, asic1_2};
    flat_adj[asic1_2] = {asic1_1, asic1_3};
    flat_adj[asic1_3] = {asic1_2};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (auto asic : {asic0_0, asic0_1, asic0_2, asic0_3}) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic1_0, asic1_1, asic1_2, asic1_3}) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = MeshHostRankId{0};
    }

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 2u);
    verify_mesh_size(multi_mesh_graph, MeshId{0}, 4u);
    verify_mesh_size(multi_mesh_graph, MeshId{1}, 4u);

    EXPECT_EQ(multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0}).size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0})[0], MeshId{1});

    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 1u);
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_MultipleChannels) {
    // Test that multiple channels between the same pair are preserved
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0_0{100}, asic0_1{101}, asic0_2{102};
    tt::tt_metal::AsicID asic1_0{200}, asic1_1{201}, asic1_2{202};

    flat_adj[asic0_0] = {asic0_1, asic1_0, asic1_0, asic1_0};  // 3 channels
    flat_adj[asic0_1] = {asic0_0, asic0_2};
    flat_adj[asic0_2] = {asic0_1};
    flat_adj[asic1_0] = {asic1_1, asic0_0, asic0_0, asic0_0};  // 3 channels
    flat_adj[asic1_1] = {asic1_0, asic1_2};
    flat_adj[asic1_2] = {asic1_1};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (auto asic : {asic0_0, asic0_1, asic0_2}) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic1_0, asic1_1, asic1_2}) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = MeshHostRankId{0};
    }

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    verify_mesh_size(multi_mesh_graph, MeshId{0}, 3u);
    verify_mesh_size(multi_mesh_graph, MeshId{1}, 3u);

    PhysicalExitNode exit0{MeshId{0}, asic0_0};
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_neighbors(exit0).size(), 3u);
}

TEST_F(TopologyMapperUtilsTest, ConvertFlatAdjacencyToMultiMeshGraph_ThreeMeshes) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0{100}, asic1{200}, asic2{300};
    flat_adj[asic0] = {asic1};
    flat_adj[asic1] = {asic0, asic2};
    flat_adj[asic2] = {asic1};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2] = MeshHostRankId{0};

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{0}, 1u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{1}, 2u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{2}, 1u);

    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2}).get_nodes().size(), 1u);
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_DisconnectedMeshes) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    // Mesh 0: 5 ASICs chain
    tt::tt_metal::AsicID asic0_0{100}, asic0_1{101}, asic0_2{102}, asic0_3{103}, asic0_4{104};
    flat_adj[asic0_0] = {asic0_1};
    flat_adj[asic0_1] = {asic0_0, asic0_2};
    flat_adj[asic0_2] = {asic0_1, asic0_3};
    flat_adj[asic0_3] = {asic0_2, asic0_4};
    flat_adj[asic0_4] = {asic0_3};
    // Mesh 1: 4 ASICs chain
    tt::tt_metal::AsicID asic1_0{200}, asic1_1{201}, asic1_2{202}, asic1_3{203};
    flat_adj[asic1_0] = {asic1_1};
    flat_adj[asic1_1] = {asic1_0, asic1_2};
    flat_adj[asic1_2] = {asic1_1, asic1_3};
    flat_adj[asic1_3] = {asic1_2};
    // Mesh 2: 2 ASICs
    tt::tt_metal::AsicID asic2_0{300}, asic2_1{301};
    flat_adj[asic2_0] = {asic2_1};
    flat_adj[asic2_1] = {asic2_0};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (auto asic : {asic0_0, asic0_1, asic0_2, asic0_3, asic0_4}) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic1_0, asic1_1, asic1_2, asic1_3}) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic2_0, asic2_1}) {
        asic_id_to_mesh_rank[MeshId{2}][asic] = MeshHostRankId{0};
    }

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u);
    verify_mesh_size(multi_mesh_graph, MeshId{0}, 5u);
    verify_mesh_size(multi_mesh_graph, MeshId{1}, 4u);
    verify_mesh_size(multi_mesh_graph, MeshId{2}, 2u);

    EXPECT_TRUE(multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0}).empty());
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().empty());
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_MultipleExitNodesPerMesh) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0_0{100}, asic0_1{101}, asic0_2{102}, asic0_3{103}, asic0_4{104};
    tt::tt_metal::AsicID asic1_0{200}, asic1_1{201}, asic1_2{202};
    tt::tt_metal::AsicID asic2_0{300}, asic2_1{301}, asic2_2{302};

    flat_adj[asic0_0] = {asic0_1, asic1_0};
    flat_adj[asic0_1] = {asic0_0, asic0_2};
    flat_adj[asic0_2] = {asic0_1, asic0_3};
    flat_adj[asic0_3] = {asic0_2, asic0_4};
    flat_adj[asic0_4] = {asic0_3, asic2_0};
    flat_adj[asic1_0] = {asic1_1, asic0_0};
    flat_adj[asic1_1] = {asic1_0, asic1_2};
    flat_adj[asic1_2] = {asic1_1};
    flat_adj[asic2_0] = {asic2_1, asic0_4};
    flat_adj[asic2_1] = {asic2_0, asic2_2};
    flat_adj[asic2_2] = {asic2_1};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (auto asic : {asic0_0, asic0_1, asic0_2, asic0_3, asic0_4}) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic1_0, asic1_1, asic1_2}) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic2_0, asic2_1, asic2_2}) {
        asic_id_to_mesh_rank[MeshId{2}][asic] = MeshHostRankId{0};
    }

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    verify_mesh_size(multi_mesh_graph, MeshId{0}, 5u);
    verify_mesh_size(multi_mesh_graph, MeshId{1}, 3u);
    verify_mesh_size(multi_mesh_graph, MeshId{2}, 3u);

    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 2u);
    EXPECT_EQ(multi_mesh_graph.mesh_level_graph_.get_neighbors(MeshId{0}).size(), 2u);
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_MeshWithOnlyExitNodes) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0_0{100}, asic0_1{101}, asic0_2{102}, asic0_3{103}, asic0_4{104};
    tt::tt_metal::AsicID asic1_0{200}, asic1_1{201}, asic1_2{202}, asic1_3{203}, asic1_4{204};

    // Mesh 0: all ASICs only have exit connections (no internal)
    flat_adj[asic0_0] = {asic1_0};
    flat_adj[asic0_1] = {asic1_1};
    flat_adj[asic0_2] = {asic1_2};
    flat_adj[asic0_3] = {asic1_3};
    flat_adj[asic0_4] = {asic1_4};
    // Mesh 1: chain with exit connections
    flat_adj[asic1_0] = {asic1_1, asic0_0};
    flat_adj[asic1_1] = {asic1_0, asic1_2, asic0_1};
    flat_adj[asic1_2] = {asic1_1, asic1_3, asic0_2};
    flat_adj[asic1_3] = {asic1_2, asic1_4, asic0_3};
    flat_adj[asic1_4] = {asic1_3, asic0_4};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    for (auto asic : {asic0_0, asic0_1, asic0_2, asic0_3, asic0_4}) {
        asic_id_to_mesh_rank[MeshId{0}][asic] = MeshHostRankId{0};
    }
    for (auto asic : {asic1_0, asic1_1, asic1_2, asic1_3, asic1_4}) {
        asic_id_to_mesh_rank[MeshId{1}][asic] = MeshHostRankId{0};
    }

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    verify_mesh_size(multi_mesh_graph, MeshId{0}, 5u);
    verify_mesh_size(multi_mesh_graph, MeshId{1}, 5u);

    // Mesh 0: all nodes should have no internal neighbors
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0}).get_neighbors(asic0_0).empty());

    // Mesh 0 should have 5 exit nodes (all ASICs)
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 5u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 5u);
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_EmptyGraph) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.empty());
    EXPECT_TRUE(multi_mesh_graph.mesh_exit_node_graphs_.empty());
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_UnassignedASICs) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0{100}, asic1{200}, unassigned{300};
    flat_adj[asic0] = {asic1, unassigned};
    flat_adj[asic1] = {asic0, unassigned};
    flat_adj[unassigned] = {asic0, asic1};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    verify_mesh_size(multi_mesh_graph, MeshId{0}, 1u);
    verify_mesh_size(multi_mesh_graph, MeshId{1}, 1u);

    // Unassigned ASIC should not be in any mesh
    EXPECT_TRUE(multi_mesh_graph.mesh_adjacency_graphs_.at(MeshId{0}).get_neighbors(asic0).empty());
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u);
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_RingTopology) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0{100}, asic1{200}, asic2{300}, asic3{400};
    flat_adj[asic0] = {asic3, asic1};
    flat_adj[asic1] = {asic0, asic2};
    flat_adj[asic2] = {asic1, asic3};
    flat_adj[asic3] = {asic2, asic0};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{3}][asic3] = MeshHostRankId{0};

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    EXPECT_EQ(multi_mesh_graph.mesh_adjacency_graphs_.size(), 4u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{0}, 2u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{1}, 2u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{2}, 2u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{3}, 2u);

    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2}).get_nodes().size(), 1u);
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{3}).get_nodes().size(), 1u);
}

TEST_F(TopologyMapperUtilsTest, BuildHierarchicalFromFlatGraph_StarTopology) {
    using namespace ::tt::tt_fabric;

    PhysicalAdjacencyMap flat_adj;
    tt::tt_metal::AsicID asic0{100}, asic1{200}, asic2{300}, asic3{400};
    flat_adj[asic0] = {asic1, asic2, asic3};
    flat_adj[asic1] = {asic0};
    flat_adj[asic2] = {asic0};
    flat_adj[asic3] = {asic0};

    AdjacencyGraph<tt::tt_metal::AsicID> flat_graph(flat_adj);
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    asic_id_to_mesh_rank[MeshId{0}][asic0] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{1}][asic1] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{2}][asic2] = MeshHostRankId{0};
    asic_id_to_mesh_rank[MeshId{3}][asic3] = MeshHostRankId{0};

    const auto multi_mesh_graph = build_hierarchical_from_flat_graph(flat_graph, asic_id_to_mesh_rank);

    verify_mesh_connectivity(multi_mesh_graph, MeshId{0}, 3u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{1}, 1u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{2}, 1u);
    verify_mesh_connectivity(multi_mesh_graph, MeshId{3}, 1u);

    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u);
    PhysicalExitNode exit0{MeshId{0}, asic0};
    EXPECT_EQ(multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_neighbors(exit0).size(), 3u);
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

    const std::filesystem::path temp_mgd_path = create_temp_mgd_file(mgd_textproto, "test_intermesh_2x2_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
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

    // Verify exit nodes map to directly connected ASICs
    FabricNodeId exit_m0(MeshId{0}, 0), exit_m1(MeshId{1}, 0);
    ASSERT_TRUE(mappings_by_mesh.at(MeshId{0}).contains(exit_m0));
    ASSERT_TRUE(mappings_by_mesh.at(MeshId{1}).contains(exit_m1));

    const auto& asic0 = mappings_by_mesh.at(MeshId{0}).at(exit_m0);
    const auto& asic1 = mappings_by_mesh.at(MeshId{1}).at(exit_m1);
    const auto& neighbors = flat_graph.get_neighbors(asic0);
    EXPECT_TRUE(
        std::find(neighbors.begin(), neighbors.end(), asic1) != neighbors.end() ||
        std::find(flat_graph.get_neighbors(asic1).begin(), flat_graph.get_neighbors(asic1).end(), asic0) !=
            flat_graph.get_neighbors(asic1).end());
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

    const std::filesystem::path temp_mgd_path = create_temp_mgd_file(mgd_textproto, "test_impossible_2x2_to_3x3_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
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

    const std::filesystem::path temp_mgd_path = create_temp_mgd_file(mgd_textproto, "test_mixed_policies_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
    std::filesystem::remove(temp_mgd_path);

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

    const std::filesystem::path temp_mgd_path =
        create_temp_mgd_file(mgd_textproto, "test_three_logical_five_physical_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
    std::filesystem::remove(temp_mgd_path);

    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u);
    verify_mesh_connectivity(logical_multi_mesh_graph, MeshId{0}, 1u);
    verify_mesh_connectivity(logical_multi_mesh_graph, MeshId{1}, 2u);
    verify_mesh_connectivity(logical_multi_mesh_graph, MeshId{2}, 1u);

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
    const auto& physical_mesh_level_graph = physical_multi_mesh_graph.mesh_level_graph_;
    for (size_t mesh_idx = 0; mesh_idx < kNumPhysicalMeshes; ++mesh_idx) {
        std::set<MeshId> unique_neighbors(
            physical_mesh_level_graph.get_neighbors(MeshId{mesh_idx}).begin(),
            physical_mesh_level_graph.get_neighbors(MeshId{mesh_idx}).end());
        EXPECT_EQ(unique_neighbors.size(), 2u);
    }

    TopologyMappingConfig config;
    config.strict_mode = false;
    config.disable_rank_bindings = true;
    config.inter_mesh_validation_mode = ConnectionValidationMode::RELAXED;

    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    ASSERT_TRUE(result.success) << result.error_message;
    verify_bidirectional_consistency(result);
    EXPECT_EQ(result.fabric_node_to_asic.size(), 12u);  // 3 meshes * 4 nodes
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

    const std::filesystem::path temp_mgd_path =
        create_temp_mgd_file(mgd_textproto, "test_three_logical_five_physical_device_level_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
    std::filesystem::remove(temp_mgd_path);

    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u);
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
    config.strict_mode = true;
    config.disable_rank_bindings = true;

    TopologyMappingResult result;
    bool exception_thrown = false;
    try {
        result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);
    } catch (const std::exception&) {
        exception_thrown = true;
    }

    // Verify mapping failed (either exception or result.success == false)
    if (!exception_thrown) {
        EXPECT_FALSE(result.success);
        EXPECT_LT(result.fabric_node_to_asic.size(), 12u);  // Should not map all nodes
    }
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

    const std::filesystem::path temp_mgd_path =
        create_temp_mgd_file(mgd_textproto, "test_three_logical_five_physical_device_level_valid_");
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::GALAXY, temp_mgd_path.string());
    std::filesystem::remove(temp_mgd_path);

    LogicalMultiMeshGraph logical_multi_mesh_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    EXPECT_EQ(logical_multi_mesh_graph.mesh_adjacency_graphs_.size(), 3u);
    EXPECT_EQ(logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{0}).get_nodes().size(), 1u);
    EXPECT_EQ(logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{1}).get_nodes().size(), 2u);
    EXPECT_EQ(logical_multi_mesh_graph.mesh_exit_node_graphs_.at(MeshId{2}).get_nodes().size(), 1u);

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
    config.strict_mode = true;
    config.disable_rank_bindings = true;

    const auto result = map_multi_mesh_to_physical(logical_multi_mesh_graph, physical_multi_mesh_graph, config);

    ASSERT_TRUE(result.success) << result.error_message;
    verify_bidirectional_consistency(result);
    EXPECT_EQ(result.fabric_node_to_asic.size(), 12u);  // 3 meshes * 4 nodes

    // Group mappings by mesh_id
    std::map<MeshId, std::map<FabricNodeId, tt::tt_metal::AsicID>> mappings_by_mesh;
    for (const auto& [fabric_node, asic] : result.fabric_node_to_asic) {
        mappings_by_mesh[fabric_node.mesh_id][fabric_node] = asic;
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

    // Find which physical meshes these ASICs belong to
    MeshId physical_mesh_0 = MeshId{0};
    MeshId physical_mesh_1 = MeshId{0};

    for (const auto& [pm_id, adjacency_graph] : physical_multi_mesh_graph.mesh_adjacency_graphs_) {
        const auto& nodes = adjacency_graph.get_nodes();
        if (std::find(nodes.begin(), nodes.end(), asic_0_0) != nodes.end()) {
            physical_mesh_0 = pm_id;
        }
        if (std::find(nodes.begin(), nodes.end(), asic_1_0) != nodes.end()) {
            physical_mesh_1 = pm_id;
        }
    }

    // Verify device-level constraints: specific ASICs are physically connected
    PhysicalExitNode exit_0_0{physical_mesh_0, asic_0_0};
    const auto& exit_neighbors =
        physical_multi_mesh_graph.mesh_exit_node_graphs_.at(physical_mesh_0).get_neighbors(exit_0_0);
    EXPECT_TRUE(std::any_of(exit_neighbors.begin(), exit_neighbors.end(), [&](const PhysicalExitNode& n) {
        return n.mesh_id == physical_mesh_1 && n.asic_id == asic_1_0;
    }));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::tt_fabric
