// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <filesystem>  // Add this to the includes section

#include "tt_fabric_test_interfaces.hpp"  // For IRouteManager, etc.
#include <tt-metalium/mesh_graph.hpp>     // For InterMeshConnectivity
#include <tt-metalium/control_plane.hpp>  // For ControlPlane access
#include "assert.hpp"
#include "tt-logger/tt-logger.hpp"

namespace tt::tt_fabric::fabric_tests {

// Type aliases for clarity
using NodeGraph = std::unordered_map<FabricNodeId, std::vector<FabricNodeId>>;  // Directed graph: node -> neighbors
using CyclePath = std::vector<FabricNodeId>;  // A single cycle as a path (e.g., A -> B -> C -> A)

// Function to get complete fabric path using control plane API
// This is kept within the test infrastructure to avoid modifying core control plane
std::vector<FabricNodeId> get_fabric_path_from_control_plane(
    const tt::tt_fabric::ControlPlane& control_plane, FabricNodeId src_fabric_node_id, FabricNodeId dst_fabric_node_id);

// DFS state for cycle detection
enum class DFSState { UNVISITED, VISITING, VISITED };

// Helper to detect cycles in a graph using DFS (finds all cycles via backtracking)
bool has_cycle_dfs(
    const NodeGraph& graph,
    FabricNodeId node,
    std::unordered_map<FabricNodeId, DFSState>& state,
    std::vector<FabricNodeId>& path,
    std::vector<CyclePath>& cycles);

// Entry point for cycle detection
std::vector<CyclePath> detect_cycles(const NodeGraph& graph);

// Build a routing path graph using full fabric path from control plane
NodeGraph build_path_graph_from_full_path(const std::vector<FabricNodeId>& full_path);

// Build a routing path graph by actually tracing the path hop-by-hop (with optional full path support)
NodeGraph build_path_graph(FabricNodeId src, FabricNodeId dest, const IRouteManager& route_manager);

// Enhanced function to build path graphs for multicast scenarios
NodeGraph build_multicast_path_graph(
    FabricNodeId src, const std::vector<FabricNodeId>& destinations, const IRouteManager& route_manager);

// Helper function to build path graphs from test traffic patterns
std::vector<NodeGraph> build_path_graphs_from_test_patterns(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs, const IRouteManager& route_manager);

// Enhanced function to build comprehensive path graphs that can handle complex routing scenarios
NodeGraph build_comprehensive_path_graph(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs, const IRouteManager& route_manager);

// Overlay multiple path graphs into one combined graph (union of edges)
NodeGraph overlay_graphs(const std::vector<NodeGraph>& path_graphs);

// Dump cycles to YAML for debugging
void dump_cycles_to_yaml(
    const std::vector<CyclePath>& cycles,
    const std::string& test_name,
    int level,
    const std::string& output_dir = "generated/fabric");

// Main cycle detection function for inter-mesh traffic
bool detect_cycles_in_random_inter_mesh_traffic(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    uint32_t max_retry_attempts = 3);

// Detect and handle cycles with retry logic
bool detect_and_handle_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    bool is_deadlock_prevention_enabled);

}  // namespace tt::tt_fabric::fabric_tests
