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

#include "tt_fabric_test_common.hpp"                    // For FabricNodeId, etc.
#include "tt_metal/fabric/routing_table_generator.hpp"  // For path generation
#include "tt_metal/fabric/mesh_graph.hpp"               // For connectivity
#include "assert.hpp"
#include "tt-logger/tt-logger.hpp"

namespace tt::tt_fabric::fabric_tests {

// Type aliases for clarity
using NodeGraph = std::unordered_map<FabricNodeId, std::vector<FabricNodeId>>;  // Directed graph: node -> neighbors
using CyclePath = std::vector<FabricNodeId>;  // A single cycle as a path (e.g., A -> B -> C -> A)

// DFS state for cycle detection
enum class DFSState { UNVISITED, VISITING, VISITED };

// Helper to detect cycles in a graph using DFS (finds all cycles via backtracking)
bool has_cycle_dfs(
    const NodeGraph& graph,
    FabricNodeId node,
    std::unordered_map<FabricNodeId, DFSState>& state,
    std::vector<FabricNodeId>& path,
    std::vector<CyclePath>& cycles) {
    state[node] = DFSState::VISITING;
    path.push_back(node);

    auto it = graph.find(node);
    if (it != graph.end()) {
        for (const auto& neighbor : it->second) {
            if (state[neighbor] == DFSState::UNVISITED) {
                if (has_cycle_dfs(graph, neighbor, state, path, cycles)) {
                    return true;
                }
            } else if (state[neighbor] == DFSState::VISITING) {
                // Back-edge found: cycle
                CyclePath cycle;
                auto cycle_start_it = std::find(path.begin(), path.end(), neighbor);
                if (cycle_start_it != path.end()) {
                    cycle.assign(cycle_start_it, path.end());
                    cycle.push_back(neighbor);  // Close the cycle
                    cycles.push_back(cycle);
                }
                return true;  // Cycle found, but continue to find others if needed
            }
        }
    }

    state[node] = DFSState::VISITED;
    path.pop_back();
    return false;
}

// Detect all cycles in a graph
std::vector<CyclePath> detect_cycles(const NodeGraph& graph) {
    std::unordered_map<FabricNodeId, DFSState> state;
    std::vector<CyclePath> cycles;
    std::vector<FabricNodeId> path;

    // Initialize state for all nodes
    for (const auto& [node, _] : graph) {
        state[node] = DFSState::UNVISITED;
    }

    // Run DFS from each unvisited node
    for (const auto& [node, _] : graph) {
        if (state[node] == DFSState::UNVISITED) {
            has_cycle_dfs(graph, node, state, path, cycles);
        }
    }

    return cycles;
}

// Build directed graph for a single (src, dest) path from routing tables
// Uses RoutingTableGenerator::get_paths_to_all_meshes; takes first path
NodeGraph build_path_graph(
    FabricNodeId src, FabricNodeId dest, const RoutingTableGenerator& rt_gen, const InterMeshConnectivity& inter_conn) {
    NodeGraph path_graph;

    // Query paths from src mesh to dest mesh
    auto paths = rt_gen.get_paths_to_all_meshes(src.mesh_id, inter_conn);
    auto& dest_paths = paths[*dest.mesh_id];
    if (dest_paths.empty()) {
        log_warning(tt::LogTest, "No path from {} to {}", src, dest);
        return {};
    }

    // Use first (shortest) path: vector of (chip_id, mesh_id) pairs
    const auto& path = dest_paths[0];
    FabricNodeId current(src.mesh_id, src.chip_id);
    path_graph[current] = {};

    // Build graph edges from path hops
    for (size_t i = 1; i < path.size(); ++i) {  // Start from first hop
        FabricNodeId next(path[i].second, path[i].first);
        path_graph[current].push_back(next);
        path_graph[next] = {};
        current = next;
    }

    // Ensure dest is reached
    if (current != dest) {
        log_warning(tt::LogTest, "Path does not reach dest {} from {}", dest, src);
    }

    return path_graph;
}

// Overlay multiple path graphs into one combined graph (union of edges)
NodeGraph overlay_graphs(const std::vector<NodeGraph>& path_graphs) {
    NodeGraph combined;
    for (const auto& pg : path_graphs) {
        for (const auto& [node, neighbors] : pg) {
            auto& combined_neighbors = combined[node];
            for (const auto& neigh : neighbors) {
                if (std::find(combined_neighbors.begin(), combined_neighbors.end(), neigh) ==
                    combined_neighbors.end()) {
                    combined_neighbors.push_back(neigh);
                }
            }
        }
    }
    return combined;
}

// Dump cycles to YAML file
void dump_cycles_to_yaml(
    const std::vector<CyclePath>& cycles,
    const std::string& test_name,
    int level,
    const std::string& output_dir = "generated/fabric") {
    std::filesystem::create_directories(output_dir);
    std::filesystem::path file_path =
        std::filesystem::path(output_dir) / ("cycles_" + test_name + "_level" + std::to_string(level) + ".yaml");

    YAML::Node yaml;
    YAML::Node cycle_list;

    for (const auto& cycle : cycles) {
        YAML::Node path_node;
        for (const auto& node : cycle) {
            path_node.push_back(fmt::format("M{}D{}", *node.mesh_id, node.chip_id));
        }
        cycle_list.push_back(path_node);
    }

    yaml["cycles"] = cycle_list;

    std::ofstream fout(file_path);
    if (fout.is_open()) {
        fout << yaml;
        log_info(tt::LogTest, "Dumped level {} cycles for test {} to {}", level, test_name, file_path.string());
    } else {
        log_error(tt::LogTest, "Failed to write cycles YAML for {}", test_name);
    }
}

// Main detection function
// pairs: Vector of (src, dest) from test expansion
// rt_gen: Access to routing table generator
// inter_conn: Inter-mesh connectivity for path queries
// Returns true if cycles detected (handles dumping and errors)
bool detect_and_handle_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const RoutingTableGenerator& rt_gen,
    const InterMeshConnectivity& inter_conn,
    const std::string& test_name,
    bool is_deadlock_prevention_enabled) {
    std::vector<NodeGraph> individual_graphs;
    std::vector<CyclePath> level1_cycles;

    // Level 1: Check each individual path for cycles
    for (const auto& [src, dest] : pairs) {
        auto path_graph = build_path_graph(src, dest, rt_gen, inter_conn);
        auto cycles = detect_cycles(path_graph);
        if (!cycles.empty()) {
            level1_cycles.insert(level1_cycles.end(), cycles.begin(), cycles.end());
        }
        individual_graphs.push_back(path_graph);
    }

    if (!level1_cycles.empty()) {
        dump_cycles_to_yaml(level1_cycles, test_name, 1);
        if (!is_deadlock_prevention_enabled) {
            TT_THROW("Level 1 cycles detected in routing tables for test {}", test_name);
        }
        log_warning(tt::LogTest, "Level 1 cycles detected in test {}", test_name);
        return true;  // Cycles found
    }

    // Level 2: Overlay and check combined graph
    auto combined_graph = overlay_graphs(individual_graphs);
    auto level2_cycles = detect_cycles(combined_graph);

    if (!level2_cycles.empty()) {
        dump_cycles_to_yaml(level2_cycles, test_name, 2);
        if (!is_deadlock_prevention_enabled) {
            TT_THROW("Level 2 cycles detected from pattern overlay in test {}", test_name);
        }
        log_warning(tt::LogTest, "Level 2 cycles detected in test {}", test_name);
        return true;
    }

    log_info(tt::LogTest, "No cycles detected for test {}", test_name);
    return false;  // No cycles
}

}  // namespace tt::tt_fabric::fabric_tests
