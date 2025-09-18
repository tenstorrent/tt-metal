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
    const tt::tt_fabric::ControlPlane& control_plane,
    FabricNodeId src_fabric_node_id,
    FabricNodeId dst_fabric_node_id) {
    try {
        // Get the complete route using the first available channel (channel 0)
        // For cycle detection, we only care about the node sequence, not the specific channels
        auto full_route = control_plane.get_fabric_route(src_fabric_node_id, dst_fabric_node_id, 0);

        // Extract just the FabricNodeId from each hop
        std::vector<FabricNodeId> path;
        path.reserve(full_route.size());

        for (const auto& [node_id, channel_id] : full_route) {
            path.push_back(node_id);
        }

        // Ensure the path includes the destination node if it's not already there
        if (!path.empty() && path.back() != dst_fabric_node_id) {
            path.push_back(dst_fabric_node_id);
        }

        return path;
    } catch (const std::exception& e) {
        log_warning(
            tt::LogTest,
            "Failed to get fabric route from control plane for {}->{}: {}",
            src_fabric_node_id,
            dst_fabric_node_id,
            e.what());
        return {};
    }
}

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
                return true;
            }
        }
    }

    path.pop_back();
    state[node] = DFSState::VISITED;
    return false;
}

// Entry point for cycle detection
std::vector<CyclePath> detect_cycles(const NodeGraph& graph) {
    std::vector<CyclePath> cycles;
    std::unordered_map<FabricNodeId, DFSState> state;
    std::vector<FabricNodeId> path;

    // Initialize all nodes as unvisited
    for (const auto& [node, _] : graph) {
        state[node] = DFSState::UNVISITED;
    }

    // Check each unvisited node
    for (const auto& [node, _] : graph) {
        if (state[node] == DFSState::UNVISITED) {
            has_cycle_dfs(graph, node, state, path, cycles);
        }
    }

    return cycles;
}

// Build a routing path graph using full fabric path from control plane
NodeGraph build_path_graph_from_full_path(const std::vector<FabricNodeId>& full_path) {
    NodeGraph path_graph;

    if (full_path.size() < 2) {
        return path_graph;
    }

    // Convert the linear path into a directed graph
    for (size_t i = 0; i < full_path.size() - 1; ++i) {
        FabricNodeId current_node = full_path[i];
        FabricNodeId next_node = full_path[i + 1];

        // Initialize nodes if not present
        if (path_graph.find(current_node) == path_graph.end()) {
            path_graph[current_node] = {};
        }
        if (path_graph.find(next_node) == path_graph.end()) {
            path_graph[next_node] = {};
        }

        // Add edge from current to next node
        path_graph[current_node].push_back(next_node);
    }

    return path_graph;
}

// Build a routing path graph by actually tracing the path hop-by-hop (with optional full path support)
NodeGraph build_path_graph(FabricNodeId src, FabricNodeId dest, const IRouteManager& route_manager) {
    // Try to use the full fabric path API if available
    auto full_path = route_manager.get_full_fabric_path(src, dest);
    if (!full_path.empty()) {
        return build_path_graph_from_full_path(full_path);
    }

    // Fallback to hop-based path building
    NodeGraph path_graph;

    try {
        // Get the routing hops from source to destination
        auto hops = route_manager.get_hops_to_chip(src, dest);

        // Start tracing from the source
        FabricNodeId current_node = src;
        path_graph[current_node] = {};  // Initialize source node

        // Create a copy of hops to track remaining hops in each direction
        std::unordered_map<RoutingDirection, uint32_t> remaining_hops = hops;

        // Trace the path by following the routing directions
        while (!remaining_hops.empty()) {
            // Find the next direction to route in (prefer non-zero hop counts)
            RoutingDirection next_direction = RoutingDirection::NONE;

            for (const auto& [direction, hop_count] : remaining_hops) {
                if (hop_count > 0) {
                    next_direction = direction;
                    break;
                }
            }

            if (next_direction == RoutingDirection::NONE) {
                // No more hops to make
                break;
            }

            uint32_t hops_in_direction = remaining_hops[next_direction];

            // Trace all hops in this direction sequentially
            for (uint32_t hop = 0; hop < hops_in_direction; hop++) {
                try {
                    // Get the next node in this direction
                    FabricNodeId next_node = route_manager.get_neighbor_node_id(current_node, next_direction);

                    // Add edge from current to next node
                    path_graph[current_node].push_back(next_node);

                    // Initialize next node if not already present
                    if (path_graph.find(next_node) == path_graph.end()) {
                        path_graph[next_node] = {};
                    }

                    // Move to next node
                    current_node = next_node;

                } catch (const std::exception& e) {
                    log_warning(
                        tt::LogTest,
                        "Failed to get neighbor for {} in direction {}: {}",
                        current_node,
                        static_cast<int>(next_direction),
                        e.what());
                    // If we can't get the neighbor, create a direct edge to destination
                    if (current_node != dest) {
                        path_graph[current_node].push_back(dest);
                        path_graph[dest] = {};
                    }
                    return path_graph;
                }
            }

            // Mark this direction as completed
            remaining_hops.erase(next_direction);
        }

        // Ensure we end up at the destination
        if (current_node != dest) {
            // If we didn't reach the destination through normal routing,
            // add a direct edge (this might indicate a routing issue)
            path_graph[current_node].push_back(dest);
            path_graph[dest] = {};
            log_warning(
                tt::LogTest, "Path tracing for {}->{} ended at {} instead of destination", src, dest, current_node);
        }

    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "Exception during path building for {}->{}: {}", src, dest, e.what());
        // Fall back to simple direct connection
        path_graph[src].push_back(dest);
        path_graph[dest] = {};
    }

    return path_graph;
}

// Enhanced function to build path graphs for multicast scenarios
NodeGraph build_multicast_path_graph(
    FabricNodeId src, const std::vector<FabricNodeId>& destinations, const IRouteManager& route_manager) {
    NodeGraph combined_graph;

    // Build individual paths to each destination and combine them
    for (const auto& dest : destinations) {
        auto individual_path = build_path_graph(src, dest, route_manager);

        // Merge this path into the combined graph
        for (const auto& [node, neighbors] : individual_path) {
            auto& combined_neighbors = combined_graph[node];
            for (const auto& neighbor : neighbors) {
                if (std::find(combined_neighbors.begin(), combined_neighbors.end(), neighbor) ==
                    combined_neighbors.end()) {
                    combined_neighbors.push_back(neighbor);
                }
            }
        }
    }

    return combined_graph;
}

// Helper function to build path graphs from test traffic patterns
std::vector<NodeGraph> build_path_graphs_from_test_patterns(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs, const IRouteManager& route_manager) {
    std::vector<NodeGraph> path_graphs;
    path_graphs.reserve(pairs.size());

    for (const auto& [src, dest] : pairs) {
        auto path_graph = build_path_graph(src, dest, route_manager);
        path_graphs.push_back(std::move(path_graph));
    }

    return path_graphs;
}

// Enhanced function to build comprehensive path graphs that can handle complex routing scenarios
NodeGraph build_comprehensive_path_graph(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs, const IRouteManager& route_manager) {
    NodeGraph comprehensive_graph;

    // Build individual path graphs and merge them
    for (const auto& [src, dest] : pairs) {
        auto individual_path = build_path_graph(src, dest, route_manager);

        // Merge into comprehensive graph
        for (const auto& [node, neighbors] : individual_path) {
            auto& combined_neighbors = comprehensive_graph[node];
            for (const auto& neighbor : neighbors) {
                if (std::find(combined_neighbors.begin(), combined_neighbors.end(), neighbor) ==
                    combined_neighbors.end()) {
                    combined_neighbors.push_back(neighbor);
                }
            }
        }
    }

    return comprehensive_graph;
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

void dump_cycles_to_yaml(
    const std::vector<CyclePath>& cycles,
    const std::string& test_name,
    int level,
    const std::string& output_dir = "generated/fabric") {
    if (cycles.empty()) {
        return;
    }

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);

    // Generate filename
    std::string file_path = fmt::format("{}/cycles_detected_{}_{}.yaml", output_dir, test_name, level);

    std::ofstream fout(file_path);
    if (!fout.is_open()) {
        log_warning(tt::LogTest, "Failed to open file for cycle output: {}", file_path);
        return;
    }

    fout << "# Cycle detection results\n";
    fout << "test_name: " << test_name << "\n";
    fout << "level: " << level << "\n";
    fout << "cycles_found: " << cycles.size() << "\n";
    fout << "cycles:\n";

    for (size_t i = 0; i < cycles.size(); ++i) {
        fout << "  - id: " << i << "\n";
        fout << "    path:\n";
        for (const auto& node : cycles[i]) {
            fout << "      - mesh_id: " << *node.mesh_id << "\n";
            fout << "        chip_id: " << node.chip_id << "\n";
        }
    }

    fout.close();
    log_info(tt::LogTest, "Cycle detection results written to: {}", file_path);
}

// Enhanced cycle detection specifically for random inter-mesh traffic
bool detect_cycles_in_random_inter_mesh_traffic(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    uint32_t max_retry_attempts = 3) {
    log_info(tt::LogTest, "Checking for cycles in random inter-mesh traffic for test {}", test_name);

    // Build comprehensive graph using full fabric paths for better accuracy
    NodeGraph comprehensive_graph;
    std::vector<std::vector<FabricNodeId>> all_paths;

    // Collect all fabric paths
    const tt::tt_fabric::ControlPlane* control_plane =
        static_cast<const tt::tt_fabric::ControlPlane*>(route_manager.get_control_plane());

    for (const auto& [src, dest] : pairs) {
        std::vector<FabricNodeId> full_path;

        // Try to use control plane directly for more accurate paths
        if (control_plane != nullptr) {
            full_path = get_fabric_path_from_control_plane(*control_plane, src, dest);
        }

        // Fallback to route manager's implementation if control plane is not available or fails
        if (full_path.empty()) {
            full_path = route_manager.get_full_fabric_path(src, dest);
        }

        if (!full_path.empty()) {
            all_paths.push_back(full_path);

            // Build path graph from this full path
            auto path_graph = build_path_graph_from_full_path(full_path);

            // Merge into comprehensive graph
            for (const auto& [node, neighbors] : path_graph) {
                auto& combined_neighbors = comprehensive_graph[node];
                for (const auto& neighbor : neighbors) {
                    if (std::find(combined_neighbors.begin(), combined_neighbors.end(), neighbor) ==
                        combined_neighbors.end()) {
                        combined_neighbors.push_back(neighbor);
                    }
                }
            }
        } else {
            // Last resort: fallback to hop-based path building
            auto path_graph = build_path_graph(src, dest, route_manager);
            for (const auto& [node, neighbors] : path_graph) {
                auto& combined_neighbors = comprehensive_graph[node];
                for (const auto& neighbor : neighbors) {
                    if (std::find(combined_neighbors.begin(), combined_neighbors.end(), neighbor) ==
                        combined_neighbors.end()) {
                        combined_neighbors.push_back(neighbor);
                    }
                }
            }
        }
    }

    // Detect cycles in the comprehensive graph
    auto cycles = detect_cycles(comprehensive_graph);

    if (!cycles.empty()) {
        log_warning(tt::LogTest, "Found {} cycles in inter-mesh traffic pattern for test {}", cycles.size(), test_name);
        dump_cycles_to_yaml(cycles, test_name, 0, "generated/fabric/inter_mesh_cycles");

        // Log detailed cycle information for debugging
        for (size_t i = 0; i < cycles.size(); ++i) {
            log_warning(tt::LogTest, "Cycle {}: ", i);
            for (const auto& node : cycles[i]) {
                log_warning(tt::LogTest, "  -> {}", node);
            }
        }

        return true;  // Cycles found - caller should regenerate random pairing
    }

    log_info(tt::LogTest, "No cycles detected in inter-mesh traffic pattern for test {}", test_name);
    return false;  // No cycles
}

bool detect_and_handle_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    bool is_deadlock_prevention_enabled) {
    std::vector<CyclePath> level1_cycles;
    std::vector<NodeGraph> individual_graphs;

    // Level 1: Check each individual path for cycles
    log_info(tt::LogTest, "Level 1: Checking individual paths for cycles in test {}", test_name);

    for (const auto& [src, dest] : pairs) {
        auto path_graph = build_path_graph(src, dest, route_manager);
        auto cycles = detect_cycles(path_graph);
        if (!cycles.empty()) {
            log_warning(tt::LogTest, "Found {} cycles in path from {} to {}", cycles.size(), src, dest);
            level1_cycles.insert(level1_cycles.end(), cycles.begin(), cycles.end());
        }
        individual_graphs.push_back(std::move(path_graph));
    }

    if (!level1_cycles.empty()) {
        log_warning(tt::LogTest, "Level 1: Found {} total cycles across individual paths", level1_cycles.size());
        dump_cycles_to_yaml(level1_cycles, test_name, 1);

        if (!is_deadlock_prevention_enabled) {
            TT_THROW("Level 1 cycles detected in routing tables for test {}", test_name);
        }
        return true;  // Cycles found
    }

    // Level 2: Build comprehensive graph and check for cycles when paths overlay
    log_info(tt::LogTest, "Level 2: Checking overlaid path graph for cycles in test {}", test_name);

    auto comprehensive_graph = build_comprehensive_path_graph(pairs, route_manager);
    auto level2_cycles = detect_cycles(comprehensive_graph);

    if (!level2_cycles.empty()) {
        log_warning(tt::LogTest, "Level 2: Found {} cycles in overlaid path graph", level2_cycles.size());
        dump_cycles_to_yaml(level2_cycles, test_name, 2);

        if (!is_deadlock_prevention_enabled) {
            TT_THROW("Level 2 cycles detected from pattern overlay in test {}", test_name);
        }
        return true;
    }

    log_info(tt::LogTest, "No cycles detected for test {}", test_name);
    return false;  // No cycles
}

}  // namespace tt::tt_fabric::fabric_tests
