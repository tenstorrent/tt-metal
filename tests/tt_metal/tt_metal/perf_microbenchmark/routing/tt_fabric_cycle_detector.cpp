// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_cycle_detector.hpp"

namespace tt::tt_fabric::fabric_tests {

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

    // Create directed edges for each consecutive pair in the path
    for (size_t i = 0; i < full_path.size() - 1; ++i) {
        const auto& current_node = full_path[i];
        const auto& next_node = full_path[i + 1];

        // Add edge: current_node -> next_node
        path_graph[current_node].push_back(next_node);
    }

    return path_graph;
}

// Build a routing path graph using hop-based routing (fallback method)
NodeGraph build_path_graph(FabricNodeId src, FabricNodeId dest, const IRouteManager& route_manager) {
    NodeGraph path_graph;

    // Try to get the full fabric path first
    try {
        auto full_path = route_manager.get_full_fabric_path(src, dest);
        if (!full_path.empty()) {
            return build_path_graph_from_full_path(full_path);
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogTest, "Failed to get full fabric path for {}->{}: {}", src, dest, e.what());
    }

    // Fallback to hop-based path construction
    auto hops = route_manager.get_hops_to_chip(src, dest);
    if (hops.empty()) {
        log_warning(tt::LogTest, "No hops found for path {}->{}", src, dest);
        return path_graph;
    }

    FabricNodeId current_node = src;
    path_graph[current_node] = {};  // Ensure source node exists in graph

    // Build path by following hops
    for (const auto& [direction, num_hops] : hops) {
        if (num_hops == 0) {
            continue;
        }

        for (uint32_t hop = 0; hop < num_hops; ++hop) {
            FabricNodeId next_node = route_manager.get_neighbor_node_id(current_node, direction);

            // Add directed edge: current_node -> next_node
            path_graph[current_node].push_back(next_node);

            // Ensure next_node exists in the graph (even if it has no outgoing edges)
            if (path_graph.find(next_node) == path_graph.end()) {
                path_graph[next_node] = {};
            }

            current_node = next_node;
        }
    }

    // Verify we reached the destination
    if (current_node != dest) {
        log_warning(
            tt::LogTest, "Path construction for {}->{} ended at {} instead of destination", src, dest, current_node);

        // Add final edge to destination if needed
        if (current_node != dest) {
            path_graph[current_node].push_back(dest);
            path_graph[dest] = {};  // Ensure destination exists
        }
    }

    return path_graph;
}

// Build multicast path graph
NodeGraph build_multicast_path_graph(
    FabricNodeId src, const std::vector<FabricNodeId>& destinations, const IRouteManager& route_manager) {
    NodeGraph combined_graph;

    // Build individual paths to each destination and merge them
    for (const auto& dest : destinations) {
        auto path_graph = build_path_graph(src, dest, route_manager);

        // Merge this path into the combined graph
        for (const auto& [node, neighbors] : path_graph) {
            for (const auto& neighbor : neighbors) {
                // Add edge if it doesn't already exist
                auto& node_neighbors = combined_graph[node];
                if (std::find(node_neighbors.begin(), node_neighbors.end(), neighbor) == node_neighbors.end()) {
                    node_neighbors.push_back(neighbor);
                }
            }
        }
    }

    return combined_graph;
}

// Build path graphs from test patterns
std::vector<NodeGraph> build_path_graphs_from_test_patterns(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs, const IRouteManager& route_manager) {
    std::vector<NodeGraph> path_graphs;
    path_graphs.reserve(pairs.size());

    for (const auto& [src, dest] : pairs) {
        path_graphs.push_back(build_path_graph(src, dest, route_manager));
    }

    return path_graphs;
}

// Build comprehensive path graph from all pairs
NodeGraph build_comprehensive_path_graph(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs, const IRouteManager& route_manager) {
    auto path_graphs = build_path_graphs_from_test_patterns(pairs, route_manager);
    return overlay_graphs(path_graphs);
}

// Overlay multiple graphs into a single graph
NodeGraph overlay_graphs(const std::vector<NodeGraph>& path_graphs) {
    NodeGraph combined_graph;

    for (const auto& graph : path_graphs) {
        for (const auto& [node, neighbors] : graph) {
            for (const auto& neighbor : neighbors) {
                // Add edge if it doesn't already exist
                auto& node_neighbors = combined_graph[node];
                if (std::find(node_neighbors.begin(), node_neighbors.end(), neighbor) == node_neighbors.end()) {
                    node_neighbors.push_back(neighbor);
                }
            }
        }
    }

    return combined_graph;
}

// Dump cycles to YAML for debugging
void dump_cycles_to_yaml(
    const std::vector<CyclePath>& cycles, const std::string& test_name, int level, const std::string& output_dir) {
    if (cycles.empty()) {
        return;
    }

    std::filesystem::create_directories(output_dir);
    std::string file_path = output_dir + "/cycles_" + test_name + "_level_" + std::to_string(level) + ".yaml";

    std::ofstream fout(file_path);
    if (!fout.is_open()) {
        log_warning(tt::LogTest, "Failed to open file for writing: {}", file_path);
        return;
    }

    fout << "test_name: " << test_name << "\n";
    fout << "level: " << level << "\n";
    fout << "cycles_found: " << cycles.size() << "\n";
    fout << "cycles:\n";

    for (size_t i = 0; i < cycles.size(); ++i) {
        fout << "  - cycle_" << i << ":\n";
        fout << "      path: [";
        for (size_t j = 0; j < cycles[i].size(); ++j) {
            const auto& node = cycles[i][j];
            fout << "{mesh_id: " << *node.mesh_id << ", chip_id: " << node.chip_id << "}";
            if (j < cycles[i].size() - 1) {
                fout << ", ";
            }
        }
        fout << "]\n";
    }

    fout.close();
    log_info(tt::LogTest, "Cycles dumped to: {}", file_path);
}

// Main cycle detection function for inter-mesh traffic
bool detect_cycles_in_random_inter_mesh_traffic(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    uint32_t max_retry_attempts) {
    if (pairs.empty()) {
        log_debug(tt::LogTest, "No traffic pairs provided for cycle detection in test: {}", test_name);
        return false;
    }

    log_debug(tt::LogTest, "Starting cycle detection for test '{}' with {} traffic pairs", test_name, pairs.size());

    NodeGraph comprehensive_graph;

    // Try to get control plane for enhanced path accuracy
    const void* control_plane_ptr = route_manager.get_control_plane();
    const tt::tt_fabric::ControlPlane* control_plane = nullptr;
    if (control_plane_ptr) {
        control_plane = static_cast<const tt::tt_fabric::ControlPlane*>(control_plane_ptr);
    }

    // Build comprehensive graph from all traffic pairs
    for (const auto& [src, dest] : pairs) {
        NodeGraph path_graph;

        // First, try to get the full fabric path using control plane API
        if (control_plane) {
            try {
                auto full_path = get_fabric_path_from_control_plane(*control_plane, src, dest);
                if (!full_path.empty()) {
                    path_graph = build_path_graph_from_full_path(full_path);
                } else {
                    log_debug(tt::LogTest, "Control plane returned empty path for {}->{}, using fallback", src, dest);
                    path_graph = build_path_graph(src, dest, route_manager);
                }
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogTest,
                    "Control plane path retrieval failed for {}->{}: {}, using fallback",
                    src,
                    dest,
                    e.what());
                path_graph = build_path_graph(src, dest, route_manager);
            }
        } else {
            // Fallback to route manager's get_full_fabric_path
            try {
                auto full_path = route_manager.get_full_fabric_path(src, dest);
                if (!full_path.empty()) {
                    path_graph = build_path_graph_from_full_path(full_path);
                } else {
                    log_debug(
                        tt::LogTest,
                        "Route manager returned empty path for {}->{}, using hop-based fallback",
                        src,
                        dest);
                    path_graph = build_path_graph(src, dest, route_manager);
                }
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogTest,
                    "Route manager path retrieval failed for {}->{}: {}, using hop-based fallback",
                    src,
                    dest,
                    e.what());
                path_graph = build_path_graph(src, dest, route_manager);
            }
        }

        // Merge this path into the comprehensive graph
        for (const auto& [node, neighbors] : path_graph) {
            for (const auto& neighbor : neighbors) {
                auto& node_neighbors = comprehensive_graph[node];
                if (std::find(node_neighbors.begin(), node_neighbors.end(), neighbor) == node_neighbors.end()) {
                    node_neighbors.push_back(neighbor);
                }
            }
        }
    }

    // Detect cycles in the comprehensive graph
    auto cycles = detect_cycles(comprehensive_graph);

    if (!cycles.empty()) {
        log_warning(
            tt::LogTest,
            "Cycle detection found {} cycle(s) in test '{}' with {} traffic pairs",
            cycles.size(),
            test_name,
            pairs.size());

        // Dump cycles for debugging
        dump_cycles_to_yaml(cycles, test_name, 0, "generated/fabric");

        return true;  // Cycles detected
    }

    log_debug(tt::LogTest, "No cycles detected in test '{}' with {} traffic pairs", test_name, pairs.size());
    return false;  // No cycles detected
}

// Detect and handle cycles with retry logic
bool detect_and_handle_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    bool is_deadlock_prevention_enabled) {
    if (!is_deadlock_prevention_enabled) {
        return false;  // Cycle detection disabled
    }

    return detect_cycles_in_random_inter_mesh_traffic(pairs, route_manager, test_name);
}

}  // namespace tt::tt_fabric::fabric_tests
