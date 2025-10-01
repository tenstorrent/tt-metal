// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_cycle_detector.hpp"

namespace tt::tt_fabric::fabric_tests {

// Removed get_fabric_path_from_control_plane - not needed for cycle detection

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
        log_debug(tt::LogTest, "No hops found for path {}->{} (likely inter-mesh traffic)", src, dest);
        // For inter-mesh traffic without control plane support, create a direct edge
        // This is a simplified representation for cycle detection purposes
        if (src.mesh_id != dest.mesh_id) {
            path_graph[src].push_back(dest);
            path_graph[dest] = {};  // Ensure destination exists
            log_debug(tt::LogTest, "Created direct inter-mesh edge for cycle detection: {}->{}", src, dest);
        }
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

// Main cycle detection function for inter-mesh traffic ONLY
// Intra-mesh traffic uses dimension-ordered routing and cannot have cycles
bool detect_cycles_in_traffic(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    uint32_t max_retry_attempts) {
    if (pairs.empty()) {
        log_debug(tt::LogTest, "No traffic pairs provided for cycle detection in test: {}", test_name);
        return false;
    }

    // Filter to ONLY inter-mesh traffic pairs - intra-mesh uses dimension-ordered routing (no cycles possible)
    std::vector<std::pair<FabricNodeId, FabricNodeId>> inter_mesh_pairs;
    for (const auto& [src, dest] : pairs) {
        if (src.mesh_id != dest.mesh_id) {
            inter_mesh_pairs.push_back({src, dest});
        }
    }

    if (inter_mesh_pairs.empty()) {
        log_debug(
            tt::LogTest,
            "No inter-mesh traffic pairs found in test '{}' - no cycle detection needed (intra-mesh uses "
            "dimension-ordered routing)",
            test_name);
        return false;  // No cycles possible in intra-mesh traffic
    }

    log_debug(
        tt::LogTest,
        "Starting inter-mesh cycle detection for test '{}' with {} inter-mesh pairs (filtered from {} total pairs)",
        test_name,
        inter_mesh_pairs.size(),
        pairs.size());

    // Validate input parameters
    if (max_retry_attempts == 0) {
        log_warning(tt::LogTest, "max_retry_attempts is 0, setting to 1");
        max_retry_attempts = 1;
    }

    // FIXED: Build a flow-aware graph that tracks which flows use which edges
    // This allows us to detect true circular resource dependencies, not just bidirectional traffic

    struct DirectedEdge {
        FabricNodeId from;
        FabricNodeId to;
        bool operator==(const DirectedEdge& other) const { return from == other.from && to == other.to; }
    };

    struct EdgeHash {
        std::size_t operator()(const DirectedEdge& edge) const {
            auto h1 = std::hash<uint32_t>{}(*edge.from.mesh_id);
            auto h2 = std::hash<uint32_t>{}(edge.from.chip_id);
            auto h3 = std::hash<uint32_t>{}(*edge.to.mesh_id);
            auto h4 = std::hash<uint32_t>{}(edge.to.chip_id);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };

    // Track which flows use which edges
    std::unordered_map<DirectedEdge, std::vector<std::pair<FabricNodeId, FabricNodeId>>, EdgeHash> edge_to_flows;
    NodeGraph comprehensive_graph;

    // Build comprehensive graph from ONLY inter-mesh traffic pairs
    for (const auto& [src, dest] : inter_mesh_pairs) {
        NodeGraph path_graph;

        // For inter-mesh routing, we need to build the actual path through ethernet links
        path_graph = build_path_graph(src, dest, route_manager);

        // Merge this path into the comprehensive graph AND track which flow uses which edge
        for (const auto& [node, neighbors] : path_graph) {
            for (const auto& neighbor : neighbors) {
                DirectedEdge edge{node, neighbor};
                edge_to_flows[edge].push_back({src, dest});

                auto& node_neighbors = comprehensive_graph[node];
                if (std::find(node_neighbors.begin(), node_neighbors.end(), neighbor) == node_neighbors.end()) {
                    node_neighbors.push_back(neighbor);
                }
            }
        }
    }

    // Detect cycles in the comprehensive graph
    auto cycles = detect_cycles(comprehensive_graph);

    // FIXED: Filter out false positive cycles caused by bidirectional traffic
    // A cycle is a FALSE POSITIVE if it's just back-and-forth traffic with no resource contention
    std::vector<CyclePath> true_cycles;
    for (const auto& cycle : cycles) {
        bool is_false_positive = false;

        // Check if this is a simple 2-node bidirectional cycle (A->B->A)
        if (cycle.size() == 3 && cycle[0] == cycle[2]) {
            FabricNodeId node_a = cycle[0];
            FabricNodeId node_b = cycle[1];

            DirectedEdge forward{node_a, node_b};
            DirectedEdge reverse{node_b, node_a};

            // Check if these edges are used by different flows (no contention)
            auto forward_it = edge_to_flows.find(forward);
            auto reverse_it = edge_to_flows.find(reverse);

            if (forward_it != edge_to_flows.end() && reverse_it != edge_to_flows.end()) {
                // Check if any flow uses BOTH edges (that would be a real cycle)
                bool same_flow_uses_both = false;
                for (const auto& forward_flow : forward_it->second) {
                    for (const auto& reverse_flow : reverse_it->second) {
                        if (forward_flow == reverse_flow) {
                            same_flow_uses_both = true;
                            break;
                        }
                    }
                    if (same_flow_uses_both) {
                        break;
                    }
                }

                // If different flows use opposite directions, it's NOT a deadlock
                if (!same_flow_uses_both) {
                    is_false_positive = true;
                    log_debug(
                        tt::LogTest,
                        "Filtering out false positive 2-node cycle {}->{}->{}: bidirectional traffic with no resource "
                        "contention",
                        node_a,
                        node_b,
                        node_a);
                }
            }
        }

        if (!is_false_positive) {
            true_cycles.push_back(cycle);
        }
    }

    // Use filtered cycles instead of all detected cycles
    cycles = true_cycles;

    if (!cycles.empty()) {
        log_warning(
            tt::LogTest,
            "Cycle detection found {} cycle(s) in inter-mesh traffic for test '{}' ({} inter-mesh pairs out of {} "
            "total pairs)",
            cycles.size(),
            test_name,
            inter_mesh_pairs.size(),
            pairs.size());

        // Dump cycles for debugging
        dump_cycles_to_yaml(cycles, test_name, 0, "generated/fabric");

        return true;  // Cycles detected
    }

    log_debug(
        tt::LogTest,
        "No cycles detected in inter-mesh traffic for test '{}' ({} inter-mesh pairs out of {} total pairs)",
        test_name,
        inter_mesh_pairs.size(),
        pairs.size());
    return false;  // No cycles detected
}

// Detect and handle cycles with retry logic
bool detect_cycles_with_retry(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const IRouteManager& route_manager,
    const std::string& test_name,
    bool is_deadlock_prevention_enabled) {
    if (!is_deadlock_prevention_enabled) {
        return false;  // Cycle detection disabled
    }

    return detect_cycles_in_traffic(pairs, route_manager, test_name);
}

// New control plane-based cycle detection function
bool detect_cycles(
    const std::vector<std::pair<FabricNodeId, FabricNodeId>>& pairs,
    const ControlPlane& control_plane,
    const std::string& test_name) {
    // Delegate directly to the control plane's cycle detection method
    return control_plane.detect_inter_mesh_cycles(pairs, test_name);
}

}  // namespace tt::tt_fabric::fabric_tests
