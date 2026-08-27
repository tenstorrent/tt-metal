// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>

#include <algorithm>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_metal/impl/context/metal_context.hpp"
#include "tt-metalium/experimental/fabric/control_plane.hpp"

namespace tt::tt_fabric {

// ------------------------------------------------------------------
// Low-level control-plane wrappers
// ------------------------------------------------------------------

std::optional<RoutingDirection> pipeline_get_forwarding_direction(FabricNodeId src, FabricNodeId dst) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto dir_opt = cp.get_forwarding_direction(src, dst);
    if (!dir_opt) {
        return std::nullopt;
    }
    auto planes = cp.get_active_fabric_eth_routing_planes_in_direction(src, *dir_opt);
    if (planes.empty()) {
        return std::nullopt;
    }
    return dir_opt;
}

std::map<uint32_t, std::vector<uint32_t>> pipeline_get_chip_neighbors(FabricNodeId src, RoutingDirection direction) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto raw = cp.get_chip_neighbors(src, direction);
    std::map<uint32_t, std::vector<uint32_t>> result;
    for (const auto& [mesh_id, chip_ids] : raw) {
        auto& vec = result[*mesh_id];
        vec.insert(vec.end(), chip_ids.begin(), chip_ids.end());
    }
    return result;
}

// ------------------------------------------------------------------
// Graph layout resolution
// ------------------------------------------------------------------

namespace {

struct InternalChip {
    FabricNodeId fid;
    uint32_t row, col;
};

// Physical direct-link info between a pair of submeshes. All real links are
// retained so endpoint capabilities can participate in global assignment.
struct ConnectionInfo {
    std::vector<PipelineEndpointLink> links;  // all valid direct ethernet links, first = primary
};

using ConnectionKey = std::pair<size_t, size_t>;  // (submesh_i, submesh_j)
using EdgeLinkAssignment = std::vector<PipelineEndpointLink>;

/// Discover all direct ethernet links between every ordered pair of submeshes.
/// All valid link pairs are collected (not just the first) for constrained assignment.
std::map<ConnectionKey, ConnectionInfo> discover_connections(const std::vector<std::vector<InternalChip>>& chips) {
    std::map<ConnectionKey, ConnectionInfo> connections;
    size_t n = chips.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            for (size_t ai = 0; ai < chips[i].size(); ++ai) {
                for (size_t bi = 0; bi < chips[j].size(); ++bi) {
                    const auto& ca = chips[i][ai];
                    const auto& cb = chips[j][bi];
                    auto dir_opt = pipeline_get_forwarding_direction(ca.fid, cb.fid);
                    if (!dir_opt) {
                        continue;
                    }
                    auto neighbors = pipeline_get_chip_neighbors(ca.fid, *dir_opt);
                    uint32_t b_mesh = *cb.fid.mesh_id;
                    auto it = neighbors.find(b_mesh);
                    if (it == neighbors.end()) {
                        continue;
                    }
                    const auto& nlist = it->second;
                    if (std::find(nlist.begin(), nlist.end(), cb.fid.chip_id) != nlist.end()) {
                        connections[{i, j}].links.push_back({ca.row, ca.col, cb.row, cb.col});
                    }
                }
            }
        }
    }
    return connections;
}

/// Select one real physical link for every graph edge while enforcing endpoint
/// capabilities. The edge order and each ConnectionInfo::links order are stable,
/// making the first satisfying assignment deterministic.
std::optional<EdgeLinkAssignment> assign_edge_links(
    const std::vector<EdgeInputTuple>& edges,
    const std::map<std::string, size_t>& node_to_sub,
    const std::map<ConnectionKey, ConnectionInfo>& connections,
    const std::vector<std::vector<InternalChip>>& chips,
    const std::set<std::string>& nodes_requiring_distinct_endpoints) {
    std::vector<std::vector<PipelineEndpointLink>> edge_candidates;
    edge_candidates.reserve(edges.size());
    for (const auto& [src, dst, _] : edges) {
        auto connection_it = connections.find({node_to_sub.at(src), node_to_sub.at(dst)});
        if (connection_it == connections.end()) {
            return std::nullopt;
        }
        edge_candidates.push_back(connection_it->second.links);
    }
    std::map<std::string, uint32_t> assigned_node_chip_counts;
    for (const auto& [node, submesh_idx] : node_to_sub) {
        assigned_node_chip_counts[node] = static_cast<uint32_t>(chips[submesh_idx].size());
    }
    return select_pipeline_endpoint_links(
        edges, edge_candidates, assigned_node_chip_counts, nodes_requiring_distinct_endpoints);
}

/// Kahn's topological sort on non-loopback edges. Returns node names in stage order.
std::vector<std::string> topological_sort(
    const std::vector<std::string>& all_nodes, const std::vector<EdgeInputTuple>& edges) {
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> adj;
    for (const auto& n : all_nodes) {
        in_degree[n] = 0;
    }
    for (const auto& [src, dst, is_lb] : edges) {
        if (!is_lb) {
            adj[src].push_back(dst);
            in_degree[dst]++;
        }
    }
    std::queue<std::string> q;
    for (const auto& [n, deg] : in_degree) {
        if (deg == 0) {
            q.push(n);
        }
    }

    std::vector<std::string> order;
    order.reserve(all_nodes.size());
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        order.push_back(node);
        for (const auto& dst : adj[node]) {
            if (--in_degree[dst] == 0) {
                q.push(dst);
            }
        }
    }
    if (order.size() != all_nodes.size()) {
        throw std::runtime_error("resolve_graph_layout: cycle detected in non-loopback edges");
    }
    return order;
}

/// Backtracking search: assign submesh indices to each node in topological order.
/// Returns {node_name -> submesh_index} or throws if no valid assignment exists.
std::map<std::string, size_t> assign_submeshes(
    const std::vector<std::string>& stage_order,
    const std::vector<EdgeInputTuple>& edges,
    const std::map<ConnectionKey, ConnectionInfo>& connections,
    size_t num_submeshes,
    const std::map<std::string, uint32_t>& node_chip_counts,
    const std::vector<std::vector<InternalChip>>& chips,
    const std::set<std::string>& nodes_requiring_distinct_endpoints) {
    // Build reverse-lookup: dst -> [src] for non-loopback edges
    std::map<std::string, std::vector<std::string>> parents;
    for (const auto& [src, dst, is_lb] : edges) {
        if (!is_lb) {
            parents[dst].push_back(src);
        }
    }

    std::map<std::string, size_t> node_to_sub;
    std::set<size_t> used;

    std::function<bool(size_t)> solve = [&](size_t idx) -> bool {
        if (idx == stage_order.size()) {
            // Verify every loopback edge has a direct physical link.
            for (const auto& [src, dst, is_lb] : edges) {
                if (!is_lb) {
                    continue;
                }
                size_t si = node_to_sub.at(src);
                size_t sj = node_to_sub.at(dst);
                if (!connections.contains({si, sj})) {
                    return false;
                }
            }
            return assign_edge_links(edges, node_to_sub, connections, chips, nodes_requiring_distinct_endpoints)
                .has_value();
        }
        const auto& node = stage_order[idx];

        // Compute candidate submeshes: unassigned AND directly reachable from ALL parents.
        // Source nodes (no parents) may use any unassigned submesh.
        bool constrained = false;
        std::set<size_t> candidates;

        auto it_p = parents.find(node);
        if (it_p != parents.end()) {
            for (const auto& parent : it_p->second) {
                size_t psub = node_to_sub.at(parent);
                std::set<size_t> reachable;
                for (size_t j = 0; j < num_submeshes; ++j) {
                    if (!used.contains(j) && connections.contains({psub, j})) {
                        reachable.insert(j);
                    }
                }
                if (!constrained) {
                    candidates = reachable;
                    constrained = true;
                } else {
                    std::set<size_t> intersect;
                    for (auto s : reachable) {
                        if (candidates.contains(s)) {
                            intersect.insert(s);
                        }
                    }
                    candidates = intersect;
                }
            }
        }
        if (!constrained) {
            for (size_t j = 0; j < num_submeshes; ++j) {
                if (!used.contains(j)) {
                    candidates.insert(j);
                }
            }
        }

        // Shape constraint: a node may only land on a submesh whose chip count
        // matches the node's declared shape (rows*cols).  Without this, a stage
        // declared 4x2 can be placed on a 1x2 submesh of a different mesh whenever
        // ethernet connectivity allows, silently mis-placing it (e.g. the final-
        // layer / loopback stage landing on a 1x2 instead of its 4x2 mesh).
        auto cc_it = node_chip_counts.find(node);
        if (cc_it != node_chip_counts.end()) {
            std::set<size_t> filtered;
            for (size_t s : candidates) {
                if (chips[s].size() == cc_it->second) {
                    filtered.insert(s);
                }
            }
            candidates = std::move(filtered);
        }

        for (size_t sub : candidates) {
            node_to_sub[node] = sub;
            used.insert(sub);
            if (solve(idx + 1)) {
                return true;
            }
            node_to_sub.erase(node);
            used.erase(sub);
        }
        return false;
    };

    if (!solve(0)) {
        std::string message =
            "resolve_graph_layout: no valid submesh and physical-link assignment found — "
            "physical connectivity and per-node shape do not match the graph topology";
        if (!nodes_requiring_distinct_endpoints.empty()) {
            message +=
                ", or no topology-valid assignment gives distinct ingress/egress chips for required "
                "multi-chip stages [";
            bool first = true;
            for (const auto& node : nodes_requiring_distinct_endpoints) {
                message += (first ? "" : ", ") + node;
                first = false;
            }
            message +=
                "]. Check stage orientation, submesh ordering, and whether each constrained turn has a second "
                "physical boundary link";
        }
        throw std::runtime_error(message);
    }
    return node_to_sub;
}

}  // anonymous namespace

std::optional<std::vector<PipelineEndpointLink>> select_pipeline_endpoint_links(
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<PipelineEndpointLink>>& edge_candidates,
    const std::map<std::string, uint32_t>& assigned_node_chip_counts,
    const std::set<std::string>& nodes_requiring_distinct_endpoints) {
    if (edges.size() != edge_candidates.size()) {
        throw std::invalid_argument("select_pipeline_endpoint_links: each edge must have one candidate list");
    }

    std::map<std::string, std::vector<size_t>> incoming_edges;
    std::map<std::string, std::vector<size_t>> outgoing_edges;
    for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
        const auto& [src, dst, is_loopback] = edges[edge_idx];
        if (!is_loopback) {
            incoming_edges[dst].push_back(edge_idx);
        }
        outgoing_edges[src].push_back(edge_idx);
    }

    auto endpoints_are_valid = [&](const std::vector<PipelineEndpointLink>& assignment, size_t assigned_count) {
        for (const auto& node : nodes_requiring_distinct_endpoints) {
            auto count_it = assigned_node_chip_counts.find(node);
            if (count_it == assigned_node_chip_counts.end() || count_it->second <= 1) {
                continue;
            }
            auto in_it = incoming_edges.find(node);
            auto out_it = outgoing_edges.find(node);
            if (in_it == incoming_edges.end() || out_it == outgoing_edges.end()) {
                continue;
            }
            for (size_t in_idx : in_it->second) {
                for (size_t out_idx : out_it->second) {
                    if (in_idx >= assigned_count || out_idx >= assigned_count) {
                        continue;
                    }
                    const auto& in_link = assignment[in_idx];
                    const auto& out_link = assignment[out_idx];
                    if (in_link.entry_row == out_link.exit_row && in_link.entry_col == out_link.exit_col) {
                        return false;
                    }
                }
            }
        }
        return true;
    };

    std::vector<PipelineEndpointLink> assignment(edges.size());
    std::function<bool(size_t)> solve = [&](size_t edge_idx) {
        if (edge_idx == edges.size()) {
            return endpoints_are_valid(assignment, assignment.size());
        }
        for (const auto& link : edge_candidates[edge_idx]) {
            assignment[edge_idx] = link;
            if (endpoints_are_valid(assignment, edge_idx + 1) && solve(edge_idx + 1)) {
                return true;
            }
        }
        return false;
    };

    if (!solve(0)) {
        return std::nullopt;
    }
    return assignment;
}

GraphLayoutResult resolve_graph_layout(
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& node_chip_counts,
    const std::set<std::string>& nodes_requiring_distinct_endpoints) {
    // ------------------------------------------------------------------
    // 0. Convert chip tuples to internal representation
    // ------------------------------------------------------------------
    size_t num_submeshes = submesh_chips.size();
    std::vector<std::vector<InternalChip>> chips(num_submeshes);
    for (size_t i = 0; i < num_submeshes; ++i) {
        for (const auto& [mesh_id, chip_id, row, col] : submesh_chips[i]) {
            chips[i].push_back({FabricNodeId{MeshId{mesh_id}, chip_id}, row, col});
        }
    }

    // ------------------------------------------------------------------
    // 1. Discover physical connections between all submesh pairs
    // ------------------------------------------------------------------
    auto connections = discover_connections(chips);

    // ------------------------------------------------------------------
    // 2. Collect unique node names and separate loopback edges
    // ------------------------------------------------------------------
    std::vector<std::string> all_nodes;
    all_nodes.reserve(edges.size() * 2);
    {
        std::set<std::string> seen;
        for (const auto& [src, dst, is_lb] : edges) {
            if (seen.insert(src).second) {
                all_nodes.push_back(src);
            }
            if (seen.insert(dst).second) {
                all_nodes.push_back(dst);
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. Topological sort of non-loopback edges
    // ------------------------------------------------------------------
    auto stage_order = topological_sort(all_nodes, edges);

    // ------------------------------------------------------------------
    // 4. Assign submeshes to nodes via backtracking
    // ------------------------------------------------------------------
    auto node_to_sub = assign_submeshes(
        stage_order, edges, connections, num_submeshes, node_chip_counts, chips, nodes_requiring_distinct_endpoints);

    // ------------------------------------------------------------------
    // 5. Resolve physical coords for every edge
    // ------------------------------------------------------------------
    auto link_assignment =
        assign_edge_links(edges, node_to_sub, connections, chips, nodes_requiring_distinct_endpoints);
    if (!link_assignment.has_value()) {
        throw std::runtime_error(
            "resolve_graph_layout: internal error: selected submesh assignment has no valid physical-link "
            "assignment");
    }
    std::vector<ResolvedEdge> resolved_edges;
    resolved_edges.reserve(edges.size());
    for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
        const auto& [src, dst, is_lb] = edges[edge_idx];
        const auto& link = (*link_assignment)[edge_idx];
        resolved_edges.push_back({src, dst, is_lb, link.exit_row, link.exit_col, link.entry_row, link.entry_col});
    }

    // ------------------------------------------------------------------
    // 6. Locate H2D and D2H coords in stage-0's submesh.
    //
    // Preferred: two chips in stage-0's submesh not used by any edge.
    // Fallback (e.g. small submeshes where all chips are edge-boundary):
    //   H2D = stage-0's forward-exit chip (the chip that sends to stage 1)
    //   D2H = stage-0's loopback-entry chip (the chip that receives the return)
    // ------------------------------------------------------------------
    size_t stage0_sub = node_to_sub.at(stage_order[0]);
    std::set<std::pair<uint32_t, uint32_t>> used_coords;  // (row, col)
    for (const auto& re : resolved_edges) {
        if (node_to_sub.at(re.src) == stage0_sub) {
            used_coords.insert({re.exit_row, re.exit_col});
        }
        if (node_to_sub.at(re.dst) == stage0_sub) {
            used_coords.insert({re.entry_row, re.entry_col});
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> unclaimed;
    unclaimed.reserve(chips[stage0_sub].size());
    for (const auto& [fid, row, col] : chips[stage0_sub]) {
        if (!used_coords.contains({row, col})) {
            unclaimed.push_back({row, col});
        }
    }

    uint32_t h2d_row, h2d_col, d2h_row, d2h_col;
    if (unclaimed.size() >= 2) {
        h2d_row = unclaimed[0].first;
        h2d_col = unclaimed[0].second;
        d2h_row = unclaimed[1].first;
        d2h_col = unclaimed[1].second;
    } else {
        // Fall back: reuse the edge boundary chips of stage 0.
        //   H2D socket sits on the forward-exit chip (stage 0 → stage 1).
        //   D2H socket sits on the loopback-entry chip (last stage → stage 0).
        uint32_t fwd_exit_row = 0, fwd_exit_col = 0;
        uint32_t lb_entry_row = 0, lb_entry_col = 0;
        for (const auto& re : resolved_edges) {
            if (!re.is_loopback && re.src == stage_order[0]) {
                fwd_exit_row = re.exit_row;
                fwd_exit_col = re.exit_col;
            }
            if (re.is_loopback && re.dst == stage_order[0]) {
                lb_entry_row = re.entry_row;
                lb_entry_col = re.entry_col;
            }
        }
        h2d_row = fwd_exit_row;
        h2d_col = fwd_exit_col;
        d2h_row = lb_entry_row;
        d2h_col = lb_entry_col;
    }

    // ------------------------------------------------------------------
    // 7. Build result
    // ------------------------------------------------------------------
    GraphLayoutResult result;
    result.stage_order = std::move(stage_order);
    result.node_to_submesh = std::map<std::string, size_t>(node_to_sub.begin(), node_to_sub.end());
    result.resolved_edges = std::move(resolved_edges);
    result.h2d_entry_row = h2d_row;
    result.h2d_entry_col = h2d_col;
    result.d2h_exit_row = d2h_row;
    result.d2h_exit_col = d2h_col;
    return result;
}

GraphLayoutResult resolve_graph_layout(
    const std::vector<EdgeInputTuple>& edges,
    const std::vector<std::vector<ChipTuple>>& submesh_chips,
    const std::map<std::string, uint32_t>& node_chip_counts) {
    return resolve_graph_layout(edges, submesh_chips, node_chip_counts, {});
}

}  // namespace tt::tt_fabric
