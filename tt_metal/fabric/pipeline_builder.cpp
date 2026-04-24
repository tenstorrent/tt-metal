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

// Physical direct-link info between a pair of submeshes.
// All valid ethernet link pairs are collected so that deconfliction can
// pick an alternative when the first-found pair causes entry == exit on
// the same chip for a forwarding stage.
struct ConnectionInfo {
    struct LinkPair {
        uint32_t exit_row, exit_col;    // chip in submesh i that sends toward j
        uint32_t entry_row, entry_col;  // chip in submesh j that receives from i
    };
    std::vector<LinkPair> links;  // all valid direct ethernet links, first = primary

    // Convenience: whether any link exists (replaces old has-value check).
    bool empty() const { return links.empty(); }

    // Primary link coords (backward-compatible accessors).
    uint32_t exit_row() const { return links[0].exit_row; }
    uint32_t exit_col() const { return links[0].exit_col; }
    uint32_t entry_row() const { return links[0].entry_row; }
    uint32_t entry_col() const { return links[0].entry_col; }
};

using ConnectionKey = std::pair<size_t, size_t>;  // (submesh_i, submesh_j)

/// Discover all direct ethernet links between every ordered pair of submeshes.
/// All valid link pairs are collected (not just the first) to enable deconfliction.
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
    size_t num_submeshes) {
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
            return true;
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
        throw std::runtime_error(
            "resolve_graph_layout: no valid submesh assignment found — "
            "physical connectivity does not match the graph topology");
    }
    return node_to_sub;
}

}  // anonymous namespace

GraphLayoutResult resolve_graph_layout(
    const std::vector<EdgeInputTuple>& edges, const std::vector<std::vector<ChipTuple>>& submesh_chips) {
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
    auto node_to_sub = assign_submeshes(stage_order, edges, connections, num_submeshes);

    // ------------------------------------------------------------------
    // 5. Resolve physical coords for every edge
    // ------------------------------------------------------------------
    std::vector<ResolvedEdge> resolved_edges;
    resolved_edges.reserve(edges.size());
    for (const auto& [src, dst, is_lb] : edges) {
        size_t si = node_to_sub.at(src);
        size_t sj = node_to_sub.at(dst);
        auto it = connections.find({si, sj});
        if (it == connections.end()) {
            throw std::runtime_error(
                "resolve_graph_layout: no direct ethernet link between submesh " + std::to_string(si) + " (" + src +
                ") and submesh " + std::to_string(sj) + " (" + dst + ")");
        }
        const auto& c = it->second;
        resolved_edges.push_back({src, dst, is_lb, c.exit_row(), c.exit_col(), c.entry_row(), c.entry_col()});
    }

    // ------------------------------------------------------------------
    // 5.5. Deconflict same-chip entry/exit for forwarding stages.
    //
    // A forwarding stage i has both an entry chip (where data arrives from
    // stage i-1) and an exit chip (where data leaves to stage i+1).  If
    // the topology resolver assigned the same physical chip to both roles,
    // two persistent BRISC kernels would be dispatched to the same core,
    // causing the second generic_op to block forever.
    //
    // When this happens, scan the full list of valid ethernet links for the
    // exit edge and pick an alternative link whose exit chip differs from
    // the entry chip.  The corresponding entry chip on the next stage is
    // updated in the same step (they are a physically connected pair).
    // ------------------------------------------------------------------
    for (size_t i = 1; i < stage_order.size(); ++i) {
        size_t curr_sub = node_to_sub.at(stage_order[i]);

        // Find the resolved entry edge for this stage (non-loopback, dst == stage_order[i]).
        uint32_t entry_row = UINT32_MAX, entry_col = UINT32_MAX;
        for (const auto& re : resolved_edges) {
            if (!re.is_loopback && re.dst == stage_order[i]) {
                entry_row = re.entry_row;
                entry_col = re.entry_col;
                break;
            }
        }
        if (entry_row == UINT32_MAX) {
            continue;  // stage 0 — no entry edge
        }

        // Find the resolved exit edge for this stage (src == stage_order[i], any kind).
        ResolvedEdge* exit_re = nullptr;
        for (auto& re : resolved_edges) {
            if (re.src == stage_order[i]) {
                exit_re = &re;
                break;
            }
        }
        if (!exit_re) {
            continue;  // no exit edge (shouldn't happen in a pipeline)
        }

        if (exit_re->exit_row == entry_row && exit_re->exit_col == entry_col) {
            // Conflict: find an alternative link for the exit edge.
            size_t next_sub = node_to_sub.at(exit_re->dst);
            const auto& exit_links = connections.at({curr_sub, next_sub}).links;
            bool resolved = false;
            for (const auto& lp : exit_links) {
                if (lp.exit_row != entry_row || lp.exit_col != entry_col) {
                    exit_re->exit_row = lp.exit_row;
                    exit_re->exit_col = lp.exit_col;
                    exit_re->entry_row = lp.entry_row;
                    exit_re->entry_col = lp.entry_col;
                    resolved = true;
                    break;
                }
            }
            if (!resolved) {
                // No alternative exit link — try changing the entry edge instead.
                size_t prev_sub = node_to_sub.at(stage_order[i - 1]);
                const auto& entry_links = connections.at({prev_sub, curr_sub}).links;
                ResolvedEdge* entry_re = nullptr;
                for (auto& re : resolved_edges) {
                    if (!re.is_loopback && re.dst == stage_order[i]) {
                        entry_re = &re;
                        break;
                    }
                }
                for (const auto& lp : entry_links) {
                    if (lp.entry_row != exit_re->exit_row || lp.entry_col != exit_re->exit_col) {
                        entry_re->exit_row = lp.exit_row;
                        entry_re->exit_col = lp.exit_col;
                        entry_re->entry_row = lp.entry_row;
                        entry_re->entry_col = lp.entry_col;
                        resolved = true;
                        break;
                    }
                }
                if (!resolved) {
                    throw std::runtime_error(
                        "resolve_graph_layout: stage " + std::to_string(i) + " (" + stage_order[i] +
                        ") has only one chip at both the entry and exit "
                        "boundary — cannot deconflict entry/exit on the same chip");
                }
            }
        }
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

}  // namespace tt::tt_fabric
